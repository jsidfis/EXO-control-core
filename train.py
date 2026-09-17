import argparse
import json
import os
import sys
import time
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from models.model_factory import create_model
from models.feature_blocks import DifferenceFeatureBlock

ANGLE_COLS = [1, 2, 5, 6]
PRESSURE_COLS = [9, 10]
OUTPUT_COLS = [3, 4, 7, 8]


class ContactWeightedMSE(nn.Module):
    def __init__(self, zero_weight=2.0, nonzero_weight=1.0, knee_weight=1.0):
        super().__init__()
        self.zero_weight = float(zero_weight)
        self.nonzero_weight = float(nonzero_weight)
        self.knee_weight = float(knee_weight)

    def forward(self, pred, target, contact_mask):
        if contact_mask is None:
            return torch.mean((pred - target) ** 2)

        per_elem_loss = (pred - target) ** 2

        seg_weight = torch.where(
            contact_mask,
            torch.full_like(contact_mask.float(), self.nonzero_weight),
            torch.full_like(contact_mask.float(), self.zero_weight),
        )

        joint_weight = torch.ones(pred.shape[-1], device=pred.device)
        if pred.shape[-1] >= 4:
            joint_weight[2] = self.knee_weight
            joint_weight[3] = self.knee_weight

        weight = seg_weight.unsqueeze(-1) * joint_weight.unsqueeze(0)

        return torch.mean(per_elem_loss * weight)


class ExoAblationDataset(Dataset):
    def __init__(self, input_data, output_data, window=180, stride=30,
                 raw_pressure=None, pressure_threshold=1e-3,
                 return_contact_mask=False):
        self.input_data = torch.from_numpy(input_data).float() if isinstance(input_data, np.ndarray) else input_data.float()
        self.output_data = torch.from_numpy(output_data).float() if isinstance(output_data, np.ndarray) else output_data.float()
        self.window = window
        self.stride = stride
        self.num_samples = (len(self.input_data) - window) // stride + 1

        self.raw_pressure = None
        if raw_pressure is not None:
            self.raw_pressure = torch.from_numpy(raw_pressure).float() if isinstance(raw_pressure, np.ndarray) else raw_pressure.float()

        self.pressure_threshold = pressure_threshold
        self.return_contact_mask = return_contact_mask

        if self.return_contact_mask and self.raw_pressure is None:
            raise ValueError("return_contact_mask=True requires raw_pressure.")

        if self.return_contact_mask:
            self.contact_mask_cache = self._build_contact_mask_cache()
        else:
            self.contact_mask_cache = None

    def _build_contact_mask_cache(self):
        masks = []
        for idx in range(self.num_samples):
            start = idx * self.stride
            end = start + self.window
            p_last = self.raw_pressure[end - 1]
            is_nonzero = p_last.abs().sum() > self.pressure_threshold
            masks.append(bool(is_nonzero))
        return torch.tensor(masks, dtype=torch.bool)

    def get_contact_mask(self):
        if self.contact_mask_cache is None:
            raise ValueError("contact_mask_cache is not available.")
        return self.contact_mask_cache.clone()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window
        x = self.input_data[start:end]
        y = self.output_data[end - 1]

        if self.return_contact_mask:
            contact_mask = self.contact_mask_cache[idx]
            return x, y, contact_mask

        return x, y


class DataScaler:
    def __init__(self, method='zscore'):
        self.method = method
        self.mean = None
        self.std = None

    def fit(self, data):
        data = np.array(data, dtype=np.float64)
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-8

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def load_csv_data(csv_path, angle_cols, pressure_cols, output_cols):
    try:
        with open(csv_path, 'r') as f:
            first_line = f.readline()
            f.seek(0)
            delimiter = ',' if ',' in first_line else '\t'
            data = np.loadtxt(f, delimiter=delimiter, skiprows=1)

        angle_data = data[:, angle_cols]
        pressure_data = data[:, pressure_cols]
        output_data = data[:, output_cols]
        return angle_data, pressure_data, output_data
    except Exception as e:
        print(f"[WARNING] load CSV failed: {csv_path}, error: {e}")
        return None, None, None


def load_multiple_csvs(txt_path, angle_cols=ANGLE_COLS,
                       pressure_cols=PRESSURE_COLS, output_cols=OUTPUT_COLS):
    with open(txt_path, 'r') as f:
        csv_paths = [line.strip() for line in f if line.strip()]

    print(f"[INFO] Found {len(csv_paths)} CSV files")

    all_angles, all_pressures, all_outputs = [], [], []
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            continue
        angle, pressure, output = load_csv_data(csv_path, angle_cols, pressure_cols, output_cols)
        if angle is not None:
            all_angles.append(angle)
            all_pressures.append(pressure)
            all_outputs.append(output)

    concat_angles = np.concatenate(all_angles, axis=0)
    concat_pressures = np.concatenate(all_pressures, axis=0)
    concat_outputs = np.concatenate(all_outputs, axis=0)

    print(f"[INFO] Loaded: angles {concat_angles.shape}, pressures {concat_pressures.shape}, outputs {concat_outputs.shape}")
    return concat_angles, concat_pressures, concat_outputs, all_angles, all_pressures, all_outputs


def clip_data(angles, pressures, outputs):
    angles = angles.copy()
    pressures = pressures.copy()
    outputs = outputs.copy()
    pressures = np.clip(pressures, 0, 1500)
    outputs[:, 0:2] = np.clip(outputs[:, 0:2], -15, 15)
    outputs[:, 2:4] = np.clip(outputs[:, 2:4], -10, 20)
    return angles, pressures, outputs


VALID_DIFF_METHODS = {'raw', 'savgol', 'gaussian', 'butterworth',
                      'savgol_causal', 'gaussian_causal', 'butterworth_causal'}
CAUSAL_DIFF_METHODS = {'savgol_causal', 'gaussian_causal', 'butterworth_causal'}
NONCAUSAL_DIFF_METHODS = {'savgol', 'gaussian', 'butterworth'}


def build_features(angle_norm, pressure_norm, use_diff, use_ddiff, use_pressure,
                   diff_method='raw', savgol_window=11, savgol_polyorder=3, dt=0.005,
                   butter_cutoff=10.0, butter_order=4, gauss_sigma=3.0):
    if diff_method not in VALID_DIFF_METHODS:
        raise ValueError(
            f"Unknown diff_method='{diff_method}'. "
            f"Must be one of {sorted(VALID_DIFF_METHODS)}"
        )

    features = [angle_norm]

    if use_diff or use_ddiff:
        if diff_method == 'savgol':
            from scipy.signal import savgol_filter
            angle_base = savgol_filter(
                angle_norm, window_length=savgol_window,
                polyorder=savgol_polyorder, axis=0, mode='interp'
            ).astype(np.float32)

        elif diff_method == 'savgol_causal':
            from scipy.signal import savgol_coeffs, lfilter
            c_smooth = savgol_coeffs(savgol_window, savgol_polyorder, deriv=0, use='dot')
            b_smooth = c_smooth[::-1].copy()
            a_denom = np.array([1.0])
            angle_base = np.zeros_like(angle_norm, dtype=np.float64)
            for col in range(angle_norm.shape[1]):
                angle_base[:, col] = lfilter(b_smooth, a_denom, angle_norm[:, col].astype(np.float64))
            angle_base = angle_base.astype(np.float32)

        elif diff_method == 'butterworth':
            from scipy.signal import butter, filtfilt
            fs = 1.0 / dt
            nyq = fs / 2.0
            normalized_cutoff = min(butter_cutoff / nyq, 0.99)
            b_filt, a_filt = butter(butter_order, normalized_cutoff, btype='low')
            angle_base = np.zeros_like(angle_norm, dtype=np.float64)
            for col in range(angle_norm.shape[1]):
                angle_base[:, col] = filtfilt(b_filt, a_filt, angle_norm[:, col].astype(np.float64))
            angle_base = angle_base.astype(np.float32)

        elif diff_method == 'butterworth_causal':
            from scipy.signal import butter, lfilter
            fs = 1.0 / dt
            nyq = fs / 2.0
            normalized_cutoff = min(butter_cutoff / nyq, 0.99)
            b_filt, a_filt = butter(butter_order, normalized_cutoff, btype='low')
            angle_base = np.zeros_like(angle_norm, dtype=np.float64)
            for col in range(angle_norm.shape[1]):
                angle_base[:, col] = lfilter(b_filt, a_filt, angle_norm[:, col].astype(np.float64))
            angle_base = angle_base.astype(np.float32)

        elif diff_method == 'gaussian':
            from scipy.ndimage import gaussian_filter1d
            angle_base = np.zeros_like(angle_norm, dtype=np.float64)
            for col in range(angle_norm.shape[1]):
                angle_base[:, col] = gaussian_filter1d(angle_norm[:, col].astype(np.float64), sigma=gauss_sigma)
            angle_base = angle_base.astype(np.float32)

        elif diff_method == 'gaussian_causal':
            from scipy.signal import lfilter
            trunc = int(gauss_sigma * 4)
            x_idx = np.arange(trunc + 1)
            kernel = np.exp(-0.5 * (x_idx / gauss_sigma) ** 2)
            kernel = kernel / kernel.sum()
            b_gauss = kernel.astype(np.float64)
            a_denom_g = np.array([1.0])
            angle_base = np.zeros_like(angle_norm, dtype=np.float64)
            for col in range(angle_norm.shape[1]):
                angle_base[:, col] = lfilter(b_gauss, a_denom_g, angle_norm[:, col].astype(np.float64))
            angle_base = angle_base.astype(np.float32)

        else:
            angle_base = angle_norm

        if diff_method in NONCAUSAL_DIFF_METHODS:
            dx = np.gradient(angle_base, dt, axis=0)
        elif diff_method in CAUSAL_DIFF_METHODS:
            dx = np.zeros_like(angle_base)
            dx[1:, :] = (angle_base[1:, :] - angle_base[:-1, :]) / dt
        else:
            dx = np.zeros_like(angle_norm)
            dx[1:, :] = (angle_norm[1:, :] - angle_norm[:-1, :]) / dt

        if use_diff:
            features.append(dx.astype(np.float32))
        if use_ddiff:
            if diff_method in NONCAUSAL_DIFF_METHODS:
                ddx = np.gradient(dx, dt, axis=0)
            elif diff_method in CAUSAL_DIFF_METHODS:
                ddx = np.zeros_like(angle_base)
                ddx[1:, :] = (dx[1:, :] - dx[:-1, :]) / dt
            else:
                ddx = np.zeros_like(angle_norm)
                ddx[1:, :] = (dx[1:, :] - dx[:-1, :]) / dt
            features.append(ddx.astype(np.float32))

    if use_pressure:
        features.append(pressure_norm)

    return np.concatenate(features, axis=-1)


def build_features_per_csv(angle_list, pressure_list, use_diff, use_ddiff, use_pressure,
                           diff_method='raw', savgol_window=11, savgol_polyorder=3, dt=0.005,
                           butter_cutoff=10.0, butter_order=4, gauss_sigma=3.0):
    if diff_method not in VALID_DIFF_METHODS:
        raise ValueError(
            f"Unknown diff_method='{diff_method}'. "
            f"Must be one of {sorted(VALID_DIFF_METHODS)}"
        )

    if diff_method not in CAUSAL_DIFF_METHODS:
        angle_concat = np.concatenate(angle_list, axis=0)
        pressure_concat = np.concatenate(pressure_list, axis=0)
        return build_features(angle_concat, pressure_concat, use_diff, use_ddiff, use_pressure,
                              diff_method=diff_method, savgol_window=savgol_window,
                              savgol_polyorder=savgol_polyorder, dt=dt,
                              butter_cutoff=butter_cutoff, butter_order=butter_order,
                              gauss_sigma=gauss_sigma)

    all_features = []
    for angle_csv, pressure_csv in zip(angle_list, pressure_list):
        feat_csv = build_features(angle_csv, pressure_csv, use_diff, use_ddiff, use_pressure,
                                  diff_method=diff_method, savgol_window=savgol_window,
                                  savgol_polyorder=savgol_polyorder, dt=dt,
                                  butter_cutoff=butter_cutoff, butter_order=butter_order,
                                  gauss_sigma=gauss_sigma)
        all_features.append(feat_csv)

    return np.concatenate(all_features, axis=0)


def apply_label_shift(input_features, output_features, raw_pressure, label_shift):
    if label_shift == 0:
        return input_features, output_features, raw_pressure

    if label_shift > 0:
        inp = input_features[:-label_shift]
        out = output_features[label_shift:]
        rp = raw_pressure[label_shift:] if raw_pressure is not None else None
    else:
        k = -label_shift
        inp = input_features[k:]
        out = output_features[:-k]
        rp = raw_pressure[:-k] if raw_pressure is not None else None

    return inp, out, rp


def compute_input_dim(use_pressure, use_diff, use_ddiff):
    dim = 4
    if use_diff:
        dim += 4
    if use_ddiff:
        dim += 4
    if use_pressure:
        dim += 2
    return dim


def train_one_epoch(model, train_loader, criterion, optimizer, device, has_phase,
                    scaler=None, use_contact_loss=False, model_uses_contact_mask=False):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        if len(batch) == 3:
            x, y, contact_mask = batch
            contact_mask = contact_mask.to(device, non_blocking=True)
        else:
            x, y = batch[0], batch[1]
            contact_mask = None

        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if model_uses_contact_mask:
            pred = model(x, contact_mask=contact_mask)
        elif has_phase:
            pred, _ = model(x)
        else:
            pred = model(x)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                if use_contact_loss and contact_mask is not None:
                    loss = criterion(pred, y, contact_mask)
                else:
                    loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if use_contact_loss and contact_mask is not None:
                loss = criterion(pred, y, contact_mask)
            else:
                loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate_loss(model, val_loader, criterion, device, has_phase,
                  scaler=None, use_contact_loss=False, model_uses_contact_mask=False):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                x, y, contact_mask = batch
                contact_mask = contact_mask.to(device, non_blocking=True)
            else:
                x, y = batch[0], batch[1]
                contact_mask = None

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if model_uses_contact_mask:
                pred = model(x, contact_mask=contact_mask)
            elif has_phase:
                pred, _ = model(x)
            else:
                pred = model(x)

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    if use_contact_loss and contact_mask is not None:
                        loss = criterion(pred, y, contact_mask)
                    else:
                        loss = criterion(pred, y)
            else:
                if use_contact_loss and contact_mask is not None:
                    loss = criterion(pred, y, contact_mask)
                else:
                    loss = criterion(pred, y)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--output_base', type=str, default='results')
    parser.add_argument('--file_local', action='store_true',
                        help='Use per-CSV processing: no cross-CSV filtering/diff/windowing')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    experiment_name = config['experiment_name']
    output_dir = os.path.join(args.output_base, experiment_name, f'seed_{args.seed}')
    os.makedirs(output_dir, exist_ok=True)

    config['seed'] = args.seed
    config['device'] = args.device
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    use_amp = args.amp and device.type == 'cuda'
    print(f"[INFO] Experiment: {experiment_name}, Seed: {args.seed}, Device: {device}, AMP: {use_amp}")

    use_pressure = config.get('use_pressure', True)
    use_diff = config.get('use_diff', False)
    use_ddiff = config.get('use_ddiff', False)
    input_dim = config.get('input_dim', compute_input_dim(use_pressure, use_diff, use_ddiff))
    output_dim = config.get('output_dim', 4)
    window = config.get('window', 180)
    stride = config.get('stride', 30)
    batch_size = config.get('batch_size', 128)
    lr = config.get('lr', 1e-4)
    max_epochs = config.get('max_epochs', 100)
    patience = config.get('patience', 10)
    scaler_method = config.get('scaler_method', 'zscore')
    clip_data_flag = config.get('clip_data', True)

    return_contact_mask = config.get('return_contact_mask', False)
    pressure_threshold = config.get('pressure_threshold', 1e-3)
    model_uses_contact_mask = config.get('model_uses_contact_mask', False)
    label_shift = config.get('label_shift', 0)
    diff_method = config.get('diff_method', 'raw')
    savgol_window = config.get('savgol_window', 11)
    savgol_polyorder = config.get('savgol_polyorder', 3)
    dt = config.get('dt', 0.005)
    butter_cutoff = config.get('butter_cutoff', 10.0)
    butter_order = config.get('butter_order', 4)
    gauss_sigma = config.get('gauss_sigma', 3.0)

    print(f"[INFO] use_pressure={use_pressure}, use_diff={use_diff}, use_ddiff={use_ddiff}, input_dim={input_dim}", flush=True)
    print(f"[INFO] return_contact_mask={return_contact_mask}, model_uses_contact_mask={model_uses_contact_mask}, label_shift={label_shift}", flush=True)
    print(f"[INFO] diff_method={diff_method}", flush=True)

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data_cache_new' if args.output_base != 'results' else 'data_cache')

    if args.file_local:
        # === Per-CSV pipeline: no cross-CSV filtering/diff/windowing ===
        from file_local_data import load_and_process_per_csv, DataScaler as FLScaler, FileLocalDataset

        print("[INFO] Using FILE-LOCAL per-CSV pipeline", flush=True)

        train_feat, train_out, train_pres, train_stats, train_skipped = load_and_process_per_csv(
            args.train_path, window=window, stride=stride,
            use_diff=use_diff, use_ddiff=use_ddiff, use_pressure=use_pressure,
            diff_method=diff_method, savgol_window=savgol_window,
            savgol_polyorder=savgol_polyorder, dt=dt,
            butter_cutoff=butter_cutoff, butter_order=butter_order,
            gauss_sigma=gauss_sigma, clip_data_flag=clip_data_flag)

        val_feat, val_out, val_pres, val_stats, val_skipped = load_and_process_per_csv(
            args.val_path, window=window, stride=stride,
            use_diff=use_diff, use_ddiff=use_ddiff, use_pressure=use_pressure,
            diff_method=diff_method, savgol_window=savgol_window,
            savgol_polyorder=savgol_polyorder, dt=dt,
            butter_cutoff=butter_cutoff, butter_order=butter_order,
            gauss_sigma=gauss_sigma, clip_data_flag=clip_data_flag)

        print(f"[INFO] Train: {len(train_feat)} windows from {len(train_stats)} files, {len(train_skipped)} skipped", flush=True)
        print(f"[INFO] Val: {len(val_feat)} windows from {len(val_stats)} files, {len(val_skipped)} skipped", flush=True)

        for sf in train_skipped:
            print(f"  [SKIP] Train: {sf[0]} ({sf[1]} frames < {window})")
        for sf in val_skipped:
            print(f"  [SKIP] Val: {sf[0]} ({sf[1]} frames < {window})")

        # Fit scaler on training features (all frames)
        train_input_scaler = DataScaler(method=scaler_method)
        train_output_scaler = DataScaler(method=scaler_method)
        all_train_feature_frames = train_feat.reshape(-1, train_feat.shape[-1])
        train_input_scaler.fit(all_train_feature_frames)
        train_output_scaler.fit(train_out)

        # Normalize
        train_feat_flat = train_feat.reshape(-1, train_feat.shape[-1])
        train_feat_norm_flat = train_input_scaler.transform(train_feat_flat).astype(np.float32)
        train_input_norm = train_feat_norm_flat.reshape(train_feat.shape)
        train_output_norm = train_output_scaler.transform(train_out).astype(np.float32)

        val_feat_flat = val_feat.reshape(-1, val_feat.shape[-1])
        val_feat_norm_flat = train_input_scaler.transform(val_feat_flat).astype(np.float32)
        val_input_norm = val_feat_norm_flat.reshape(val_feat.shape)
        val_output_norm = train_output_scaler.transform(val_out).astype(np.float32)

        train_pressure_raw = train_pres
        val_pressure_raw = val_pres

        # Save window stats
        with open(os.path.join(output_dir, 'window_stats_train.json'), 'w') as f:
            json.dump({'files': train_stats, 'skipped': [{'file': s[0], 'frames': s[1]} for s in train_skipped],
                       'total_windows': len(train_feat)}, f, indent=2)
        with open(os.path.join(output_dir, 'window_stats_val.json'), 'w') as f:
            json.dump({'files': val_stats, 'skipped': [{'file': s[0], 'frames': s[1]} for s in val_skipped],
                       'total_windows': len(val_feat)}, f, indent=2)

        num_train = len(train_feat)
        num_val = len(val_feat)

        train_dataset = FileLocalDataset(
            train_input_norm, train_output_norm,
            raw_pressure=train_pressure_raw if return_contact_mask else None,
            pressure_threshold=pressure_threshold,
            return_contact_mask=return_contact_mask,
        )
        val_dataset = FileLocalDataset(
            val_input_norm, val_output_norm,
            raw_pressure=val_pressure_raw if return_contact_mask else None,
            pressure_threshold=pressure_threshold,
            return_contact_mask=return_contact_mask,
        )
        print(f"[INFO] Train samples: {num_train}, Val samples: {num_val}", flush=True)

    else:
        # === Original pipeline: concatenate then process ===
        train_csv_angles = None
        val_csv_angles = None
        # Causal methods need per-CSV data for filter state reset at CSV boundaries
        skip_cache = diff_method in CAUSAL_DIFF_METHODS
        if not skip_cache and os.path.exists(os.path.join(cache_dir, 'train_angles.npy')):
            print("[INFO] Loading from cache...", flush=True)
            train_angles = np.load(os.path.join(cache_dir, 'train_angles.npy'))
            train_pressures = np.load(os.path.join(cache_dir, 'train_pressures.npy'))
            train_outputs = np.load(os.path.join(cache_dir, 'train_outputs.npy'))
            val_angles = np.load(os.path.join(cache_dir, 'val_angles.npy'))
            val_pressures = np.load(os.path.join(cache_dir, 'val_pressures.npy'))
            val_outputs = np.load(os.path.join(cache_dir, 'val_outputs.npy'))
            print(f"[INFO] Cache loaded: train {train_angles.shape}, val {val_angles.shape}", flush=True)
        else:
            if skip_cache:
                print(f"[INFO] Skipping cache for causal method: {diff_method}", flush=True)
            train_angles, train_pressures, train_outputs, train_csv_angles, train_csv_pressures, train_csv_outputs = load_multiple_csvs(args.train_path)
            val_angles, val_pressures, val_outputs, val_csv_angles, val_csv_pressures, val_csv_outputs = load_multiple_csvs(args.val_path)

            if clip_data_flag:
                train_angles, train_pressures, train_outputs = clip_data(train_angles, train_pressures, train_outputs)
                val_angles, val_pressures, val_outputs = clip_data(val_angles, val_pressures, val_outputs)
                if train_csv_angles is not None:
                    train_csv_angles = [clip_data(a, p, o)[0] for a, p, o in zip(train_csv_angles, train_csv_pressures, train_csv_outputs)]
                    train_csv_pressures = [clip_data(a, p, o)[1] for a, p, o in zip(train_csv_angles, train_csv_pressures, train_csv_outputs)]
                    val_csv_angles = [clip_data(a, p, o)[0] for a, p, o in zip(val_csv_angles, val_csv_pressures, val_csv_outputs)]
                    val_csv_pressures = [clip_data(a, p, o)[1] for a, p, o in zip(val_csv_angles, val_csv_pressures, val_csv_outputs)]

        train_pressure_raw = train_pressures.copy()
        val_pressure_raw = val_pressures.copy()

        train_input_scaler = DataScaler(method=scaler_method)
        train_output_scaler = DataScaler(method=scaler_method)

        if diff_method in CAUSAL_DIFF_METHODS and train_csv_angles is not None:
            print(f"[INFO] Using per-CSV feature building for causal method: {diff_method}", flush=True)
            train_input_raw = build_features_per_csv(
                train_csv_angles, train_csv_pressures, use_diff, use_ddiff, use_pressure,
                diff_method=diff_method, savgol_window=savgol_window,
                savgol_polyorder=savgol_polyorder, dt=dt,
                butter_cutoff=butter_cutoff, butter_order=butter_order,
                gauss_sigma=gauss_sigma)
            val_input_raw = build_features_per_csv(
                val_csv_angles, val_csv_pressures, use_diff, use_ddiff, use_pressure,
                diff_method=diff_method, savgol_window=savgol_window,
                savgol_polyorder=savgol_polyorder, dt=dt,
                butter_cutoff=butter_cutoff, butter_order=butter_order,
                gauss_sigma=gauss_sigma)
        else:
            train_input_raw = build_features(train_angles, train_pressures, use_diff, use_ddiff, use_pressure,
                                             diff_method=diff_method, savgol_window=savgol_window,
                                             savgol_polyorder=savgol_polyorder, dt=dt,
                                             butter_cutoff=butter_cutoff, butter_order=butter_order,
                                             gauss_sigma=gauss_sigma)
            val_input_raw = build_features(val_angles, val_pressures, use_diff, use_ddiff, use_pressure,
                                           diff_method=diff_method, savgol_window=savgol_window,
                                           savgol_polyorder=savgol_polyorder, dt=dt,
                                           butter_cutoff=butter_cutoff, butter_order=butter_order,
                                           gauss_sigma=gauss_sigma)

        train_input_scaler.fit(train_input_raw)
        train_output_scaler.fit(train_outputs)

        train_input_norm = train_input_scaler.transform(train_input_raw).astype(np.float32)
        train_output_norm = train_output_scaler.transform(train_outputs).astype(np.float32)
        val_input_norm = train_input_scaler.transform(val_input_raw).astype(np.float32)
        val_output_norm = train_output_scaler.transform(val_outputs).astype(np.float32)

        del train_input_raw, val_input_raw

        if label_shift != 0:
            train_input_norm, train_output_norm, train_pressure_raw = apply_label_shift(
                train_input_norm, train_output_norm, train_pressure_raw, label_shift)
            val_input_norm, val_output_norm, val_pressure_raw = apply_label_shift(
                val_input_norm, val_output_norm, val_pressure_raw, label_shift)
            print(f"[INFO] Applied label_shift={label_shift}: input {train_input_norm.shape}, output {train_output_norm.shape}", flush=True)

        train_dataset = ExoAblationDataset(
            train_input_norm, train_output_norm, window, stride,
            raw_pressure=train_pressure_raw if return_contact_mask else None,
            pressure_threshold=pressure_threshold,
            return_contact_mask=return_contact_mask,
        )
        val_dataset = ExoAblationDataset(
            val_input_norm, val_output_norm, window, stride,
            raw_pressure=val_pressure_raw if return_contact_mask else None,
            pressure_threshold=pressure_threshold,
            return_contact_mask=return_contact_mask,
        )

        num_train = len(train_dataset)
        num_val = len(val_dataset)
        print(f"[INFO] Train samples: {num_train}, Val samples: {num_val}", flush=True)

    sampler_type = config.get('sampler_type', 'default')
    if sampler_type == 'zero_balanced' and return_contact_mask:
        contact_mask = train_dataset.get_contact_mask()
        zero_mask = ~contact_mask
        nonzero_mask = contact_mask
        n_zero = int(zero_mask.sum().item())
        n_nonzero = int(nonzero_mask.sum().item())
        print(f"[INFO] Zero-balanced sampler: n_zero={n_zero}, n_nonzero={n_nonzero}", flush=True)

        if n_zero > 0 and n_nonzero > 0:
            sample_weights = torch.zeros(len(contact_mask), dtype=torch.float)
            sample_weights[zero_mask] = 0.5 / n_zero
            sample_weights[nonzero_mask] = 0.5 / n_nonzero
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                      sampler=train_sampler, pin_memory=True, num_workers=0)
        else:
            print("[WARNING] Invalid segment counts, falling back to default sampler")
            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, pin_memory=True, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, pin_memory=True, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, pin_memory=True, num_workers=0)

    model = create_model(config['model_name'], config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model: {config['model_name']}, Params: {num_params}")

    has_phase = config['model_name'] in ('phase_tcn_ssm_last', 'phase_parallel_tcn_ssm_last')

    loss_type = config.get('loss_type', 'mse')
    if loss_type == 'contact_weighted_mse':
        criterion = ContactWeightedMSE(
            zero_weight=config.get('loss_zero_weight', 2.0),
            nonzero_weight=config.get('loss_nonzero_weight', 1.0),
            knee_weight=config.get('loss_knee_weight', 1.0),
        )
        use_contact_loss = True
    else:
        criterion = nn.MSELoss()
        use_contact_loss = False

    optimizer_name = config.get('optimizer', 'Adam')
    weight_decay = config.get('weight_decay', 0.0)
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    scheduler_name = config.get('scheduler', 'reduce_on_plateau')
    if scheduler_name == 'cosine_warm_restarts':
        T_0 = config.get('cosine_T0', 20)
        T_mult = config.get('cosine_T_mult', 2)
        eta_min = config.get('cosine_eta_min', 1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

    grad_scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    train_log = []
    start_epoch = 1

    if args.resume:
        ckpt_path = os.path.join(output_dir, 'best_model.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            best_val_loss = ckpt.get('val_loss', float('inf'))
            best_epoch = ckpt.get('epoch', 0)
            start_epoch = best_epoch + 1
            print(f"[INFO] Resumed from epoch {best_epoch}, val_loss={best_val_loss:.6f}")

    start_time = time.time()
    for epoch in range(start_epoch, max_epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                     has_phase, grad_scaler, use_contact_loss, model_uses_contact_mask)
        val_loss = evaluate_loss(model, val_loader, criterion, device,
                                has_phase, grad_scaler, use_contact_loss, model_uses_contact_mask)
        epoch_time = time.time() - epoch_start

        if scheduler_name == 'cosine_warm_restarts':
            scheduler.step()
        else:
            scheduler.step(val_loss)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        train_log.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time': round(epoch_time, 2)
        })

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'config': config,
            }, os.path.join(output_dir, 'best_model.pt'))
            np.savez(os.path.join(args.output_base, experiment_name, 'scaler.npz'),
                     input_mean=train_input_scaler.mean,
                     input_std=train_input_scaler.std,
                     output_mean=train_output_scaler.mean,
                     output_std=train_output_scaler.std)
            print(f"  -> Saved best model (val_loss={val_loss:.6f})", flush=True)
        else:
            epochs_no_improve += 1

        log_path = os.path.join(output_dir, 'train_log.csv')
        with open(log_path, 'w') as f:
            f.write('epoch,train_loss,val_loss,lr,epoch_time\n')
            for entry in train_log:
                f.write(f"{entry['epoch']},{entry['train_loss']},{entry['val_loss']},{entry['lr']},{entry['epoch_time']}\n")

        if epochs_no_improve >= patience:
            print(f"[INFO] Early stopping at epoch {epoch}, best epoch: {best_epoch}", flush=True)
            break

    elapsed = time.time() - start_time
    print(f"[INFO] Training completed in {elapsed:.1f}s, best epoch: {best_epoch}, best val_loss: {best_val_loss:.6f}", flush=True)

    try:
        checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"[WARNING] Failed to load best model: {e}", flush=True)

    try:
        all_preds, all_targets, all_contact_masks = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    x, y, contact_mask = batch
                    all_contact_masks.append(contact_mask.numpy())
                else:
                    x, y = batch[0], batch[1]

                x = x.to(device)
                if model_uses_contact_mask:
                    cm = contact_mask.to(device) if len(batch) == 3 else None
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            pred = model(x, contact_mask=cm)
                    else:
                        pred = model(x, contact_mask=cm)
                elif has_phase:
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            pred, _ = model(x)
                    else:
                        pred, _ = model(x)
                else:
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            pred = model(x)
                    else:
                        pred = model(x)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        pred_denorm = train_output_scaler.inverse_transform(all_preds)
        target_denorm = train_output_scaler.inverse_transform(all_targets)

        val_pressure_for_seg = val_pressure_raw
        num_windows = num_val
        window_size = window
        stride_val = stride

        pressure_energies = []
        if args.file_local:
            # val_pressure_raw is already window-level (num_windows, 2)
            for i in range(num_windows):
                p = val_pressure_for_seg[i]
                energy = np.mean(np.abs(p[0])) + np.mean(np.abs(p[1])) if p.ndim > 0 else float(np.abs(p[0]) + np.abs(p[1]))
                pressure_energies.append(energy)
        else:
            for i in range(num_windows):
                start_idx = i * stride_val
                end_idx = start_idx + window_size
                if end_idx <= len(val_pressure_for_seg):
                    p_window = val_pressure_for_seg[start_idx:end_idx]
                    energy = np.mean(np.abs(p_window[:, 0])) + np.mean(np.abs(p_window[:, 1]))
                else:
                    energy = 0.0
                pressure_energies.append(energy)

        pressure_energies = np.array(pressure_energies)

        np.save(os.path.join(output_dir, 'predictions.npy'), pred_denorm)
        np.save(os.path.join(output_dir, 'targets.npy'), target_denorm)
        np.save(os.path.join(output_dir, 'pressure_energy.npy'), pressure_energies)

        eps = 1e-6
        segment_labels = (pressure_energies >= eps).astype(np.int32)
        np.save(os.path.join(output_dir, 'segment_labels.npy'), segment_labels)

        if all_contact_masks:
            all_contact_masks = np.concatenate(all_contact_masks, axis=0)
            np.save(os.path.join(output_dir, 'contact_masks.npy'), all_contact_masks)

        joint_names = ['hip_l', 'hip_r', 'knee_l', 'knee_r']
        metrics = {
            'experiment': experiment_name,
            'seed': args.seed,
            'input_dim': input_dim,
            'device': args.device,
            'num_params': num_params,
            'best_epoch': best_epoch,
            'val_loss': float(best_val_loss),
            'num_windows': num_windows,
        }

        r2_per_joint = {}
        rmse_per_joint = {}
        mae_per_joint = {}
        for j, name in enumerate(joint_names):
            y_true = target_denorm[:, j]
            y_pred = pred_denorm[:, j]
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mae = np.mean(np.abs(y_true - y_pred))
            r2_per_joint[name] = float(r2)
            rmse_per_joint[name] = float(rmse)
            mae_per_joint[name] = float(mae)

        metrics['avg_r2'] = float(np.mean(list(r2_per_joint.values())))
        metrics['avg_rmse'] = float(np.mean(list(rmse_per_joint.values())))
        metrics['avg_mae'] = float(np.mean(list(mae_per_joint.values())))
        metrics['r2_per_joint'] = r2_per_joint
        metrics['rmse_per_joint'] = rmse_per_joint
        metrics['mae_per_joint'] = mae_per_joint

        metrics['test_loss'] = float(best_val_loss)

        with open(os.path.join(output_dir, 'metrics_all.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        zero_mask = segment_labels == 0
        nonzero_mask = segment_labels == 1

        for seg_name, seg_mask in [('pressure_zero_segments', zero_mask),
                                   ('pressure_nonzero_segments', nonzero_mask)]:
            n_seg = int(seg_mask.sum())
            seg_metrics = {'num_windows': n_seg}

            if n_seg >= 30:
                seg_r2, seg_rmse, seg_mae = {}, {}, {}
                for j, name in enumerate(joint_names):
                    y_true = target_denorm[seg_mask, j]
                    y_pred = pred_denorm[seg_mask, j]
                    ss_res = np.sum((y_true - y_pred) ** 2)
                    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                    r2 = 1 - ss_res / (ss_tot + 1e-10)
                    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                    mae = np.mean(np.abs(y_true - y_pred))
                    seg_r2[name] = float(r2)
                    seg_rmse[name] = float(rmse)
                    seg_mae[name] = float(mae)
                seg_metrics['avg_r2'] = float(np.mean(list(seg_r2.values())))
                seg_metrics['avg_rmse'] = float(np.mean(list(seg_rmse.values())))
                seg_metrics['avg_mae'] = float(np.mean(list(seg_mae.values())))
                seg_metrics['r2_per_joint'] = seg_r2
                seg_metrics['rmse_per_joint'] = seg_rmse
                seg_metrics['mae_per_joint'] = seg_mae
            else:
                seg_metrics['warning'] = 'too few windows for reliable segment metric'

            with open(os.path.join(output_dir, f'metrics_{seg_name}.json'), 'w') as f:
                json.dump(seg_metrics, f, indent=2)

        percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
        pe_percentiles = {f'p{p}': float(np.percentile(pressure_energies, p)) for p in percentiles}
        pe_percentiles['min'] = float(np.min(pressure_energies))
        pe_percentiles['max'] = float(np.max(pressure_energies))
        with open(os.path.join(output_dir, 'pressure_energy_stats.json'), 'w') as f:
            json.dump(pe_percentiles, f, indent=2)

        if has_phase:
            phase_embeddings = []
            model.eval()
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            _, phase = model(x)
                    else:
                        _, phase = model(x)
                    phase_embeddings.append(phase.cpu().numpy())
            phase_embeddings = np.concatenate(phase_embeddings, axis=0)
            np.save(os.path.join(output_dir, 'phase_embeddings.npy'), phase_embeddings)

        print(f"[INFO] Results saved to {output_dir}", flush=True)
        print(f"[INFO] avg_r2={metrics['avg_r2']:.4f}, avg_rmse={metrics['avg_rmse']:.4f}, avg_mae={metrics['avg_mae']:.4f}", flush=True)
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}", flush=True)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
