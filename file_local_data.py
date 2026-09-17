"""
file_local_data.py — Per-CSV data processing pipeline

Ensures filtering, differentiation, and windowing are strictly within each CSV file.
No cross-CSV boundaries for any operation.

Usage:
    from file_local_data import load_and_process_per_csv, FileLocalDataset
"""

import os
import numpy as np
from torch.utils.data import Dataset

ANGLE_COLS = [1, 2, 5, 6]
PRESSURE_COLS = [9, 10]
OUTPUT_COLS = [3, 4, 7, 8]

VALID_DIFF_METHODS = {'raw', 'savgol', 'gaussian', 'butterworth',
                      'savgol_causal', 'gaussian_causal', 'butterworth_causal'}
CAUSAL_DIFF_METHODS = {'savgol_causal', 'gaussian_causal', 'butterworth_causal'}
NONCAUSAL_DIFF_METHODS = {'savgol', 'gaussian', 'butterworth'}


def load_csv_data(csv_path, angle_cols=ANGLE_COLS, pressure_cols=PRESSURE_COLS, output_cols=OUTPUT_COLS):
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


def clip_data(angles, pressures, outputs):
    angles = angles.copy()
    pressures = pressures.copy()
    outputs = outputs.copy()
    pressures = np.clip(pressures, 0, 1500)
    outputs[:, 0:2] = np.clip(outputs[:, 0:2], -15, 15)
    outputs[:, 2:4] = np.clip(outputs[:, 2:4], -10, 20)
    return angles, pressures, outputs


def build_features_single_csv(angle, pressure, use_diff, use_ddiff, use_pressure,
                               diff_method='raw', savgol_window=11, savgol_polyorder=3,
                               dt=0.005, butter_cutoff=10.0, butter_order=4, gauss_sigma=3.0):
    """Build features for a single CSV file. No cross-file contamination."""
    if diff_method not in VALID_DIFF_METHODS:
        raise ValueError(f"Unknown diff_method='{diff_method}'")

    features = [angle]

    if use_diff or use_ddiff:
        if diff_method == 'savgol':
            from scipy.signal import savgol_filter
            angle_base = savgol_filter(
                angle, window_length=savgol_window,
                polyorder=savgol_polyorder, axis=0, mode='interp'
            ).astype(np.float32)
        elif diff_method == 'savgol_causal':
            from scipy.signal import savgol_coeffs, lfilter
            c_smooth = savgol_coeffs(savgol_window, savgol_polyorder, deriv=0, use='dot')
            b_smooth = c_smooth[::-1].copy()
            a_denom = np.array([1.0])
            angle_base = np.zeros_like(angle, dtype=np.float64)
            for col in range(angle.shape[1]):
                angle_base[:, col] = lfilter(b_smooth, a_denom, angle[:, col].astype(np.float64))
            angle_base = angle_base.astype(np.float32)
        elif diff_method == 'butterworth':
            from scipy.signal import butter, filtfilt
            fs = 1.0 / dt
            nyq = fs / 2.0
            normalized_cutoff = min(butter_cutoff / nyq, 0.99)
            b_filt, a_filt = butter(butter_order, normalized_cutoff, btype='low')
            angle_base = np.zeros_like(angle, dtype=np.float64)
            for col in range(angle.shape[1]):
                angle_base[:, col] = filtfilt(b_filt, a_filt, angle[:, col].astype(np.float64))
            angle_base = angle_base.astype(np.float32)
        elif diff_method == 'butterworth_causal':
            from scipy.signal import butter, lfilter
            fs = 1.0 / dt
            nyq = fs / 2.0
            normalized_cutoff = min(butter_cutoff / nyq, 0.99)
            b_filt, a_filt = butter(butter_order, normalized_cutoff, btype='low')
            angle_base = np.zeros_like(angle, dtype=np.float64)
            for col in range(angle.shape[1]):
                angle_base[:, col] = lfilter(b_filt, a_filt, angle[:, col].astype(np.float64))
            angle_base = angle_base.astype(np.float32)
        elif diff_method == 'gaussian':
            from scipy.ndimage import gaussian_filter1d
            angle_base = np.zeros_like(angle, dtype=np.float64)
            for col in range(angle.shape[1]):
                angle_base[:, col] = gaussian_filter1d(angle[:, col].astype(np.float64), sigma=gauss_sigma)
            angle_base = angle_base.astype(np.float32)
        elif diff_method == 'gaussian_causal':
            from scipy.signal import lfilter
            trunc = int(gauss_sigma * 4)
            x_idx = np.arange(trunc + 1)
            kernel = np.exp(-0.5 * (x_idx / gauss_sigma) ** 2)
            kernel = kernel / kernel.sum()
            b_gauss = kernel.astype(np.float64)
            a_denom_g = np.array([1.0])
            angle_base = np.zeros_like(angle, dtype=np.float64)
            for col in range(angle.shape[1]):
                angle_base[:, col] = lfilter(b_gauss, a_denom_g, angle[:, col].astype(np.float64))
            angle_base = angle_base.astype(np.float32)
        else:
            angle_base = angle

        if diff_method in NONCAUSAL_DIFF_METHODS:
            dx = np.gradient(angle_base, dt, axis=0)
        elif diff_method in CAUSAL_DIFF_METHODS:
            dx = np.zeros_like(angle_base)
            dx[1:, :] = (angle_base[1:, :] - angle_base[:-1, :]) / dt
        else:
            dx = np.zeros_like(angle)
            dx[1:, :] = (angle[1:, :] - angle[:-1, :]) / dt

        if use_diff:
            features.append(dx.astype(np.float32))
        if use_ddiff:
            if diff_method in NONCAUSAL_DIFF_METHODS:
                ddx = np.gradient(dx, dt, axis=0)
            elif diff_method in CAUSAL_DIFF_METHODS:
                ddx = np.zeros_like(angle_base)
                ddx[1:, :] = (dx[1:, :] - dx[:-1, :]) / dt
            else:
                ddx = np.zeros_like(angle)
                ddx[1:, :] = (dx[1:, :] - dx[:-1, :]) / dt
            features.append(ddx.astype(np.float32))

    if use_pressure:
        features.append(pressure)

    return np.concatenate(features, axis=-1)


def load_and_process_per_csv(txt_path, window=180, stride=30, use_diff=True, use_ddiff=True,
                              use_pressure=True, diff_method='raw', savgol_window=11,
                              savgol_polyorder=3, dt=0.005, butter_cutoff=10.0,
                              butter_order=4, gauss_sigma=3.0, clip_data_flag=True):
    """
    Load CSVs and process each independently:
    1. Per-CSV: load, clip, build features (filter+diff), window
    2. Skip CSVs shorter than window
    3. Track source_file for each window

    Returns:
        all_inputs: (N, input_dim) concatenated normalized features
        all_outputs: (N, output_dim) concatenated normalized outputs
        all_pressure_raw: (N, 2) raw pressure for segment analysis
        source_files: list of (filename, num_windows) tuples
        skipped_files: list of (filename, length) tuples
        input_scaler_mean, input_scaler_std, output_scaler_mean, output_scaler_std
        window_stats: list of per-file window statistics
    """
    with open(txt_path, 'r') as f:
        csv_paths = [line.strip() for line in f if line.strip()]

    csv_features = []
    csv_outputs_raw = []
    csv_pressures_raw = []
    source_file_indices = []
    window_stats = []
    skipped_files = []
    global_idx = 0

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            continue

        angle, pressure, output = load_csv_data(csv_path)
        if angle is None:
            continue

        if clip_data_flag:
            angle, pressure, output = clip_data(angle, pressure, output)

        n_frames = len(angle)
        if n_frames < window:
            skipped_files.append((os.path.basename(csv_path), n_frames))
            continue

        features = build_features_single_csv(
            angle, pressure, use_diff, use_ddiff, use_pressure,
            diff_method=diff_method, savgol_window=savgol_window,
            savgol_polyorder=savgol_polyorder, dt=dt,
            butter_cutoff=butter_cutoff, butter_order=butter_order,
            gauss_sigma=gauss_sigma
        )

        n_windows = (n_frames - window) // stride + 1
        basename = os.path.basename(csv_path)

        window_stats.append({
            'filename': basename,
            'n_frames': n_frames,
            'n_windows': n_windows,
            'start_global_idx': global_idx,
        })

        for w in range(n_windows):
            start = w * stride
            end = start + window
            csv_features.append(features[start:end])
            csv_outputs_raw.append(output[end - 1])
            csv_pressures_raw.append(pressure[end - 1])

        source_file_indices.extend([global_idx] * n_windows)
        global_idx += 1

    all_features = np.array(csv_features, dtype=np.float32)
    all_outputs = np.array(csv_outputs_raw, dtype=np.float64)
    all_pressure = np.array(csv_pressures_raw, dtype=np.float32)

    return all_features, all_outputs, all_pressure, window_stats, skipped_files


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


class FileLocalDataset(Dataset):
    """Dataset that holds pre-windowed data with source_file tracking."""

    def __init__(self, input_data, output_data, raw_pressure=None, pressure_threshold=1e-3,
                 return_contact_mask=False, source_file_ids=None):
        self.input_data = torch.from_numpy(input_data).float() if isinstance(input_data, np.ndarray) else input_data.float()
        self.output_data = torch.from_numpy(output_data).float() if isinstance(output_data, np.ndarray) else output_data.float()
        self.raw_pressure = None
        if raw_pressure is not None:
            self.raw_pressure = torch.from_numpy(raw_pressure).float() if isinstance(raw_pressure, np.ndarray) else raw_pressure.float()
        self.pressure_threshold = pressure_threshold
        self.return_contact_mask = return_contact_mask
        self.source_file_ids = source_file_ids

        if self.return_contact_mask and self.raw_pressure is not None:
            self.contact_mask_cache = self._build_contact_mask_cache()
        else:
            self.contact_mask_cache = None

    def _build_contact_mask_cache(self):
        is_nonzero = self.raw_pressure.abs().sum(dim=-1) > self.pressure_threshold
        return is_nonzero

    def get_contact_mask(self):
        if self.contact_mask_cache is None:
            raise ValueError("contact_mask_cache not available")
        return self.contact_mask_cache.clone()

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        x = self.input_data[idx]
        y = self.output_data[idx]

        if self.return_contact_mask:
            return x, y, self.contact_mask_cache[idx]
        return x, y


import torch
