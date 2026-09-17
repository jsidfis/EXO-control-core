# EXO-control core code

Core model, training, and configuration code for exoskeleton joint torque prediction. The main model combines a temporal convolution branch and a state-space branch with gated fusion. This repository also contains LSTM, GRU, Transformer, TCN-only, SSM-only, and width-matched TCN comparisons.

## Contents

| Path | Purpose |
| --- | --- |
| `train.py` | Training, epoch-level validation, checkpoint selection, and final validation metrics |
| `file_local_data.py` | Filtering, feature construction, and sliding windows independently within each CSV |
| `models/` | Main model, ablation models, and their building blocks |
| `baselines/` | LSTM, GRU, and Transformer baselines |
| `configs/file_local/` | Preprocessing and baseline experiment settings |
| `configs/ablation/` | Single-branch, fusion, and matched-width TCN settings |
| `configs/subject_disjoint/` | Configurations for subject-disjoint training runs |
| `splits/subject_disjoint_split.yaml` | Subject identifiers assigned to train, validation, and test |
| `scripts/make_subject_disjoint_manifests.py` | Builds subject-disjoint CSV path lists from a local source manifest |

## Environment and input

The code was checked with Python 3.12.4, PyTorch 2.3.0, NumPy 1.26.4, SciPy 1.13.1, and PyYAML 6.0.1. Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

The training input is a text manifest with one CSV path per line. Paths are read as written, relative to the working directory or absolute. Each CSV needs a header row and at least 11 numeric columns. The zero-based column indices used by the released pipeline are angles `[1, 2, 5, 6]`, pressure `[9, 10]`, and target torques `[3, 4, 7, 8]`.

Datasets, generated manifests, checkpoints, model weights, and experiment results are not included. Supply your own data in the same column layout. The frozen subject identifiers are included, while the original manifests are omitted because they contain machine-specific absolute paths.

## Train and validate

For a file-local experiment, run from the repository root:

```bash
python train.py \
  --config configs/file_local/fl_z8_savgol.yaml \
  --train_path /path/to/train.txt \
  --val_path /path/to/val.txt \
  --output_base results \
  --device cpu \
  --seed 42 \
  --file_local
```

`--file_local` keeps filtering, differentiation, and window construction within each CSV. The input and target scalers are fit using training data. The trainer evaluates validation loss each epoch, selects the best checkpoint, and writes final validation metrics under `--output_base`. The reported multi-seed experiments used seeds `42`, `2024`, and `3407`; run the same command with each seed to repeat that protocol. Set `--device cuda:0` when a compatible GPU is available.

To create manifests from an existing local CSV path list grouped by subject ID (for example `BT01`), use:

```bash
python scripts/make_subject_disjoint_manifests.py \
  --source-manifest /path/to/all_csvs.txt \
  --split-config splits/subject_disjoint_split.yaml \
  --output-dir local_splits/subject_disjoint
```

The resulting `train.txt` and `val.txt` can be passed to `train.py` with a configuration under `configs/subject_disjoint/`. The independent test analysis scripts and original test outputs are outside this core-code release; the validation metrics produced by `train.py` are not test-set metrics.

## License

The code in this repository is released under the MIT License. This license does not cover datasets or third-party assets that are not included here.
