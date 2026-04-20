# Abdominal OAR Segmentation (BTCV)

Automatic segmentation of organs-at-risk (OAR) in abdominal CT volumes, built on [MONAI](https://monai.io/). The project targets the **BTCV** (Beyond the Cranial Vault) multi-organ dataset — 30 annotated training volumes and 20 testing volumes, 13 abdominal organs + background — and benchmarks several 3D backbones against the MaskMed paper included in the repo (`maskmed_paper.pdf`).

## Segmented Organs

The class map is the standard BTCV 13-organ labelling plus background. The authoritative list lives in `src/utils/config.py` (`DEFAULT_CLASS_NAMES`) and is used whenever a config does not supply its own `data.class_names`:

| Label | Class                  |
|-------|------------------------|
| 0     | Background             |
| 1     | Spleen                 |
| 2     | Right Kidney           |
| 3     | Left Kidney            |
| 4     | Gallbladder            |
| 5     | Esophagus              |
| 6     | Liver                  |
| 7     | Stomach                |
| 8     | Aorta                  |
| 9     | IVC (Inferior Vena Cava) |
| 10    | Portal & Splenic Veins |
| 11    | Pancreas               |
| 12    | Right Adrenal Gland    |
| 13    | Left Adrenal Gland     |

Verified against `data/raw/RawData/Training/label/*.nii.gz` — every annotated volume contains integer labels in `[0, 13]`, i.e. all 13 organs plus background.

Class count is configurable — `configs/baseline.yaml`, `configs/btcv_raw.yaml`, `configs/swin_unetr.yaml`, and the MaskMed configs use all 14 classes, while `configs/averaged_binary.yaml` collapses every organ into a single foreground class (`num_classes: 2`, `class_names: ["Background", "Foreground"]`). To train on a different subset, override `data.num_classes` and `data.class_names` in the YAML.

## Features

- **Pluggable architectures** via a model factory: MONAI 3D residual U-Net, Swin UNETR, and a MaskMed-style set-prediction model with FSAD fusion and a Hungarian-matching head
- **Multiple loss options**: DiceCE with per-class weighting for heavy class imbalance (background ≈ 94 % of voxels), and the MaskMed set criterion (classification + mask BCE + mask Dice)
- **5-fold cross-validation** with configurable splits and seeded patient assignment
- **Mixed precision training** (AMP) on CUDA, with either Adam or SGD + polynomial LR schedules
- **Sliding-window inference** for processing full volumes at native resolution
- **Metrics**: per-organ Dice, HD95, and Surface Dice (configurable mm tolerance)
- **DICOM RT-Struct export** for clinical integration
- **TensorBoard logging** for real-time training monitoring
- **Docker support** for containerized inference

## Project Structure

```
.
├── src/
│   ├── data/              # Dataset discovery, preprocessing, transforms, intensity normalization
│   ├── models/            # Model factory + UNet3D, SwinUNETR, MaskMed backbones
│   ├── training/          # Trainer class, DiceCE and MaskMed losses
│   ├── evaluation/        # Metrics (Dice, HD95, Surface Dice), visualization
│   ├── export/            # NIfTI → DICOM RTSTRUCT conversion
│   └── utils/             # Config helpers, device selection
├── scripts/
│   ├── train.py                    # Training (single fold or full CV)
│   ├── predict.py                  # Inference on new volumes
│   ├── evaluate.py                 # Compute metrics against ground truth
│   ├── export_rtstruct.py          # Export predictions as DICOM RTSTRUCT
│   └── normalize_nifti_headers.py  # Fix NIfTI qform/sform headers
├── configs/
│   ├── baseline.yaml         # 14-class 3D U-Net (default)
│   ├── btcv_raw.yaml         # U-Net on raw BTCV inputs with dataset caching
│   ├── swin_unetr.yaml       # Swin UNETR backbone
│   ├── maskmed.yaml          # MaskMed set-prediction model
│   ├── maskmed_v2.yaml       # MaskMed v2 variant
│   └── averaged_binary.yaml  # Binary (foreground vs background) config
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA: volume stats, label distributions
│   └── 02_fold0_prediction_viewer.py # marimo viewer for fold-0 predictions
├── data/
│   ├── raw/               # Original BTCV NIfTI volumes (img/ + label/)
│   └── processed/         # Header-normalized / resampled volumes
├── maskmed_paper.pdf      # Reference paper for the MaskMed backbone
├── Dockerfile
├── requirements.txt
└── setup.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or install as editable package
pip install -e .
```

**Requirements**: Python 3.8+, PyTorch 2.0+, MONAI 1.3+. A CUDA-capable GPU with at least 8 GB VRAM is recommended; the MaskMed and Swin UNETR configs benefit from 16 GB+.

## Data Preparation

Obtain the BTCV challenge data (`RawData.zip`) and arrange it as:

```
data/raw/RawData/
├── Training/
│   ├── img/
│   │   ├── img0001.nii.gz   # 30 training volumes
│   │   └── ...
│   └── label/
│       ├── label0001.nii.gz
│       └── ...
└── Testing/
    └── img/
        ├── img0061.nii.gz   # 20 testing volumes (no labels)
        └── ...
```

Some BTCV volumes ship with inconsistent `qform`/`sform` headers. Normalize them (and optionally resample to a uniform grid) with:

```bash
python scripts/normalize_nifti_headers.py \
  --input-root data/raw/RawData \
  --output-root data/processed/RawData
```

At training/inference time, MONAI transforms resample to 1.5 × 1.5 × 2.0 mm spacing, clip intensities to [-175, 250] HU (soft-tissue window), and apply z-score normalization. These values are set in each config's `data` block.

## Usage

### Training

```bash
# Train a single fold with the default 3D U-Net
python scripts/train.py --config configs/baseline.yaml --fold 0

# Train all 5 CV folds
python scripts/train.py --config configs/baseline.yaml --fold -1

# Swap in a different architecture
python scripts/train.py --config configs/swin_unetr.yaml --fold 0
python scripts/train.py --config configs/maskmed.yaml    --fold 0

# Resume from checkpoint
python scripts/train.py --config configs/baseline.yaml --fold 0 \
  --resume results/checkpoints/fold_0/best_model.pth
```

Training logs are written to TensorBoard under `results/checkpoints/fold_<N>/logs/`. Monitor with:

```bash
tensorboard --logdir results/checkpoints
```

### Inference

```bash
python scripts/predict.py \
  --config configs/baseline.yaml \
  --checkpoint results/checkpoints/fold_0/best_model.pth \
  --input data/raw/RawData/Testing \
  --output results/predictions
```

### Evaluation

```bash
python scripts/evaluate.py \
  --config configs/baseline.yaml \
  --pred-dir results/predictions \
  --gt-dir data/processed/RawData/Training \
  --output results/metrics/scores.csv \
  --tolerance 3.0
```

Outputs a CSV with per-patient, per-organ Dice, HD95, and Surface Dice scores.

### DICOM RT-Struct Export

```bash
python scripts/export_rtstruct.py \
  --config configs/baseline.yaml \
  --ct-dicom-dir /path/to/ct/dicom \
  --prediction results/predictions/Patient_01.nii.gz \
  --output results/exports/Patient_01_RTSTRUCT.dcm
```

## Configuration

All settings are defined in YAML config files. Key sections:

| Section      | Key parameters                                                                |
|--------------|--------------------------------------------------------------------------------|
| `data`       | `patch_size`, `target_spacing`, `intensity_clip`, `num_classes`, `cache_rate` |
| `training`   | `batch_size`, `num_epochs`, `learning_rate`, `num_folds`, `amp`, `optimizer`  |
| `model`      | `architecture` (`UNet`, `SwinUNETR`, `MaskMed`) + architecture-specific params |
| `loss`       | `name` (`DiceCE` or `MaskMed`) + corresponding weights / `class_weights`       |
| `evaluation` | `sliding_window_size`, `overlap`, `sw_batch_size`                              |

See `configs/baseline.yaml` for the full reference configuration, and `configs/maskmed.yaml` / `configs/swin_unetr.yaml` for examples of the alternative backbones.

## Current Status

Baseline 3D U-Net reaches roughly **0.72 mean Dice** on the BTCV training split under 5-fold CV, versus the ~0.87 reported by the MaskMed paper. The MaskMed and Swin UNETR configs in this repo are being tuned to close that gap (see `notes.md`).

## Docker

```bash
# Build
docker build -t btcv-oar .

# Run inference
docker run --gpus all \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  btcv-oar \
  --config configs/baseline.yaml \
  --checkpoint /model.pth \
  --input /input \
  --output /output
```
