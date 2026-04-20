# MALBCV OAR Segmentation

Automatic segmentation of organs-at-risk (OAR) in abdominal CT volumes using a 3D U-Net built on [MONAI](https://monai.io/). Designed for the MALBCV dataset (84 annotated CT volumes, 13 abdominal organs + background).

## Segmented Organs

| Label | Organ               | Label | Organ                  |
|-------|---------------------|-------|------------------------|
| 1     | Spleen              | 8     | Aorta                  |
| 2     | Right Kidney        | 9     | Inferior Vena Cava     |
| 3     | Left Kidney         | 10    | Portal & Splenic Veins |
| 4     | Gallbladder         | 11    | Pancreas               |
| 5     | Esophagus           | 12    | Right Adrenal Gland    |
| 6     | Liver               | 13    | Left Adrenal Gland     |
| 7     | Stomach             |       |                        |

## Features

- **3D U-Net** (~30M params) with instance normalization and residual blocks
- **DiceCE loss** with per-class weighting for extreme class imbalance (background ~94% of voxels)
- **5-fold cross-validation** with configurable splits
- **Mixed precision training** (AMP) on CUDA
- **Sliding window inference** for processing full volumes without downsampling
- **Comprehensive metrics**: Dice coefficient, HD95, Surface Dice
- **DICOM RT-Struct export** for clinical integration
- **TensorBoard logging** for real-time training monitoring
- **Docker support** for containerized inference

## Project Structure

```
.
├── src/
│   ├── data/              # Dataset discovery, preprocessing, transforms
│   ├── models/            # 3D U-Net architecture
│   ├── training/          # Trainer class, DiceCE loss
│   ├── evaluation/        # Metrics (Dice, HD95, Surface Dice), visualization
│   ├── export/            # NIfTI to DICOM RTSTRUCT conversion
│   └── utils/             # Config helpers, device selection
├── scripts/
│   ├── train.py           # Training (single fold or full CV)
│   ├── predict.py         # Inference on new volumes
│   ├── evaluate.py        # Compute metrics against ground truth
│   ├── export_rtstruct.py # Export predictions as DICOM RTSTRUCT
│   └── normalize_nifti_headers.py  # Fix NIfTI qform/sform headers
├── configs/
│   ├── baseline.yaml      # 14-class segmentation config
│   └── averaged_binary.yaml  # Binary segmentation config
├── notebooks/
│   └── 01_data_exploration.ipynb  # EDA: volume stats, label distributions
├── data/
│   ├── raw/               # Original MALBCV NIfTI volumes
│   └── processed/         # Resampled & normalized volumes
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

**Requirements**: Python 3.8+, PyTorch 2.0+, MONAI 1.3+. A CUDA-capable GPU with at least 8 GB VRAM is recommended.

## Data Preparation

Place your MALBCV data under `data/raw/RawData/`:

```
data/raw/RawData/
├── Training/
│   ├── img/
│   │   ├── img0001.nii.gz
│   │   └── ...
│   └── label/
│       ├── label0001.nii.gz
│       └── ...
└── Testing/
    └── img/
        └── ...
```

Optionally fix NIfTI header issues and resample to uniform spacing:

```bash
python scripts/normalize_nifti_headers.py \
  --input-root data/raw/RawData \
  --output-root data/processed/RawData
```

Input volumes are resampled to 1.5 x 1.5 x 2.0 mm spacing, clipped to [-175, 250] HU (soft tissue window), and z-score normalized.

## Usage

### Training

```bash
# Train a single fold
python scripts/train.py --config configs/baseline.yaml --fold 0

# Train all 5 folds
python scripts/train.py --config configs/baseline.yaml --fold -1

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

| Section      | Key parameters                                                  |
|--------------|-----------------------------------------------------------------|
| `data`       | `patch_size`, `target_spacing`, `intensity_clip`, `num_classes` |
| `training`   | `batch_size`, `num_epochs`, `learning_rate`, `num_folds`, `amp` |
| `model`      | `channels`, `strides`, `num_res_units`, `dropout`, `norm`       |
| `loss`       | `dice_weight`, `ce_weight`, `class_weights`                     |
| `evaluation` | `sliding_window_size`, `overlap`, `sw_batch_size`               |

See `configs/baseline.yaml` for the full reference configuration.

## Docker

```bash
# Build
docker build -t malbcv-oar .

# Run inference
docker run --gpus all \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  malbcv-oar \
  --config configs/baseline.yaml \
  --checkpoint /model.pth \
  --input /input \
  --output /output
```
