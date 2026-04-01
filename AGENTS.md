# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/`: `src/data` handles dataset loading and preprocessing, `src/models` defines the 3D U-Net, `src/training` contains losses and the trainer, `src/evaluation` covers metrics and visualization, and `src/export` writes RTSTRUCT output. CLI entry points are in `scripts/` (`train.py`, `predict.py`, `evaluate.py`, `export_rtstruct.py`). Runtime configuration is in `configs/baseline.yaml`. Keep raw and processed datasets under `data/raw` and `data/processed`, and write generated artifacts to `results/` instead of committing them.

## Build, Test, and Development Commands
Set up the environment with `python -m venv .venv && source .venv/bin/activate`, then install dependencies with `pip install -r requirements.txt` or `pip install -e .`.

- `python scripts/train.py --config configs/baseline.yaml`: train one fold with the baseline config.
- `python scripts/train.py --config configs/baseline.yaml --fold -1`: run all configured cross-validation folds.
- `python scripts/predict.py --config configs/baseline.yaml --checkpoint results/checkpoints/fold_0/best_model.pth --input data/raw/RawData/Testing --output results/predictions`: run inference.
- `python scripts/evaluate.py --pred-dir results/predictions --gt-dir data/processed/train --output results/metrics/scores.csv`: compute Dice, HD95, and surface Dice.
- `docker build -t malbcv-oar .`: build the container image from the provided `Dockerfile`.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, descriptive snake_case for functions and variables, and PascalCase for classes such as `Trainer`. Match the existing pattern of small, focused modules and explicit imports from `src.*`. Keep YAML keys lowercase with underscores. Prefer type-safe path handling with `pathlib.Path` in scripts and utilities.

## Testing Guidelines
This snapshot does not include a dedicated `tests/` suite yet. Treat script-level validation as the minimum bar: run training on a small fold, verify inference writes `.nii.gz` outputs, and confirm evaluation produces `results/metrics/scores.csv`. When adding tests, place them under `tests/`, use `pytest`, and name files `test_<module>.py`.

## Commit & Pull Request Guidelines
Git history is not available in this workspace, so no repository-specific commit convention can be inferred. Use short, imperative commit subjects such as `Add surface dice logging`. In pull requests, include the purpose, affected data paths or configs, reproducible commands, and before/after metric changes when model behavior is affected. Attach sample figures only when visualization output changes.

## Data & Configuration Notes
Do not hardcode local dataset locations outside `configs/*.yaml`. Large checkpoints, raw DICOM/NIfTI data, and generated metrics should stay out of version control unless explicitly required.
