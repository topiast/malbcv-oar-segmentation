"""Helpers for dataset-level intensity normalization."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np

from src.data.dataset import discover_training_pairs

logger = logging.getLogger(__name__)


def _resolve_repo_relative_path(path_value: str | Path) -> Path:
    """
    Resolve a filesystem path against the repository root when needed.

    Configs in this repo use paths like ``data/...`` and ``results/...``. Some
    notebook runners still evaluate those relative to their launch directory, so
    we fall back to the repo root derived from this source file.
    """
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return path

    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / path
    if candidate.exists():
        return candidate
    return candidate


def get_nnunet_ct_stats(config: dict) -> dict[str, float]:
    """
    Resolve dataset-level CT normalization statistics.

    nnU-Net CT normalization clips to foreground intensity percentiles computed from
    the training set and then applies z-scoring with the corresponding foreground
    mean and standard deviation.
    """
    data_cfg = config["data"]
    explicit_stats = data_cfg.get("nnunet_ct_stats")
    if explicit_stats is not None:
        return {
            "mean": float(explicit_stats["mean"]),
            "std": float(explicit_stats["std"]),
            "percentile_00_5": float(explicit_stats["percentile_00_5"]),
            "percentile_99_5": float(explicit_stats["percentile_99_5"]),
        }

    train_root = _resolve_repo_relative_path(
        data_cfg.get("train_dir", data_cfg.get("processed_dir", data_cfg["raw_dir"]))
    )

    cache_value = data_cfg.get("normalization_cache")
    cache_path = (
        _resolve_repo_relative_path(cache_value)
        if cache_value is not None
        else train_root / "nnunet_ct_stats.json"
    )
    if cache_path.exists():
        with cache_path.open() as f:
            cached = json.load(f)
        return {
            "mean": float(cached["mean"]),
            "std": float(cached["std"]),
            "percentile_00_5": float(cached["percentile_00_5"]),
            "percentile_99_5": float(cached["percentile_99_5"]),
        }

    training_pairs = discover_training_pairs(train_root)
    if not training_pairs:
        raise FileNotFoundError(
            f"Cannot compute nnU-Net CT normalization stats because no training data was found in {train_root}"
        )

    logger.info("Computing nnU-Net CT normalization stats from %d training cases", len(training_pairs))
    rng = np.random.default_rng(42)
    sampled_foreground_values: list[np.ndarray] = []
    max_percentile_samples_per_case = 200_000

    count = 0
    value_sum = 0.0
    squared_sum = 0.0

    for item in training_pairs:
        image = nib.load(item["image"]).get_fdata(dtype=np.float32)
        label = nib.load(item["label"]).get_fdata(dtype=np.float32)
        foreground = image[label > 0]
        if foreground.size == 0:
            continue

        count += int(foreground.size)
        value_sum += float(foreground.sum(dtype=np.float64))
        squared_sum += float(np.square(foreground, dtype=np.float64).sum(dtype=np.float64))

        if foreground.size > max_percentile_samples_per_case:
            sample_indices = rng.choice(foreground.size, size=max_percentile_samples_per_case, replace=False)
            sampled_foreground_values.append(foreground[sample_indices].astype(np.float32, copy=False))
        else:
            sampled_foreground_values.append(foreground.astype(np.float32, copy=False))

    if count == 0 or not sampled_foreground_values:
        raise ValueError(f"No foreground voxels found while computing normalization stats from {train_root}")

    percentile_values = np.concatenate(sampled_foreground_values, axis=0)
    mean = value_sum / count
    variance = max(squared_sum / count - mean ** 2, 1e-8)
    stats = {
        "mean": float(mean),
        "std": float(np.sqrt(variance)),
        "percentile_00_5": float(np.percentile(percentile_values, 0.5)),
        "percentile_99_5": float(np.percentile(percentile_values, 99.5)),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    logger.info("Saved nnU-Net CT normalization stats to %s", cache_path)
    return stats
