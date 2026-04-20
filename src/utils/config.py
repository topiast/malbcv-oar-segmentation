"""Configuration helpers for dataset labels, paths, and intensity scaling."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path


DEFAULT_CLASS_NAMES = [
    "Background",
    "Spleen",
    "Right Kidney",
    "Left Kidney",
    "Gallbladder",
    "Esophagus",
    "Liver",
    "Stomach",
    "Aorta",
    "IVC",
    "Portal & Splenic Veins",
    "Pancreas",
    "Right Adrenal Gland",
    "Left Adrenal Gland",
]

PATH_KEYS = {
    "data": {
        "raw_dir",
        "train_dir",
        "processed_dir",
        "test_dir",
        "split_file",
        "normalization_cache",
    },
    "output": {
        "checkpoint_dir",
        "prediction_dir",
        "figures_dir",
        "metrics_dir",
    },
}


def resolve_config_paths(config: dict, base_dir: str | Path) -> dict:
    """
    Return a config copy with known relative filesystem paths resolved.

    Repository configs use repo-relative paths like ``data/...`` and
    ``results/...``. Resolving them once at load time makes notebooks and CLI
    entry points insensitive to the caller's current working directory.
    """
    resolved = deepcopy(config)
    base_dir = Path(base_dir)

    for section, keys in PATH_KEYS.items():
        section_cfg = resolved.get(section)
        if not isinstance(section_cfg, dict):
            continue

        for key in keys:
            value = section_cfg.get(key)
            if value in (None, ""):
                continue

            path = Path(value)
            if not path.is_absolute():
                section_cfg[key] = str(base_dir / path)

    return resolved


def get_class_names(config: dict) -> list[str]:
    """Return configured class names, or sensible defaults for the class count."""
    data_cfg = config["data"]
    num_classes = data_cfg["num_classes"]
    class_names = data_cfg.get("class_names")

    if class_names:
        if len(class_names) != num_classes:
            raise ValueError(
                f"Expected {num_classes} class names, got {len(class_names)}"
            )
        return class_names

    if num_classes == len(DEFAULT_CLASS_NAMES):
        return DEFAULT_CLASS_NAMES

    return ["Background"] + [f"Class {idx}" for idx in range(1, num_classes)]


def get_foreground_label_map(config: dict) -> dict[int, str]:
    """Return a mapping from non-background label ids to display names."""
    return {
        idx: name
        for idx, name in enumerate(get_class_names(config))
        if idx != 0
    }


def get_intensity_clip(config: dict) -> tuple[float, float]:
    """Return the configured intensity clip range."""
    data_cfg = config["data"]
    clip = data_cfg.get("intensity_clip", data_cfg.get("hu_clip"))
    if clip is None or len(clip) != 2:
        raise ValueError("Config must define data.intensity_clip or data.hu_clip")
    return float(clip[0]), float(clip[1])
