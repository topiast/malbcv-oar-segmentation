"""Configuration helpers for dataset labels and intensity scaling."""

from __future__ import annotations


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
