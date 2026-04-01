"""Evaluation metrics for segmentation labels."""

import numpy as np
from surface_distance import (
    compute_surface_distances,
    compute_robust_hausdorff,
    compute_surface_dice_at_tolerance,
)

DEFAULT_LABEL_MAP = {1: "Foreground"}


def compute_dice(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    """Volumetric Dice coefficient for a single organ."""
    pred_mask = pred == label
    gt_mask = gt == label

    intersection = np.sum(pred_mask & gt_mask)
    total = np.sum(pred_mask) + np.sum(gt_mask)

    if total == 0:
        return 1.0 if np.sum(gt_mask) == 0 else 0.0

    return (2.0 * intersection) / (total + 1e-8)


def compute_hausdorff_95(
    pred: np.ndarray, gt: np.ndarray, label: int, spacing: tuple[float, ...]
) -> float:
    """95th percentile Hausdorff distance in mm."""
    pred_mask = pred == label
    gt_mask = gt == label

    if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
        return float("inf")

    distances = compute_surface_distances(gt_mask, pred_mask, spacing)
    return float(compute_robust_hausdorff(distances, 95))


def compute_surface_dice(
    pred: np.ndarray,
    gt: np.ndarray,
    label: int,
    spacing: tuple[float, ...],
    tolerance_mm: float = 3.0,
) -> float:
    """Surface Dice at a given tolerance (mm)."""
    pred_mask = pred == label
    gt_mask = gt == label

    if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
        return 0.0

    distances = compute_surface_distances(gt_mask, pred_mask, spacing)
    return float(compute_surface_dice_at_tolerance(distances, tolerance_mm))


def evaluate_patient(
    pred_volume: np.ndarray,
    gt_volume: np.ndarray,
    spacing: tuple[float, ...],
    tolerance_mm: float = 3.0,
    label_map: dict[int, str] | None = None,
) -> dict:
    """
    Compute all metrics for one patient.

    Args:
        pred_volume: Predicted label volume with integer labels.
        gt_volume: Ground truth label volume.
        spacing: Voxel spacing in mm (z, y, x) matching array axis order.
        tolerance_mm: Tolerance for Surface Dice.
        label_map: Mapping from label id to display name for non-background labels.

    Returns:
        Dict mapping organ name -> {dice, hd95, surface_dice}.
    """
    results = {}
    active_label_map = label_map or DEFAULT_LABEL_MAP
    for label, name in active_label_map.items():
        results[name] = {
            "dice": compute_dice(pred_volume, gt_volume, label),
            "hd95": compute_hausdorff_95(pred_volume, gt_volume, label, spacing),
            "surface_dice": compute_surface_dice(
                pred_volume, gt_volume, label, spacing, tolerance_mm
            ),
        }
    return results
