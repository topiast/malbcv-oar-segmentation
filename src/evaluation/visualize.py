"""Visualization utilities for segmentation results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_LABEL_COLORS = {
    1: (1.0, 0.39, 0.28, 0.5),
    2: (0.27, 0.51, 0.71, 0.5),
    3: (0.24, 0.70, 0.44, 0.5),
    4: (1.0, 0.84, 0.0, 0.5),
}


def _build_overlay(
    mask: np.ndarray,
    label_colors: dict[int, tuple[float, float, float, float]] | None = None,
) -> np.ndarray:
    """Convert integer label mask to an RGBA overlay."""
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    active_colors = label_colors or DEFAULT_LABEL_COLORS
    for label, color in active_colors.items():
        overlay[mask == label] = color
    return overlay


def plot_slice_overlay(
    ct_slice: np.ndarray,
    label_slice: np.ndarray,
    title: str = "",
    ax: plt.Axes | None = None,
    label_colors: dict[int, tuple[float, float, float, float]] | None = None,
) -> plt.Axes:
    """Plot a CT slice with label contours overlaid."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.imshow(ct_slice, cmap="gray", origin="lower")
    overlay = _build_overlay(label_slice, label_colors=label_colors)
    ax.imshow(overlay, origin="lower")
    ax.set_title(title)
    ax.axis("off")
    return ax


def plot_prediction_comparison(
    ct_volume: np.ndarray,
    gt_volume: np.ndarray,
    pred_volume: np.ndarray,
    slice_idx: int | None = None,
    save_path: str | Path | None = None,
    label_colors: dict[int, tuple[float, float, float, float]] | None = None,
):
    """
    Plot a 3-row comparison: CT, GT overlay, prediction overlay, error map.

    Args:
        ct_volume: CT array (Z, Y, X).
        gt_volume: Ground truth labels (Z, Y, X).
        pred_volume: Predicted labels (Z, Y, X).
        slice_idx: Axial slice index. If None, picks the slice with most organs.
        save_path: If provided, save figure to this path.
    """
    if slice_idx is None:
        # Pick axial slice with most foreground voxels
        fg_per_slice = np.sum(gt_volume > 0, axis=(1, 2))
        slice_idx = int(np.argmax(fg_per_slice))

    ct_ax = ct_volume[slice_idx]
    gt_ax = gt_volume[slice_idx]
    pred_ax = pred_volume[slice_idx]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # CT only
    axes[0].imshow(ct_ax, cmap="gray", origin="lower")
    axes[0].set_title("CT")
    axes[0].axis("off")

    # Ground truth overlay
    plot_slice_overlay(
        ct_ax,
        gt_ax,
        title="Ground Truth",
        ax=axes[1],
        label_colors=label_colors,
    )

    # Prediction overlay
    plot_slice_overlay(
        ct_ax,
        pred_ax,
        title="Prediction",
        ax=axes[2],
        label_colors=label_colors,
    )

    # Error map
    axes[3].imshow(ct_ax, cmap="gray", origin="lower")
    error = np.zeros((*ct_ax.shape, 4), dtype=np.float32)
    # False positives: predicted organ where GT is background or different organ
    fp = (pred_ax > 0) & (pred_ax != gt_ax)
    fn = (gt_ax > 0) & (pred_ax != gt_ax)
    error[fp] = (1.0, 0.0, 0.0, 0.6)  # Red = false positive
    error[fn] = (0.0, 0.0, 1.0, 0.6)  # Blue = false negative
    axes[3].imshow(error, origin="lower")
    axes[3].set_title("Errors (Red=FP, Blue=FN)")
    axes[3].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_training_curves(
    train_losses: list[float],
    val_dices: dict[str, list[float]],
    val_epochs: list[int],
    save_path: str | Path | None = None,
):
    """
    Plot training loss and per-class validation Dice curves.

    Args:
        train_losses: Loss per epoch.
        val_dices: Dict mapping class name -> list of Dice scores at val_epochs.
        val_epochs: Epoch numbers where validation was run.
        save_path: If provided, save figure to this path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Loss curve
    ax1.plot(train_losses, linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    for class_name, dices in val_dices.items():
        ax2.plot(val_epochs, dices, label=class_name, linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score")
    ax2.set_title("Validation Dice per Class")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
