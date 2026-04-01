"""Device selection helpers."""

from __future__ import annotations

import torch


def get_best_available_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_device_name(device: torch.device) -> str:
    """Return a short display name for a torch device."""
    return device.type
