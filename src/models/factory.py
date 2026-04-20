"""Model registry for segmentation backbones."""

from __future__ import annotations

from torch import nn

from src.models.maskmed import build_model as build_maskmed
from src.models.swin_unetr import build_model as build_swin_unetr
from src.models.unet3d import build_model as build_unet


MODEL_BUILDERS = {
    "maskmed": build_maskmed,
    "unet": build_unet,
    "swinunetr": build_swin_unetr,
}


def _normalize_architecture_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "")


def build_model(config: dict) -> nn.Module:
    """Build the configured model without changing existing training code."""
    architecture = config["model"].get("architecture", "UNet")
    normalized = _normalize_architecture_name(architecture)

    if normalized not in MODEL_BUILDERS:
        supported = ", ".join(sorted(MODEL_BUILDERS))
        raise ValueError(
            f"Unknown model architecture '{architecture}'. Supported: {supported}"
        )

    return MODEL_BUILDERS[normalized](config)


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
