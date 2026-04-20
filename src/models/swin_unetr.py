"""Swin UNETR model builder."""

from __future__ import annotations

import inspect

from torch import nn


def build_model(config: dict) -> nn.Module:
    """
    Build a Swin UNETR model from config.

    The constructor signature varies slightly across MONAI versions, so kwargs are
    filtered dynamically for compatibility.
    """
    from monai.networks.nets import SwinUNETR

    model_cfg = config["model"]
    signature = inspect.signature(SwinUNETR)

    kwargs = {
        "spatial_dims": 3,
        "in_channels": model_cfg["in_channels"],
        "out_channels": model_cfg["out_channels"],
        "feature_size": model_cfg.get("feature_size", 48),
        "depths": tuple(model_cfg.get("depths", [2, 2, 2, 2])),
        "num_heads": tuple(model_cfg.get("num_heads", [3, 6, 12, 24])),
        "drop_rate": model_cfg.get("drop_rate", 0.0),
        "attn_drop_rate": model_cfg.get("attn_drop_rate", 0.0),
        "dropout_path_rate": model_cfg.get("dropout_path_rate", 0.0),
        "norm_name": model_cfg.get("norm", "instance"),
        "normalize": model_cfg.get("normalize", True),
        "use_checkpoint": model_cfg.get("use_checkpoint", False),
        "use_v2": model_cfg.get("use_v2", False),
        "downsample": model_cfg.get("downsample", "merging"),
    }

    if "img_size" in signature.parameters:
        kwargs["img_size"] = tuple(config["data"]["patch_size"])

    filtered_kwargs = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return SwinUNETR(**filtered_kwargs)
