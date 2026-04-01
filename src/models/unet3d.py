"""
3D U-Net model definition using MONAI.

Uses Instance Normalization (stable with small batch sizes typical in 3D medical imaging)
and residual units for better gradient flow.
"""

from monai.networks.nets import UNet
from monai.networks.layers import Norm


NORM_MAP = {
    "instance": Norm.INSTANCE,
    "batch": Norm.BATCH,
    "group": Norm.GROUP,
}


def build_model(config: dict) -> UNet:
    """
    Build a 3D U-Net from config.

    Architecture: 5-level encoder-decoder with skip connections.
    Default: [32, 64, 128, 256, 512] channels, ~30M parameters.
    Fits on 8GB GPU with batch_size=2, patch_size=[128, 128, 64].

    Args:
        config: Full config dict (uses config["model"] section).

    Returns:
        MONAI UNet model.
    """
    model_cfg = config["model"]

    norm_type = NORM_MAP.get(model_cfg.get("norm", "instance"), Norm.INSTANCE)

    model = UNet(
        spatial_dims=3,
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        channels=model_cfg["channels"],
        strides=model_cfg["strides"],
        num_res_units=model_cfg["num_res_units"],
        norm=norm_type,
        dropout=model_cfg["dropout"],
    )

    return model


def count_parameters(model: UNet) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
