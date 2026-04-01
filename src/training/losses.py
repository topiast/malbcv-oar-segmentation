"""
Loss functions for organ segmentation.

DiceCE (Dice + Cross-Entropy) is the default:
- Dice handles class imbalance (background is ~99% of voxels)
- CE provides stable gradients early in training
"""

import torch
from monai.losses import DiceCELoss, DiceLoss


def build_loss(config: dict):
    """
    Build loss function from config.

    Args:
        config: Full config dict (uses config["loss"] section).

    Returns:
        Loss function module.
    """
    loss_cfg = config["loss"]
    loss_name = loss_cfg["name"]
    class_weights = loss_cfg.get("class_weights")
    weight = None
    if class_weights is not None:
        weight = torch.tensor(class_weights, dtype=torch.float32)

    if loss_name == "DiceCE":
        loss_fn = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            lambda_dice=loss_cfg["dice_weight"],
            lambda_ce=loss_cfg["ce_weight"],
            weight=weight,
        )
    elif loss_name == "Dice":
        loss_fn = DiceLoss(
            to_onehot_y=True,
            softmax=True,
        )
    else:
        raise ValueError(f"Unknown loss: {loss_name}. Supported: DiceCE, Dice")

    return loss_fn
