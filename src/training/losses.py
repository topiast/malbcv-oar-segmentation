"""
Loss functions for organ segmentation.

Includes both conventional voxel-wise losses and a MaskMed-style set prediction
criterion with bipartite matching.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from monai.losses import DiceCELoss, DiceLoss
from scipy.optimize import linear_sum_assignment


def _dice_loss_from_logits(logits: Tensor, targets: Tensor) -> Tensor:
    probs = torch.sigmoid(logits)
    numerator = 2 * (probs * targets).flatten(1).sum(dim=1)
    denominator = probs.flatten(1).sum(dim=1) + targets.flatten(1).sum(dim=1)
    return 1 - (numerator + 1.0) / (denominator + 1.0)


class MaskMedCriterion(nn.Module):
    """Set-based loss with Hungarian matching for the MaskMed architecture."""

    def __init__(
        self,
        num_classes: int,
        class_weight: float = 2.0,
        mask_bce_weight: float = 10.0,
        mask_dice_weight: float = 10.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.foreground_classes = num_classes - 1
        self.no_object_index = self.foreground_classes
        self.class_weight = class_weight
        self.mask_bce_weight = mask_bce_weight
        self.mask_dice_weight = mask_dice_weight
        self.last_components = {
            "class_ce": 0.0,
            "mask_bce": 0.0,
            "mask_dice": 0.0,
            "weighted_total": 0.0,
        }

    def _build_targets(self, labels: Tensor) -> list[dict[str, Tensor]]:
        labels = labels.squeeze(1)
        targets = []
        for sample in labels:
            class_ids = torch.unique(sample)
            class_ids = class_ids[class_ids > 0]
            class_ids, _ = torch.sort(class_ids)

            if len(class_ids) == 0:
                targets.append({
                    "classes": sample.new_zeros((0,), dtype=torch.long),
                    "masks": sample.new_zeros((0, *sample.shape), dtype=torch.float32),
                })
                continue

            masks = torch.stack([(sample == class_id).float() for class_id in class_ids], dim=0)
            targets.append({
                "classes": (class_ids - 1).long(),
                "masks": masks,
            })
        return targets

    def _pairwise_mask_cost(self, pred_masks: Tensor, target_masks: Tensor) -> tuple[Tensor, Tensor]:
        num_queries = pred_masks.shape[0]
        num_targets = target_masks.shape[0]

        bce_cost = pred_masks.new_zeros((num_queries, num_targets))
        dice_cost = pred_masks.new_zeros((num_queries, num_targets))

        for target_index in range(num_targets):
            target = target_masks[target_index].unsqueeze(0).expand(num_queries, -1, -1, -1)
            bce = F.binary_cross_entropy_with_logits(pred_masks, target, reduction="none")
            bce_cost[:, target_index] = bce.flatten(1).mean(dim=1)
            dice_cost[:, target_index] = _dice_loss_from_logits(pred_masks, target)

        return bce_cost, dice_cost

    def _match_single(
        self,
        pred_logits: Tensor,
        pred_masks: Tensor,
        target: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        pred_logits = pred_logits.float()
        pred_masks = pred_masks.float()
        target_classes = target["classes"]
        target_masks = target["masks"]

        if target_classes.numel() == 0:
            empty = pred_logits.new_zeros((0,), dtype=torch.long)
            return empty, empty

        class_probs = pred_logits.softmax(dim=-1)
        class_cost = -class_probs[:, target_classes]
        bce_cost, dice_cost = self._pairwise_mask_cost(pred_masks, target_masks)

        total_cost = (
            self.class_weight * class_cost
            + self.mask_bce_weight * bce_cost
            + self.mask_dice_weight * dice_cost
        )
        query_indices, target_indices = linear_sum_assignment(total_cost.detach().cpu().numpy())
        return (
            pred_logits.new_tensor(query_indices, dtype=torch.long),
            pred_logits.new_tensor(target_indices, dtype=torch.long),
        )

    def _stage_losses(
        self,
        pred_logits: Tensor,
        pred_masks: Tensor,
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, num_queries = pred_logits.shape[:2]
        pred_logits = pred_logits.float()
        pred_masks = pred_masks.float()
        class_targets = pred_logits.new_full((batch_size, num_queries), self.no_object_index, dtype=torch.long)

        matched_mask_logits = []
        matched_mask_targets = []

        for batch_index, (query_indices, target_indices) in enumerate(indices):
            target = targets[batch_index]
            if len(query_indices) == 0:
                continue

            class_targets[batch_index, query_indices] = target["classes"][target_indices]
            target_masks = target["masks"][target_indices].unsqueeze(1)
            resized_masks = F.interpolate(
                target_masks,
                size=pred_masks.shape[-3:],
                mode="nearest",
            ).squeeze(1)
            matched_mask_logits.append(pred_masks[batch_index, query_indices])
            matched_mask_targets.append(resized_masks)

        loss_class = F.cross_entropy(pred_logits.transpose(1, 2), class_targets)

        if not matched_mask_logits:
            zero = pred_logits.sum() * 0.0
            return loss_class, zero, zero

        matched_mask_logits_tensor = torch.cat(matched_mask_logits, dim=0)
        matched_mask_targets_tensor = torch.cat(matched_mask_targets, dim=0)

        loss_mask_bce = F.binary_cross_entropy_with_logits(
            matched_mask_logits_tensor,
            matched_mask_targets_tensor,
        )
        loss_mask_dice = _dice_loss_from_logits(
            matched_mask_logits_tensor,
            matched_mask_targets_tensor,
        ).mean()
        return loss_class, loss_mask_bce, loss_mask_dice

    def forward(self, outputs: dict, labels: Tensor) -> Tensor:
        if not isinstance(outputs, dict) or "stages" not in outputs:
            raise TypeError("MaskMedCriterion expects model outputs with a 'stages' list")

        stage_outputs = outputs["stages"]
        stage_weights = outputs.get("stage_weights")
        if stage_weights is None:
            stage_weights = [1.0 / len(stage_outputs)] * len(stage_outputs)

        targets = self._build_targets(labels)
        final_output = stage_outputs[-1]
        indices = [
            self._match_single(final_output["pred_logits"][batch_index], final_output["pred_masks"][batch_index], targets[batch_index])
            for batch_index in range(labels.shape[0])
        ]

        total_loss = labels.sum() * 0.0
        total_class = labels.sum() * 0.0
        total_mask_bce = labels.sum() * 0.0
        total_mask_dice = labels.sum() * 0.0
        for weight, stage_output in zip(stage_weights, stage_outputs, strict=True):
            loss_class, loss_mask_bce, loss_mask_dice = self._stage_losses(
                stage_output["pred_logits"],
                stage_output["pred_masks"],
                targets,
                indices,
            )
            total_class = total_class + weight * loss_class
            total_mask_bce = total_mask_bce + weight * loss_mask_bce
            total_mask_dice = total_mask_dice + weight * loss_mask_dice
            total_loss = total_loss + weight * (
                self.class_weight * loss_class
                + self.mask_bce_weight * loss_mask_bce
                + self.mask_dice_weight * loss_mask_dice
            )
        self.last_components = {
            "class_ce": float(total_class.detach().item()),
            "mask_bce": float(total_mask_bce.detach().item()),
            "mask_dice": float(total_mask_dice.detach().item()),
            "weighted_total": float(total_loss.detach().item()),
        }
        return total_loss

    def get_last_components(self) -> dict[str, float]:
        """Return detached component values from the most recent forward pass."""
        return dict(self.last_components)


def build_loss(config: dict):
    """Build loss function from config."""
    loss_cfg = config["loss"]
    loss_name = loss_cfg["name"]
    class_weights = loss_cfg.get("class_weights")
    weight = None
    if class_weights is not None:
        weight = torch.tensor(class_weights, dtype=torch.float32)

    if loss_name == "DiceCE":
        return DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            lambda_dice=loss_cfg["dice_weight"],
            lambda_ce=loss_cfg["ce_weight"],
            weight=weight,
        )
    if loss_name == "Dice":
        return DiceLoss(
            to_onehot_y=True,
            softmax=True,
        )
    if loss_name == "MaskMed":
        return MaskMedCriterion(
            num_classes=config["data"]["num_classes"],
            class_weight=loss_cfg.get("class_weight", 2.0),
            mask_bce_weight=loss_cfg.get("mask_bce_weight", 10.0),
            mask_dice_weight=loss_cfg.get("mask_dice_weight", 10.0),
        )

    raise ValueError(f"Unknown loss: {loss_name}. Supported: DiceCE, Dice, MaskMed")
