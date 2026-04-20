"""
MONAI transform pipelines for training and validation.

Training pipeline supports both the legacy project augmentations and a closer
nnU-Net-style configuration. Validation and inference apply only deterministic
preprocessing.
"""

import numpy as np

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    Lambdad,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandFlipd,
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandZoomd,
    EnsureTyped,
)

from src.utils.config import get_intensity_clip
from src.data.normalization import get_nnunet_ct_stats


def _build_intensity_transform(config: dict):
    mode = config["data"].get("normalization", "legacy_clip_01")
    if mode == "nnunet_ct":
        stats = get_nnunet_ct_stats(config)

        def _normalize_ct(image):
            image = np.clip(image, stats["percentile_00_5"], stats["percentile_99_5"])
            return ((image - stats["mean"]) / max(stats["std"], 1e-8)).astype(np.float32)

        return Lambdad(keys=["image"], func=_normalize_ct)

    intensity_min, intensity_max = get_intensity_clip(config)
    return ScaleIntensityRanged(
        keys=["image"],
        a_min=intensity_min,
        a_max=intensity_max,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    )


def get_train_transforms(config: dict) -> Compose:
    """
    Training transforms: preprocessing + augmentation + random patch extraction.

    ``augmentation_profile=nnunet_like`` approximates the nnU-Net v2 default 3D
    augmentation recipe while retaining the repo's fixed patch size.
    """
    train_cfg = config["training"]
    augmentation_profile = train_cfg.get("augmentation_profile", "legacy")
    samples_per_volume = train_cfg.get("samples_per_volume", 4 if augmentation_profile == "legacy" else 1)

    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=config["data"]["target_spacing"],
            mode=("bilinear", "nearest"),
        ),
        _build_intensity_transform(config),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=config["data"]["patch_size"],
            pos=config["data"]["pos_neg_ratio"],
            neg=1,
            num_samples=samples_per_volume,
        ),
    ]

    if augmentation_profile == "nnunet_like":
        transforms.extend([
            RandAffined(
                keys=["image", "label"],
                prob=0.2,
                rotate_range=(0.52, 0.52, 0.52),
                scale_range=(0.3, 0.3, 0.3),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            RandGaussianNoised(keys=["image"], prob=0.1, std=0.1),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.2,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
            ),
            RandScaleIntensityd(keys=["image"], factors=0.25, prob=0.15),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
            RandZoomd(
                keys=["image"],
                prob=0.25,
                min_zoom=0.5,
                max_zoom=1.0,
                mode="trilinear",
                padding_mode="constant",
                keep_size=True,
            ),
        ])
    else:
        transforms.extend([
            # --- Legacy project augmentations ---
            RandAffined(
                keys=["image", "label"],
                prob=0.3,
                rotate_range=(0.26, 0.26, 0.26),
                scale_range=(0.15, 0.15, 0.15),
                mode=("bilinear", "nearest"),
            ),
            RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
            RandGaussianSmoothd(keys=["image"], prob=0.2),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        ])

    transforms.extend([
        EnsureTyped(keys=["image", "label"]),
    ])
    return Compose(transforms)


def get_val_transforms(config: dict) -> Compose:
    """Validation transforms: same preprocessing, NO augmentation."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=config["data"]["target_spacing"],
            mode=("bilinear", "nearest"),
        ),
        _build_intensity_transform(config),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_inference_transforms(config: dict) -> Compose:
    """Inference transforms for test volumes (no labels)."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=config["data"]["target_spacing"],
            mode="bilinear",
        ),
        _build_intensity_transform(config),
        CropForegroundd(keys=["image"], source_key="image"),
        EnsureTyped(keys=["image"]),
    ])
