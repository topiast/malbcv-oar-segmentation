"""
MONAI transform pipelines for training and validation.

Training pipeline includes data augmentation (random flips, affine, noise, intensity shifts).
Validation pipeline applies only deterministic preprocessing (no augmentation).
"""

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureTyped,
)

from src.utils.config import get_intensity_clip


def get_train_transforms(config: dict) -> Compose:
    """
    Training transforms: preprocessing + augmentation + random patch extraction.

    RandCropByPosNegLabeld extracts patches with a 2:1 foreground:background ratio,
    ensuring the model sees organ voxels frequently despite extreme class imbalance.
    """
    intensity_min, intensity_max = get_intensity_clip(config)

    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=config["data"]["target_spacing"],
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=intensity_min,
            a_max=intensity_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # --- Augmentation (training only) ---
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=config["data"]["patch_size"],
            pos=config["data"]["pos_neg_ratio"],
            neg=1,
            num_samples=4,  # 4 patches per volume per batch
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandAffined(
            keys=["image", "label"],
            prob=0.3,
            rotate_range=(0.26, 0.26, 0.26),   # ±15 degrees
            scale_range=(0.15, 0.15, 0.15),     # ±15%
            mode=("bilinear", "nearest"),
        ),
        RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
        RandGaussianSmoothd(keys=["image"], prob=0.2),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms(config: dict) -> Compose:
    """Validation transforms: same preprocessing, NO augmentation."""
    intensity_min, intensity_max = get_intensity_clip(config)

    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=config["data"]["target_spacing"],
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=intensity_min,
            a_max=intensity_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_inference_transforms(config: dict) -> Compose:
    """Inference transforms for test volumes (no labels)."""
    intensity_min, intensity_max = get_intensity_clip(config)

    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=config["data"]["target_spacing"],
            mode="bilinear",
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=intensity_min,
            a_max=intensity_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        EnsureTyped(keys=["image"]),
    ])
