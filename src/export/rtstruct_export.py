"""Export prediction masks to DICOM RT Structure Set format."""

import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)

DEFAULT_ROI_CONFIG = {
    1: {"name": "Foreground", "color": [255, 99, 71]},
}


def build_roi_config(class_names: list[str]) -> dict[int, dict]:
    """Build ROI metadata from configured class names."""
    palette = [
        [255, 99, 71],
        [70, 130, 180],
        [60, 179, 113],
        [255, 215, 0],
        [186, 85, 211],
        [255, 140, 0],
        [46, 139, 87],
        [220, 20, 60],
    ]
    return {
        idx: {"name": name, "color": palette[(idx - 1) % len(palette)]}
        for idx, name in enumerate(class_names)
        if idx != 0
    }


def export_rtstruct(
    ct_dicom_dir: str | Path,
    prediction_path: str | Path,
    output_path: str | Path,
    roi_config: dict[int, dict] | None = None,
):
    """
    Convert a NIfTI prediction mask to DICOM RT Structure Set.

    Args:
        ct_dicom_dir: Directory containing the original CT DICOM series.
        prediction_path: Path to NIfTI prediction mask with integer labels.
        output_path: Output path for the RTSTRUCT DICOM file.
        roi_config: Optional mapping from label id to ROI metadata.

    Requires:
        Original CT in DICOM format (not NIfTI) for proper RTSTRUCT generation.
        Install rt-utils: pip install rt-utils
    """
    try:
        from rt_utils import RTStructBuilder
    except ImportError:
        raise ImportError("rt-utils is required for RTSTRUCT export: pip install rt-utils")

    ct_dicom_dir = Path(ct_dicom_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load prediction
    pred = sitk.ReadImage(str(prediction_path))
    pred_array = sitk.GetArrayFromImage(pred)  # shape: (Z, Y, X)

    # Create RTSTRUCT from DICOM series
    rtstruct = RTStructBuilder.create_new(dicom_series_path=str(ct_dicom_dir))

    active_roi_config = roi_config or DEFAULT_ROI_CONFIG

    for label_val, organ_cfg in active_roi_config.items():
        mask = (pred_array == label_val).astype(bool)
        if mask.any():
            rtstruct.add_roi(
                mask=mask,
                color=organ_cfg["color"],
                name=organ_cfg["name"],
            )
            logger.info(f"Added ROI: {organ_cfg['name']} ({mask.sum()} voxels)")
        else:
            logger.warning(f"No voxels for {organ_cfg['name']}, skipping")

    rtstruct.save(str(output_path))
    logger.info(f"RTSTRUCT saved to {output_path}")


def prediction_to_nifti(
    pred_array: np.ndarray,
    reference_image: sitk.Image,
    output_path: str | Path,
):
    """
    Save a prediction array as NIfTI, copying spatial metadata from a reference.

    Args:
        pred_array: Integer label array (Z, Y, X).
        reference_image: SimpleITK image to copy spacing/origin/direction from.
        output_path: Output NIfTI path.
    """
    pred_image = sitk.GetImageFromArray(pred_array.astype(np.uint8))
    pred_image.SetSpacing(reference_image.GetSpacing())
    pred_image.SetOrigin(reference_image.GetOrigin())
    pred_image.SetDirection(reference_image.GetDirection())

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(pred_image, str(output_path))
    logger.info(f"Prediction saved to {output_path}")
