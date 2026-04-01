"""
Preprocessing pipeline for abdominal organ segmentation datasets.

Steps per patient:
1. Load NIfTI volume and labels
2. Resample to target spacing (1.5 x 1.5 x 2.0 mm)
3. Clip HU values to [-175, 250] (abdominal soft tissue window)
4. Z-score normalize intensity
5. Save processed volumes to output directory
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

logger = logging.getLogger(__name__)

ORGAN_NAMES = {
    0: "Background",
    1: "Spleen",
    2: "Right Kidney",
    3: "Left Kidney",
    4: "Gallbladder",
    5: "Esophagus",
    6: "Liver",
    7: "Stomach",
    8: "Aorta",
    9: "IVC",
    10: "Portal & Splenic Veins",
    11: "Pancreas",
    12: "Right Adrenal Gland",
    13: "Left Adrenal Gland",
}


def resample_volume(
    image: sitk.Image,
    target_spacing: tuple[float, ...],
    interpolator=sitk.sitkLinear,
) -> sitk.Image:
    """Resample a SimpleITK image to a target spacing."""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(image)


def preprocess_patient(
    ct_path: str | Path,
    gt_path: str | Path,
    output_dir: str | Path,
    target_spacing: tuple[float, ...] = (1.5, 1.5, 2.0),
    hu_min: float = -175.0,
    hu_max: float = 250.0,
) -> dict:
    """
    Preprocess a single patient's CT and ground truth volumes.

    Returns:
        dict with metadata about the processed volumes.
    """
    ct_path = Path(ct_path)
    gt_path = Path(gt_path)
    output_dir = Path(output_dir)

    # Extract patient name from filename (e.g., "img0001" from "img0001.nii.gz")
    patient_name = ct_path.name.replace(".nii.gz", "").replace(".nii", "")
    patient_out = output_dir / patient_name
    patient_out.mkdir(parents=True, exist_ok=True)

    # Load volumes
    ct = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)
    gt = sitk.ReadImage(str(gt_path), sitk.sitkUInt8)

    original_spacing = ct.GetSpacing()
    original_size = ct.GetSize()

    # Resample CT (linear interpolation) and GT (nearest neighbor)
    ct_resampled = resample_volume(ct, target_spacing, sitk.sitkLinear)
    gt_resampled = resample_volume(gt, target_spacing, sitk.sitkNearestNeighbor)

    # Clip HU values
    ct_array = sitk.GetArrayFromImage(ct_resampled)
    ct_array = np.clip(ct_array, hu_min, hu_max)

    # Z-score normalization
    mean_val = ct_array.mean()
    std_val = ct_array.std()
    ct_array = (ct_array - mean_val) / (std_val + 1e-8)

    # Convert back to SimpleITK image, preserving spatial info
    ct_processed = sitk.GetImageFromArray(ct_array)
    ct_processed.SetSpacing(ct_resampled.GetSpacing())
    ct_processed.SetOrigin(ct_resampled.GetOrigin())
    ct_processed.SetDirection(ct_resampled.GetDirection())

    # Save
    ct_out_path = patient_out / f"{patient_name}.nii.gz"
    gt_out_path = patient_out / "GT.nii.gz"
    sitk.WriteImage(ct_processed, str(ct_out_path))
    sitk.WriteImage(gt_resampled, str(gt_out_path))

    # Collect metadata
    gt_array = sitk.GetArrayFromImage(gt_resampled)
    metadata = {
        "patient": patient_name,
        "original_spacing": original_spacing,
        "original_size": original_size,
        "resampled_size": ct_resampled.GetSize(),
        "hu_mean": float(mean_val),
        "hu_std": float(std_val),
    }
    for label, name in ORGAN_NAMES.items():
        metadata[f"voxels_{name.lower().replace(' ', '_').replace('&', 'and')}"] = int(np.sum(gt_array == label))

    logger.info(f"Processed {patient_name}: {original_size} -> {ct_resampled.GetSize()}")
    return metadata


def preprocess_all(
    input_dir: str | Path,
    output_dir: str | Path,
    target_spacing: tuple[float, ...] = (1.5, 1.5, 2.0),
    hu_min: float = -175.0,
    hu_max: float = 250.0,
) -> list[dict]:
    """
    Preprocess all patients in an img/label training directory.

    Expected structure:
        input_dir/
            img/        # CT volumes (img0001.nii.gz, img0002.nii.gz, ...)
            label/      # Ground truth (label0001.nii.gz, label0002.nii.gz, ...)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_dir = input_dir / "img"
    label_dir = input_dir / "label"

    ct_files = sorted(img_dir.glob("img*.nii.gz"))
    if not ct_files:
        raise FileNotFoundError(f"No CT files found in {img_dir}")

    all_metadata = []
    for ct_file in tqdm(ct_files, desc="Preprocessing"):
        # Match img0001.nii.gz -> label0001.nii.gz
        case_id = ct_file.name.replace("img", "").replace(".nii.gz", "")
        gt_file = label_dir / f"label{case_id}.nii.gz"

        if not gt_file.exists():
            logger.warning(f"Label file not found: {gt_file}, skipping")
            continue

        metadata = preprocess_patient(
            ct_file, gt_file, output_dir, target_spacing, hu_min, hu_max
        )
        all_metadata.append(metadata)

    logger.info(f"Preprocessed {len(all_metadata)} patients")
    return all_metadata


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw training data")
    parser.add_argument("--input", type=str, required=True, help="Path to a training directory containing img/ and label/")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    parser.add_argument(
        "--spacing", type=float, nargs=3, default=[1.5, 1.5, 2.0],
        help="Target voxel spacing in mm (x y z)",
    )
    parser.add_argument("--hu-min", type=float, default=-175.0, help="Min HU clipping value")
    parser.add_argument("--hu-max", type=float, default=250.0, help="Max HU clipping value")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    preprocess_all(
        args.input,
        args.output,
        target_spacing=tuple(args.spacing),
        hu_min=args.hu_min,
        hu_max=args.hu_max,
    )


if __name__ == "__main__":
    main()
