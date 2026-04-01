#!/usr/bin/env python
"""
Inference script: run trained model on new CT volumes.

Usage:
    python scripts/predict.py --config configs/baseline.yaml --checkpoint results/checkpoints/fold_0/best_model.pth --input data/raw/test --output results/predictions
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.transforms import get_inference_transforms
from src.data.dataset import get_test_datalist
from src.export.rtstruct_export import prediction_to_nifti
from src.models.unet3d import build_model

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run inference on CT volumes")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Directory with CT NIfTI files or patient dirs")
    parser.add_argument("--output", type=str, default="results/predictions", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Model loaded from {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        nifti_files = [Path(item["image"]) for item in get_test_datalist(input_dir)]
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info(f"Found {len(nifti_files)} volumes to process")

    # Transforms
    inference_transforms = get_inference_transforms(config)
    eval_cfg = config["evaluation"]

    for nifti_path in nifti_files:
        patient_name = nifti_path.stem.replace(".nii", "")
        logger.info(f"Processing {patient_name}...")

        # Create single-item dataset
        data = [{"image": str(nifti_path)}]
        ds = Dataset(data=data, transform=inference_transforms)
        loader = DataLoader(ds, batch_size=1, num_workers=0)

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)

                # Sliding window inference
                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=eval_cfg["sliding_window_size"],
                    sw_batch_size=eval_cfg["sw_batch_size"],
                    predictor=model,
                    overlap=eval_cfg["overlap"],
                )

                # Convert to label map
                pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Save prediction
        import SimpleITK as sitk
        ref_image = sitk.ReadImage(str(nifti_path))
        out_path = output_dir / f"{patient_name}.nii.gz"
        prediction_to_nifti(pred, ref_image, out_path)
        logger.info(f"  Saved prediction to {out_path}")

    logger.info(f"Inference complete. {len(nifti_files)} predictions saved to {output_dir}")


if __name__ == "__main__":
    main()
