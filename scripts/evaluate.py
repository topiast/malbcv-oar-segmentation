#!/usr/bin/env python
"""
Evaluation script: compute metrics on saved predictions vs ground truth.

Usage:
    python scripts/evaluate.py --pred-dir results/predictions --gt-dir data/processed/train --output results/metrics/scores.csv
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.metrics import evaluate_patient
from src.utils.config import get_foreground_label_map, resolve_config_paths

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions")
    parser.add_argument("--config", type=str, default=None, help="Optional config YAML for class names")
    parser.add_argument("--pred-dir", type=str, required=True, help="Directory with predicted NIfTI masks")
    parser.add_argument("--gt-dir", type=str, required=True, help="Directory with ground truth data (processed or raw)")
    parser.add_argument("--output", type=str, default="results/metrics/scores.csv", help="Output CSV path")
    parser.add_argument("--tolerance", type=float, default=3.0, help="Surface Dice tolerance in mm")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_map = None

    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = project_root / config_path

        with config_path.open() as f:
            config = resolve_config_paths(yaml.safe_load(f), project_root)
        label_map = get_foreground_label_map(config)

    # Find prediction files
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_files:
        logger.error(f"No prediction files found in {pred_dir}")
        sys.exit(1)

    all_results = []
    for pred_file in pred_files:
        patient_name = pred_file.stem.replace(".nii", "")

        # Find matching ground truth across supported directory layouts:
        # 1. Processed: gt_dir/<patient_name>/GT.nii.gz
        # 2. Raw img/label layout: gt_dir/label/label<id>.nii.gz
        gt_file = gt_dir / patient_name / "GT.nii.gz"
        if not gt_file.exists():
            # Try raw img/label layout (img0001 -> label0001)
            case_id = patient_name.replace("img", "")
            gt_file = gt_dir / "label" / f"label{case_id}.nii.gz"
        if not gt_file.exists():
            gt_file = gt_dir / f"{patient_name}_seg.nii.gz"
        if not gt_file.exists():
            logger.warning(f"No ground truth for {patient_name}, skipping")
            continue

        # Load volumes
        pred_nii = nib.load(str(pred_file))
        gt_nii = nib.load(str(gt_file))

        pred_data = np.round(pred_nii.get_fdata()).astype(np.int32)
        gt_data = np.round(gt_nii.get_fdata()).astype(np.int32)

        spacing = gt_nii.header.get_zooms()[:3]

        # Compute metrics
        results = evaluate_patient(
            pred_data,
            gt_data,
            spacing,
            tolerance_mm=args.tolerance,
            label_map=label_map,
        )

        for organ, metrics in results.items():
            row = {
                "patient": patient_name,
                "organ": organ,
                "dice": f"{metrics['dice']:.4f}",
                "hd95": f"{metrics['hd95']:.2f}",
                "surface_dice": f"{metrics['surface_dice']:.4f}",
            }
            all_results.append(row)
            logger.info(f"{patient_name} | {organ}: Dice={metrics['dice']:.4f}, HD95={metrics['hd95']:.2f}mm")

    # Write CSV
    if all_results:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["patient", "organ", "dice", "hd95", "surface_dice"])
            writer.writeheader()
            writer.writerows(all_results)
        logger.info(f"Results saved to {output_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY (mean ± std)")
        print("=" * 60)
        summary_names = label_map.values() if label_map else sorted({r["organ"] for r in all_results})
        for organ_name in summary_names:
            organ_rows = [r for r in all_results if r["organ"] == organ_name]
            if organ_rows:
                dices = [float(r["dice"]) for r in organ_rows]
                hd95s = [float(r["hd95"]) for r in organ_rows if float(r["hd95"]) != float("inf")]
                print(
                    f"{organ_name:22s} | Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f} | "
                    f"HD95: {np.mean(hd95s):.2f} ± {np.std(hd95s):.2f} mm"
                )
    else:
        logger.warning("No results computed")


if __name__ == "__main__":
    main()
