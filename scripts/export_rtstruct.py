#!/usr/bin/env python
"""
Export predictions to DICOM RT Structure Set format.

Usage:
    python scripts/export_rtstruct.py --ct-dicom-dir /path/to/ct/dicom --prediction results/predictions/Patient_01.nii.gz --output results/exports/Patient_01_RTSTRUCT.dcm
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.export.rtstruct_export import build_roi_config, export_rtstruct
from src.utils.config import get_class_names, resolve_config_paths

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export predictions to DICOM RTSTRUCT")
    parser.add_argument("--config", type=str, default=None, help="Optional config YAML for ROI names")
    parser.add_argument("--ct-dicom-dir", type=str, required=True, help="Directory with original CT DICOM series")
    parser.add_argument("--prediction", type=str, required=True, help="Path to prediction NIfTI mask")
    parser.add_argument("--output", type=str, required=True, help="Output RTSTRUCT DICOM file path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    roi_config = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = project_root / config_path

        with config_path.open() as f:
            config = resolve_config_paths(yaml.safe_load(f), project_root)
        roi_config = build_roi_config(get_class_names(config))

    export_rtstruct(args.ct_dicom_dir, args.prediction, args.output, roi_config=roi_config)


if __name__ == "__main__":
    main()
