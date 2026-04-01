#!/usr/bin/env python
"""Mirror a NIfTI dataset tree with cleaned qform/sform header metadata."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)


def iter_nifti_files(root: Path) -> list[Path]:
    """Return all .nii and .nii.gz files under a dataset root."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and (path.name.endswith(".nii") or path.name.endswith(".nii.gz"))
    )


def build_normalized_image(image: nib.spatialimages.SpatialImage) -> nib.Nifti1Image:
    """Return a copy of a NIfTI image with valid qform/sform metadata."""
    normalized = nib.Nifti1Image.from_image(image)
    affine = np.asarray(image.affine, dtype=float)

    # Write scanner-space transforms explicitly so nibabel no longer needs
    # to repair invalid qfac/qform metadata at load time.
    normalized.set_qform(affine, code=1)
    normalized.set_sform(affine, code=1)
    return normalized


def normalize_tree(input_root: Path, output_root: Path, overwrite: bool = False) -> int:
    """Normalize all NIfTI headers under input_root into output_root."""
    nifti_files = iter_nifti_files(input_root)
    if not nifti_files:
        raise FileNotFoundError(f"No NIfTI files found under {input_root}")

    written = 0
    for input_path in tqdm(nifti_files, desc="Normalizing headers"):
        output_path = output_root / input_path.relative_to(input_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not overwrite:
            continue

        image = nib.load(str(input_path))
        normalized = build_normalized_image(image)
        nib.save(normalized, str(output_path))
        written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize NIfTI qform/sform headers once into a processed dataset tree.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/raw/RawData"),
        help="Path to the original raw dataset directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/RawData"),
        help="Path to write the normalized dataset tree.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite files even if the normalized output already exists.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info(f"Input root: {args.input_root}")
    logger.info(f"Output root: {args.output_root}")

    written = normalize_tree(args.input_root, args.output_root, overwrite=args.overwrite)
    logger.info(f"Header normalization complete. Wrote {written} files.")


if __name__ == "__main__":
    main()
