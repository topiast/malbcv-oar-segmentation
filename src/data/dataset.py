"""Dataset utilities for supported segmentation directory layouts."""

from pathlib import Path

from sklearn.model_selection import KFold


def discover_training_pairs(data_dir: str | Path) -> list[dict]:
    """
    Discover image/label pairs from known directory layouts.

    Supported layouts:
    - Processed patient subdirs: ``<case>/<case>.nii.gz`` + ``<case>/GT.nii.gz``
    - Raw BTCV: ``img/img0001.nii.gz`` + ``label/label0001.nii.gz``
    - Averaged binary data: ``averaged-training-images/*_avg.nii.gz`` +
      ``averaged-training-labels/*_avg_seg.nii.gz``
    """
    data_dir = Path(data_dir)
    data_list: list[dict] = []

    patient_dirs = (
        sorted(d for d in data_dir.iterdir() if d.is_dir()) if data_dir.exists() else []
    )
    for patient_dir in patient_dirs:
        ct_file = patient_dir / f"{patient_dir.name}.nii.gz"
        gt_file = patient_dir / "GT.nii.gz"
        if ct_file.exists() and gt_file.exists():
            data_list.append({"image": str(ct_file), "label": str(gt_file)})

    if data_list:
        return data_list

    img_dir = data_dir / "img"
    label_dir = data_dir / "label"
    if img_dir.exists() and label_dir.exists():
        for ct_file in sorted(img_dir.glob("img*.nii.gz")):
            case_id = ct_file.name.replace("img", "").replace(".nii.gz", "")
            gt_file = label_dir / f"label{case_id}.nii.gz"
            if gt_file.exists():
                data_list.append({"image": str(ct_file), "label": str(gt_file)})

    if data_list:
        return data_list

    avg_img_dir = data_dir / "averaged-training-images"
    avg_label_dir = data_dir / "averaged-training-labels"
    if avg_img_dir.exists() and avg_label_dir.exists():
        for ct_file in sorted(avg_img_dir.glob("*_avg.nii.gz")):
            gt_file = avg_label_dir / ct_file.name.replace("_avg.nii.gz", "_avg_seg.nii.gz")
            if gt_file.exists():
                data_list.append({"image": str(ct_file), "label": str(gt_file)})

    return data_list


def get_training_datalists(
    data_dir: str | Path,
    fold: int = 0,
    num_folds: int = 5,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Generate train/val file lists for a given cross-validation fold.

    Args:
        data_dir: Path to a supported training dataset root.
        fold: Which fold to use for validation (0 to num_folds-1).
        num_folds: Total number of cross-validation folds.
        seed: Random seed for reproducible splits.

    Returns:
        Tuple of (train_files, val_files), each a list of dicts.
    """
    data_list = discover_training_pairs(data_dir)

    if not data_list:
        raise FileNotFoundError(
            f"No valid training data found in {data_dir}. "
            "Expected processed patient subdirs, raw BTCV img/label directories, "
            "or averaged-training-images plus averaged-training-labels."
        )

    if fold >= num_folds:
        raise ValueError(f"fold={fold} must be < num_folds={num_folds}")

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = list(kf.split(data_list))
    train_idx, val_idx = splits[fold]

    train_files = [data_list[i] for i in train_idx]
    val_files = [data_list[i] for i in val_idx]

    return train_files, val_files


def get_btcv_datalists(
    data_dir: str | Path,
    fold: int = 0,
    num_folds: int = 5,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Backward-compatible alias for older imports."""
    return get_training_datalists(data_dir, fold=fold, num_folds=num_folds, seed=seed)


def get_test_datalist(test_dir: str | Path) -> list[dict]:
    """Generate file list for test volumes across supported layouts."""
    test_dir = Path(test_dir)

    test_files = sorted(
        f
        for f in test_dir.glob("*.nii.gz")
        if "seg" not in f.name.lower() and f.name != "GT.nii.gz"
    )

    if not test_files:
        test_files = sorted(
            f
            for f in test_dir.glob("*/*.nii.gz")
            if "seg" not in f.name.lower() and f.name != "GT.nii.gz"
        )

    if not test_files:
        raise FileNotFoundError(f"No test NIfTI files found in {test_dir}")

    return [{"image": str(f)} for f in test_files]
