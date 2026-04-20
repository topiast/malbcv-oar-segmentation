# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pyyaml",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "torch>=2.0.0",
#     "monai[all]>=1.3.0",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import json
    import os
    import sys
    from functools import lru_cache
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Compose,
        CropForegroundd,
        EnsureChannelFirstd,
        EnsureTyped,
        Lambdad,
        LoadImaged,
        Orientationd,
        ScaleIntensityRanged,
        Spacingd,
    )

    plt.rcParams["figure.figsize"] = (18, 8)
    return (
        Compose,
        CropForegroundd,
        EnsureChannelFirstd,
        EnsureTyped,
        Lambdad,
        LoadImaged,
        Patch,
        Orientationd,
        Path,
        ScaleIntensityRanged,
        Spacingd,
        json,
        lru_cache,
        np,
        os,
        pd,
        plt,
        sliding_window_inference,
        sys,
        to_rgba,
        torch,
        yaml,
    )


@app.cell
def _(Path, os, sys):
    def _candidate_roots():
        seen = set()

        for base in [Path.cwd().resolve(), Path(__file__).resolve().parent]:
            for candidate in [base, *base.parents]:
                if candidate not in seen:
                    seen.add(candidate)
                    yield candidate

        for entry in list(sys.path):
            if not entry:
                continue
            try:
                candidate = Path(entry).resolve()
            except OSError:
                continue
            for root in [candidate, *candidate.parents]:
                if root not in seen:
                    seen.add(root)
                    yield root

    REPO_ROOT = None
    for candidate in _candidate_roots():
        if (candidate / "configs" / "maskmed_v2.yaml").exists() and (candidate / "src").is_dir():
            REPO_ROOT = candidate
            break

    if REPO_ROOT is None:
        raise FileNotFoundError("Could not locate repo root containing configs/maskmed_v2.yaml and src/")

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)
    return (REPO_ROOT,)


@app.cell
def _():
    from src.data.dataset import get_training_datalists
    from src.evaluation.metrics import evaluate_patient
    from src.models.factory import build_model
    from src.utils.config import get_class_names, get_foreground_label_map, resolve_config_paths

    return (
        build_model,
        evaluate_patient,
        get_class_names,
        get_foreground_label_map,
        get_training_datalists,
        resolve_config_paths,
    )


@app.cell
def _(mo):
    mo.md("""
    # Fold 0 Prediction Viewer

    This notebook uses the `results/checkpoints/fold_0/best_model.pth` checkpoint and
    treats the fold 0 validation split as the held-out data for visualization.

    That means the selectable cases below are the cases **not used to train** the fold 0 model.
    """)
    return


@app.cell
def _(REPO_ROOT, mo):
    stats_path = REPO_ROOT / "data" / "processed" / "RawData" / "Training" / "nnunet_ct_stats.json"
    mo.md(
        f"""
        **Detected repo root**: `{REPO_ROOT}`

        **Stats file**: `{stats_path}`

        **Stats file exists**: `{stats_path.exists()}`
        """
    )
    return


@app.cell
def _(
    Path,
    REPO_ROOT,
    get_class_names,
    get_foreground_label_map,
    get_training_datalists,
    json,
    pd,
    resolve_config_paths,
    yaml,
):
    # config_path = REPO_ROOT / "configs" / "baseline.yaml"
    config_path = REPO_ROOT / "configs" / "maskmed_v2.yaml"
    # checkpoint_path = REPO_ROOT / "results" / "checkpoints" / "fold_0" / "best_model.pth"
    # results/checkpoints_maskmed/fold_0/best_model.pth
    checkpoint_path = REPO_ROOT / "results" / "checkpoints_maskmed" / "fold_0" / "best_model.pth"
    #results/checkpoints_maskedmed_v1/fold_0/checkpoint_epoch_450.pth
    with open(config_path) as f:
        config = resolve_config_paths(yaml.safe_load(f), config_path.parent.parent)

    _stats_path = REPO_ROOT / "data" / "processed" / "RawData" / "Training" / "nnunet_ct_stats.json"
    if _stats_path.exists():
        with open(_stats_path) as _f:
            config["data"]["nnunet_ct_stats"] = json.load(_f)

    for _key in ["raw_dir", "train_dir", "processed_dir", "test_dir"]:
        if _key in config["data"]:
            config["data"][_key] = str(Path(config["data"][_key]).resolve())

    train_dir = Path(config["data"]["train_dir"])

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    _, val_files = get_training_datalists(
        train_dir,
        fold=0,
        num_folds=config["training"]["num_folds"],
        seed=config["training"]["seed"],
    )

    class_names = get_class_names(config)
    label_map = get_foreground_label_map(config)

    held_out_cases = []
    for item in val_files:
        image_path = Path(item["image"])
        label_path = Path(item["label"])
        case_id = image_path.stem.replace(".nii", "")
        held_out_cases.append(
            {
                "case_id": case_id,
                "image_path": image_path,
                "label_path": label_path,
            }
        )

    case_lookup = {case["case_id"]: case for case in held_out_cases}
    case_ids = [case["case_id"] for case in held_out_cases]

    held_out_df = pd.DataFrame(
        {
            "case_id": [case["case_id"] for case in held_out_cases],
            "image_path": [str(case["image_path"].relative_to(REPO_ROOT)) for case in held_out_cases],
            "label_path": [str(case["label_path"].relative_to(REPO_ROOT)) for case in held_out_cases],
        }
    )

    target_spacing = tuple(float(v) for v in config["data"]["target_spacing"])
    organ_options = ["All organs"] + [label_map[idx] for idx in sorted(label_map)]
    selected_case_default = case_ids[0]
    return (
        case_ids,
        case_lookup,
        checkpoint_path,
        class_names,
        config,
        config_path,
        held_out_df,
        label_map,
        organ_options,
        selected_case_default,
        target_spacing,
        train_dir,
    )


@app.cell
def _(
    checkpoint_path,
    config_path,
    held_out_df,
    mo,
    selected_case_default,
    train_dir,
):
    mo.vstack(
        [
            mo.md(
                f"""
                **Checkpoint**: `{checkpoint_path}`

                **Config**: `{config_path}`

                **Training pool**: `{train_dir}`

                **Held-out fold 0 cases**: `{len(held_out_df)}` total, default selection `{selected_case_default}`
                """
            ),
            held_out_df,
        ]
    )
    return


@app.cell
def _(build_model, checkpoint_path, config, torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _raw_model = model

    def model(x):
        out = _raw_model(x)
        return out["logits"] if isinstance(out, dict) else out

    return device, model


@app.cell
def _(
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    config,
    np,
):
    def _build_viewer_val_transforms(cfg):
        data_cfg = cfg["data"]
        normalization = data_cfg.get("normalization", "legacy_clip_01")

        if normalization == "nnunet_ct":
            stats = data_cfg.get("nnunet_ct_stats")
            if stats is None:
                raise ValueError("Notebook expected data.nnunet_ct_stats to be loaded before building transforms")

            def _normalize_ct(image):
                image = np.clip(image, stats["percentile_00_5"], stats["percentile_99_5"])
                return ((image - stats["mean"]) / max(stats["std"], 1e-8)).astype(np.float32)

            intensity_transform = Lambdad(keys=["image"], func=_normalize_ct)
        else:
            intensity_min, intensity_max = data_cfg["intensity_clip"]
            intensity_transform = ScaleIntensityRanged(
                keys=["image"],
                a_min=float(intensity_min),
                a_max=float(intensity_max),
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )

        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=cfg["data"]["target_spacing"],
                mode=("bilinear", "nearest"),
            ),
            intensity_transform,
            CropForegroundd(keys=["image", "label"], source_key="image"),
            EnsureTyped(keys=["image", "label"]),
        ])

    eval_cfg = config["evaluation"]
    val_transforms = _build_viewer_val_transforms(config)
    return eval_cfg, val_transforms


@app.cell
def _(class_names, to_rgba):
    palette_names = [
        "tomato",
        "royalblue",
        "seagreen",
        "gold",
        "darkorchid",
        "darkorange",
        "firebrick",
        "deepskyblue",
        "mediumvioletred",
        "peru",
        "yellowgreen",
        "slateblue",
        "teal",
    ]
    label_colors = {
        idx: to_rgba(palette_names[(idx - 1) % len(palette_names)], alpha=1.0)
        for idx, _name in enumerate(class_names)
        if idx != 0
    }
    return (label_colors,)


@app.cell
def _(
    case_lookup,
    device,
    eval_cfg,
    evaluate_patient,
    label_map,
    lru_cache,
    model,
    np,
    pd,
    sliding_window_inference,
    target_spacing,
    torch,
    val_transforms,
):
    def _to_plain_tensor(value):
        return value.as_tensor() if hasattr(value, "as_tensor") else value

    def _best_slice(mask):
        return {
            "axial": int(np.argmax(np.sum(mask > 0, axis=(0, 1)))),
            "coronal": int(np.argmax(np.sum(mask > 0, axis=(0, 2)))),
            "sagittal": int(np.argmax(np.sum(mask > 0, axis=(1, 2)))),
        }

    @lru_cache(maxsize=8)
    def load_case_bundle(case_id):
        case = case_lookup[case_id]
        sample = val_transforms(
            {
                "image": str(case["image_path"]),
                "label": str(case["label_path"]),
            }
        )

        image_tensor = _to_plain_tensor(sample["image"]).to(dtype=torch.float32)
        label_tensor = _to_plain_tensor(sample["label"]).to(dtype=torch.long)

        with torch.no_grad():
            outputs = sliding_window_inference(
                inputs=image_tensor.unsqueeze(0).to(device),
                roi_size=eval_cfg["sliding_window_size"],
                sw_batch_size=eval_cfg["sw_batch_size"],
                predictor=model,
                overlap=eval_cfg["overlap"],
            )

        pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.int32)
        image = image_tensor.squeeze(0).cpu().numpy()
        label = label_tensor.squeeze(0).cpu().numpy().astype(np.int32)

        metrics = evaluate_patient(
            pred,
            label,
            spacing=target_spacing,
            tolerance_mm=3.0,
            label_map=label_map,
        )

        metrics_df = (
            pd.DataFrame(
                [
                    {
                        "organ": organ,
                        "dice": values["dice"],
                        "hd95_mm": values["hd95"],
                        "surface_dice": values["surface_dice"],
                    }
                    for organ, values in metrics.items()
                ]
            )
            .sort_values(["dice", "surface_dice"], ascending=[False, False])
            .reset_index(drop=True)
        )

        foreground = (label > 0) | (pred > 0)

        return {
            "case_id": case_id,
            "image": image,
            "label": label,
            "prediction": pred,
            "metrics_df": metrics_df,
            "best_slice": _best_slice(foreground),
            "shape": image.shape,
        }

    return (load_case_bundle,)


@app.cell
def _(case_ids, mo, organ_options, selected_case_default):
    case_selector = mo.ui.dropdown(
        options=case_ids,
        value=selected_case_default,
        label="Held-out fold 0 case",
    )
    plane_selector = mo.ui.dropdown(
        options=["axial", "coronal", "sagittal"],
        value="axial",
        label="Plane",
    )
    organ_selector = mo.ui.dropdown(
        options=organ_options,
        value="All organs",
        label="Organ focus",
    )

    mo.hstack([case_selector, plane_selector, organ_selector], justify="start")
    return case_selector, organ_selector, plane_selector


@app.cell
def _(case_selector, load_case_bundle):
    case_bundle = load_case_bundle(case_selector.value)
    return (case_bundle,)


@app.cell
def _(case_bundle, mo, plane_selector):
    plane = plane_selector.value
    axis_map = {
        "axial": 2,
        "coronal": 1,
        "sagittal": 0,
    }
    max_index = int(case_bundle["shape"][axis_map[plane]] - 1)
    default_index = int(min(case_bundle["best_slice"][plane], max_index))
    slice_selector = mo.ui.slider(
        start=0,
        stop=max_index,
        step=1,
        value=default_index,
        label=f"{plane.title()} slice",
    )
    slice_selector
    return (slice_selector,)


@app.cell
def _(mo):
    gt_alpha = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.55, label="Ground-truth alpha")
    pred_alpha = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.55, label="Prediction alpha")
    error_alpha = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.75, label="Error alpha")
    window_low = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.0, label="CT display min")
    window_high = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=1.0, label="CT display max")

    mo.vstack(
        [
            mo.hstack([gt_alpha, pred_alpha, error_alpha], justify="start"),
            mo.hstack([window_low, window_high], justify="start"),
        ]
    )
    return error_alpha, gt_alpha, pred_alpha, window_high, window_low


@app.cell
def _(case_bundle, mo, target_spacing):
    metrics_preview = case_bundle["metrics_df"].copy()
    metrics_preview["dice"] = metrics_preview["dice"].map(lambda value: f"{value:.4f}")
    metrics_preview["hd95_mm"] = metrics_preview["hd95_mm"].map(
        lambda value: "inf" if value == float("inf") else f"{value:.2f}"
    )
    metrics_preview["surface_dice"] = metrics_preview["surface_dice"].map(lambda value: f"{value:.4f}")

    mo.vstack(
        [
            mo.md(
                f"""
                **Selected case**: `{case_bundle["case_id"]}`

                **Preprocessed volume shape**: `{case_bundle["shape"]}`

                **Evaluation spacing**: `{target_spacing}` mm
                """
            ),
            metrics_preview,
        ]
    )
    return


@app.cell
def _(label_map, organ_selector):
    selected_label = None
    if organ_selector.value != "All organs":
        inverse_label_map = {name: idx for idx, name in label_map.items()}
        selected_label = inverse_label_map[organ_selector.value]
    return (selected_label,)


@app.cell
def _(
    Patch,
    case_bundle,
    error_alpha,
    gt_alpha,
    label_colors,
    label_map,
    np,
    organ_selector,
    pd,
    plane_selector,
    plt,
    pred_alpha,
    selected_label,
    slice_selector,
    window_high,
    window_low,
):
    def _extract_plane(volume, plane, index):
        if plane == "axial":
            return volume[:, :, index]
        if plane == "coronal":
            return volume[:, index, :]
        return volume[index, :, :]

    def _to_display(slice_2d):
        return slice_2d.T

    def _masked_labels(slice_2d):
        if selected_label is None:
            return slice_2d
        return np.where(slice_2d == selected_label, slice_2d, 0)

    def _rgba_overlay(mask_slice, alpha):
        overlay = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
        for label, color in label_colors.items():
            rgb = color[:3]
            overlay[mask_slice == label] = (*rgb, alpha)
        return overlay

    image_slice = _extract_plane(case_bundle["image"], plane_selector.value, slice_selector.value)
    label_slice = _masked_labels(_extract_plane(case_bundle["label"], plane_selector.value, slice_selector.value))
    pred_slice = _masked_labels(_extract_plane(case_bundle["prediction"], plane_selector.value, slice_selector.value))

    image_slice = _to_display(image_slice)
    label_slice = _to_display(label_slice)
    pred_slice = _to_display(pred_slice)

    if window_low.value >= window_high.value:
        display_min = 0.0
        display_max = 1.0
    else:
        display_min = window_low.value
        display_max = window_high.value

    error = np.zeros((*label_slice.shape, 4), dtype=np.float32)
    false_positive = (pred_slice > 0) & (pred_slice != label_slice)
    false_negative = (label_slice > 0) & (pred_slice != label_slice)
    error[false_positive] = (1.0, 0.1, 0.1, error_alpha.value)
    error[false_negative] = (0.1, 0.35, 1.0, error_alpha.value)

    union_overlay = np.zeros((*label_slice.shape, 4), dtype=np.float32)
    union_overlay[label_slice > 0] = (0.0, 1.0, 1.0, gt_alpha.value)
    union_overlay[pred_slice > 0] = (1.0, 0.0, 1.0, pred_alpha.value)
    overlap = (label_slice > 0) & (pred_slice > 0)
    union_overlay[overlap] = (1.0, 1.0, 1.0, max(gt_alpha.value, pred_alpha.value))

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    titles = [
        f"CT ({plane_selector.value}, slice {slice_selector.value})",
        "Ground Truth",
        "Prediction",
        "GT + Pred + Errors",
    ]

    gt_overlay = _rgba_overlay(label_slice, gt_alpha.value)
    pred_overlay = _rgba_overlay(pred_slice, pred_alpha.value)

    for ax, title in zip(axes, titles):
        ax.imshow(image_slice, cmap="gray", origin="lower", vmin=display_min, vmax=display_max)
        ax.set_title(title)
        ax.axis("off")

    axes[1].imshow(gt_overlay, origin="lower")
    axes[2].imshow(pred_overlay, origin="lower")
    axes[3].imshow(union_overlay, origin="lower")
    axes[3].imshow(error, origin="lower")

    focus_name = organ_selector.value
    fig.suptitle(f"{case_bundle['case_id']} | focus: {focus_name}", fontsize=14)

    legend_handles = [
        Patch(facecolor=label_colors[idx], label=f"{idx}: {name}")
        for idx, name in sorted(label_map.items())
        if idx in label_colors
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(7, len(legend_handles)),
        fontsize=8,
        framealpha=0.8,
        bbox_to_anchor=(0.5, -0.08),
    )
    plt.tight_layout()

    summary = pd.DataFrame(
        {
            "view": ["visible gt voxels", "visible predicted voxels", "false positives", "false negatives"],
            "count": [
                int(np.sum(label_slice > 0)),
                int(np.sum(pred_slice > 0)),
                int(np.sum(false_positive)),
                int(np.sum(false_negative)),
            ],
        }
    )
    return fig, summary


@app.cell
def _(fig, mo, summary):
    mo.vstack(
        [
            fig,
            summary,
            mo.md(
                """
                In the combined panel, cyan marks ground truth, magenta marks prediction,
                white marks overlap, red marks false positives, and blue marks false negatives.
                """
            ),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
