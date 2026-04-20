# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nibabel",
#     "numpy",
#     "matplotlib",
#     "seaborn",
#     "tqdm",
#     "pandas",
# ]
# ///

import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import nibabel as nib
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from tqdm import tqdm
    import pandas as pd

    plt.rcParams['figure.figsize'] = (14, 8)
    sns.set_style('whitegrid')
    return Path, nib, np, pd, plt, tqdm


@app.cell
def _(mo):
    mo.md("""
    # MALBCV Data Exploration

    This notebook explores the MALBCV training data available in this workspace. It prefers the
    header-normalized processed dataset under `data/processed`, and falls back to the original
    raw Synapse files under `data/raw` if needed.
    """)
    return


@app.cell
def _(Path):
    _NOTEBOOK_DIR = Path(__file__).parent
    DATA_PROCESSED_ROOT = _NOTEBOOK_DIR.parent / "data/processed"
    DATA_RAW_ROOT = _NOTEBOOK_DIR.parent / "data/raw"

    def _btcv_label_path(label_dir, image_path):
        case_id = image_path.name.replace("img", "").replace(".nii.gz", "")
        return case_id, label_dir / f"label{case_id}.nii.gz"

    def _build_btcv_layout(root, name):
        training_dir = root / "RawData/Training"
        label_dir = training_dir / "label"
        return {
            "name": name,
            "img_dir": training_dir / "img",
            "label_dir": label_dir,
            "image_pattern": "img*.nii.gz",
            "label_for_image": lambda image_path, label_dir=label_dir: _btcv_label_path(label_dir, image_path),
            "label_names": {
                0: "Background", 1: "Spleen", 2: "Right Kidney", 3: "Left Kidney",
                4: "Gallbladder", 5: "Esophagus", 6: "Liver", 7: "Stomach",
                8: "Aorta", 9: "IVC", 10: "Portal & Splenic Veins", 11: "Pancreas",
                12: "Right Adrenal Gland", 13: "Left Adrenal Gland",
            },
            "label_colors": {
                1: "darkred", 2: "green", 3: "limegreen", 4: "cyan", 5: "red",
                6: "saddlebrown", 7: "orange", 8: "blue", 9: "purple",
                10: "magenta", 11: "yellow", 12: "salmon", 13: "lightgreen",
            },
            "intensity_label": "HU",
            "expected_cases": 30,
        }

    LAYOUTS = [
        _build_btcv_layout(DATA_RAW_ROOT, "MALBCV training (BTCV layout)"),
    ]
    return (LAYOUTS,)


@app.cell
def _(LAYOUTS):
    def _discover_training_pairs(layouts):
        checked = []
        for layout in layouts:
            img_dir = layout["img_dir"]
            label_dir = layout["label_dir"]
            checked.append((layout["name"], img_dir, label_dir))
            if not img_dir.exists() or not label_dir.exists():
                continue

            pairs = []
            for ct_path in sorted(img_dir.glob(layout["image_pattern"])):
                case_id, gt_path = layout["label_for_image"](ct_path)
                if gt_path.exists():
                    pairs.append((case_id, ct_path, gt_path))

            if pairs:
                return layout, pairs, checked

        checked_lines = [f"- {name}: img={img_dir} label={label_dir}" for name, img_dir, label_dir in checked]
        raise FileNotFoundError("No training image/label pairs found. Checked:\n" + "\n".join(checked_lines))

    ACTIVE_LAYOUT, case_pairs, _checked_layouts = _discover_training_pairs(LAYOUTS)
    IMG_DIR = ACTIVE_LAYOUT["img_dir"]
    LABEL_DIR = ACTIVE_LAYOUT["label_dir"]
    LABEL_NAMES = ACTIVE_LAYOUT["label_names"]
    LABEL_COLORS = ACTIVE_LAYOUT["label_colors"]
    INTENSITY_LABEL = ACTIVE_LAYOUT["intensity_label"]
    EXPECTED_CASES = ACTIVE_LAYOUT["expected_cases"]
    ct_files = [ct_path for _, ct_path, _ in case_pairs]

    print(f"Using layout: {ACTIVE_LAYOUT['name']}")
    print(f"Image dir: {IMG_DIR}")
    print(f"Label dir: {LABEL_DIR}")
    print(f"Found {len(case_pairs)} CT/label pairs")
    print(f"Configured labels: {LABEL_NAMES}")
    print(f"Intensity label: {INTENSITY_LABEL}")
    return (
        EXPECTED_CASES,
        INTENSITY_LABEL,
        LABEL_COLORS,
        LABEL_NAMES,
        case_pairs,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Load and Inspect a Single Patient
    """)
    return


@app.cell
def _(INTENSITY_LABEL, LABEL_NAMES, case_pairs, nib, np):
    _case_id, _ct_file, _gt_file = case_pairs[0]

    _ct = nib.load(str(_ct_file))
    _gt = nib.load(str(_gt_file))

    ct_data = _ct.get_fdata()
    gt_data = _gt.get_fdata()
    spacing = _ct.header.get_zooms()
    ct_file = _ct_file
    case_id = _case_id
    unique_labels = [int(v) for v in np.unique(gt_data)]

    print(f"Case ID: {case_id}")
    print(f"CT shape: {ct_data.shape}")
    print(f"GT shape: {gt_data.shape}")
    print(f"Voxel spacing: {spacing} mm")
    print(f"{INTENSITY_LABEL} range: [{ct_data.min():.0f}, {ct_data.max():.0f}]")
    print(f"Unique labels: {unique_labels}")
    print(f"Affine matrix:\n{_ct.affine}")

    print("\nVoxel counts per label:")
    for _label, _name in LABEL_NAMES.items():
        _count = int(np.sum(gt_data == _label))
        print(f"  {_name:25s}: {_count:>10,} voxels ({100 * _count / gt_data.size:.3f}%)")
    return ct_data, ct_file, gt_data, spacing


@app.cell
def _(LABEL_COLORS, ct_data, ct_file, gt_data, np, plt, spacing):
    def _plot_views(ct_data, gt_data, patient_name, spacing):
        """Plot axial, sagittal, and coronal views with label overlay."""
        sx, sy, sz = spacing[:3]

        fg_coords = np.argwhere(gt_data > 0)
        if len(fg_coords) == 0:
            print(f"No foreground in {patient_name}")
            return None
        center = fg_coords.mean(axis=0).astype(int)

        rgba_colors = {
            label: plt.matplotlib.colors.to_rgba(color, alpha=0.4)
            for label, color in LABEL_COLORS.items()
        }

        slices = [
            (ct_data[:, :, center[2]], gt_data[:, :, center[2]], f"Axial (z={center[2]})", sy / sx),
            (ct_data[:, center[1], :], gt_data[:, center[1], :], f"Coronal (y={center[1]})", sz / sx),
            (ct_data[center[0], :, :], gt_data[center[0], :, :], f"Sagittal (x={center[0]})", sz / sy),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(patient_name, fontsize=14)

        for ax, (ct_sl, gt_sl, title, aspect) in zip(axes, slices):
            ax.imshow(ct_sl.T, cmap="gray", origin="lower", aspect=aspect)
            overlay = np.zeros((*ct_sl.T.shape, 4))
            for label, color in rgba_colors.items():
                overlay[gt_sl.T == label] = color
            ax.imshow(overlay, origin="lower", aspect=aspect)
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()
        return fig

    _fig_views = _plot_views(ct_data, gt_data, ct_file.stem, spacing)
    _fig_views
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Dataset-Wide Statistics
    """)
    return


@app.cell
def _(LABEL_NAMES, case_pairs, nib, np, tqdm):
    stats = []

    for _case_id, _ct_path, _gt_path in tqdm(case_pairs, desc="Computing stats"):
        _ct = nib.load(str(_ct_path))
        _gt = nib.load(str(_gt_path))
        _ct_data = _ct.get_fdata()
        _gt_data = _gt.get_fdata()
        _spacing = _ct.header.get_zooms()

        _info = {
            "patient": _case_id,
            "shape_x": _ct_data.shape[0],
            "shape_y": _ct_data.shape[1],
            "shape_z": _ct_data.shape[2],
            "spacing_x": float(_spacing[0]),
            "spacing_y": float(_spacing[1]),
            "spacing_z": float(_spacing[2]),
            "intensity_min": float(_ct_data.min()),
            "intensity_max": float(_ct_data.max()),
            "intensity_mean": float(_ct_data.mean()),
            "intensity_std": float(_ct_data.std()),
        }

        _total_voxels = _gt_data.size
        for _label, _name in LABEL_NAMES.items():
            _key = _name.lower().replace(" ", "_").replace("&", "and")
            _count = int(np.sum(_gt_data == _label))
            _info[f"voxels_{_key}"] = _count
            _info[f"pct_{_key}"] = 100 * _count / _total_voxels

        _unexpected = sorted(int(v) for v in np.unique(_gt_data) if int(v) not in LABEL_NAMES)
        _info["unexpected_labels"] = _unexpected
        stats.append(_info)

    print(f"Collected stats for {len(stats)} patients")
    return (stats,)


@app.cell
def _(pd, stats):
    df = pd.DataFrame(stats)
    df.set_index('patient', inplace=True)
    df.head()
    return (df,)


@app.cell
def _(df, plt):
    fig_spacing, axes_spacing = plt.subplots(1, 3, figsize=(15, 4))
    for _ax, _dim in zip(axes_spacing, ['x', 'y', 'z']):
        _ax.hist(df[f'spacing_{_dim}'], bins=20, edgecolor='black')
        _ax.set_title(f'{_dim.upper()} spacing (mm)')
        _ax.set_xlabel('mm')
    fig_spacing.suptitle('Voxel Spacing Distribution', fontsize=14)
    plt.tight_layout()

    print("Spacing ranges:")
    for _dim in ['x', 'y', 'z']:
        _col = f'spacing_{_dim}'
        print(f"  {_dim}: [{df[_col].min():.2f}, {df[_col].max():.2f}] mm (mean: {df[_col].mean():.2f})")

    fig_spacing
    return


@app.cell
def _(df, plt):
    fig_dims, axes_dims = plt.subplots(1, 3, figsize=(15, 4))
    for _ax, _dim in zip(axes_dims, ['x', 'y', 'z']):
        _ax.hist(df[f'shape_{_dim}'], bins=20, edgecolor='black')
        _ax.set_title(f'{_dim.upper()} dimension (voxels)')
    fig_dims.suptitle('Volume Dimensions', fontsize=14)
    plt.tight_layout()
    fig_dims
    return


@app.cell
def _(LABEL_COLORS, LABEL_NAMES, df, plt):
    _foreground_keys = [
        (label, name.lower().replace(" ", "_").replace("&", "and"), name)
        for label, name in LABEL_NAMES.items()
        if label != 0
    ]

    _label_pcts = [df[f"pct_{key}"].mean() for _, key, _ in _foreground_keys]
    _label_names = [name for _, _, name in _foreground_keys]
    _label_colors = [LABEL_COLORS.get(label, "steelblue") for label, _, _ in _foreground_keys]

    fig_imbalance, ax_imbalance = plt.subplots(figsize=(max(6, 2 + 1.5 * len(_label_names)), 5))
    _bars = ax_imbalance.bar(_label_names, _label_pcts, color=_label_colors, edgecolor="black")
    ax_imbalance.set_ylabel("% of total voxels")
    ax_imbalance.set_title("Mean Label Size (% of total volume)")
    ax_imbalance.tick_params(axis="x", rotation=45)
    for _bar, _pct in zip(_bars, _label_pcts):
        ax_imbalance.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.01, f"{_pct:.3f}%", ha="center", fontsize=9)
    plt.tight_layout()

    print("Relative label distribution (excluding background):")
    _total_fg = sum(df[f"voxels_{key}"].mean() for _, key, _ in _foreground_keys)
    for _, _key, _name in _foreground_keys:
        _mean_voxels = df[f"voxels_{_key}"].mean()
        _share = 100 * _mean_voxels / _total_fg if _total_fg else 0.0
        print(f"  {_name:20s}: {_share:5.1f}%")

    fig_imbalance
    return


@app.cell
def _(LABEL_COLORS, LABEL_NAMES, df, plt):
    _label_keys = [
        (label, name.lower().replace(' ', '_').replace('&', 'and'), name)
        for label, name in LABEL_NAMES.items()
    ]

    _total_counts = [df[f'voxels_{key}'].sum() for _, key, _ in _label_keys]
    _total_voxels = sum(_total_counts)
    _label_names_all = [name for _, _, name in _label_keys]
    _label_colors_all = [LABEL_COLORS.get(label, 'lightgray') if label != 0 else 'dimgray' for label, _, _ in _label_keys]
    _label_pcts_all = [100 * count / _total_voxels for count in _total_counts]

    fig_agg, axes_agg = plt.subplots(1, 2, figsize=(16, 5))

    axes_agg[0].bar(_label_names_all, _total_counts, color=_label_colors_all, edgecolor='black')
    axes_agg[0].set_title('Total Voxel Count by Label')
    axes_agg[0].set_ylabel('Voxel count')
    axes_agg[0].tick_params(axis='x', rotation=45)

    axes_agg[1].bar(_label_names_all, _label_pcts_all, color=_label_colors_all, edgecolor='black')
    axes_agg[1].set_title('Dataset Label Share (%)')
    axes_agg[1].set_ylabel('% of all voxels')
    axes_agg[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    print('Aggregate label distribution:')
    for (_, _, _name), _count, _pct in zip(_label_keys, _total_counts, _label_pcts_all):
        print(f'  {_name:20s}: {_count:>12,} voxels ({_pct:6.3f}%)')

    fig_agg
    return


@app.cell
def _(INTENSITY_LABEL, case_pairs, nib, np, plt):
    fig_intensity, ax_intensity = plt.subplots(figsize=(10, 5))
    for _case_id, _ct_path, _ in case_pairs[:5]:
        _ct = nib.load(str(_ct_path))
        _ct_data = _ct.get_fdata().flatten()
        _sample = np.random.choice(_ct_data, size=min(100000, len(_ct_data)), replace=False)
        ax_intensity.hist(_sample, bins=100, alpha=0.3, label=_case_id, density=True)

    ax_intensity.set_xlabel(INTENSITY_LABEL)
    ax_intensity.set_ylabel("Density")
    ax_intensity.set_title(f"{INTENSITY_LABEL} Distribution (5 patients)")
    ax_intensity.legend()
    plt.tight_layout()
    fig_intensity
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Anomaly Check
    """)
    return


@app.cell
def _(EXPECTED_CASES, INTENSITY_LABEL, LABEL_NAMES, df, stats):
    _foreground_keys = [
        (label, name.lower().replace(" ", "_").replace("&", "and"), name)
        for label, name in LABEL_NAMES.items()
        if label != 0
    ]

    print("Checking for anomalies...\n")

    for _, _row in df.iterrows():
        if _row["unexpected_labels"]:
            print(f"WARNING: {_row.name} has unexpected labels {_row['unexpected_labels']}")

        for _, _key, _label_name in _foreground_keys:
            if _row[f"voxels_{_key}"] == 0:
                print(f"WARNING: {_row.name} has 0 voxels for {_label_name}!")

        if _row["spacing_z"] > 5.0:
            print(f"NOTE: {_row.name} has large z-spacing: {_row['spacing_z']:.2f} mm")

    print("\nDimension summary:")
    print(f"  Z slices: {int(df['shape_z'].min())} to {int(df['shape_z'].max())}")
    print(f"  In-plane: {int(df['shape_x'].min())}x{int(df['shape_y'].min())} to {int(df['shape_x'].max())}x{int(df['shape_y'].max())}")
    print(f"  {INTENSITY_LABEL} range: [{df['intensity_min'].min():.0f}, {df['intensity_max'].max():.0f}]")

    if EXPECTED_CASES is not None and len(stats) != EXPECTED_CASES:
        print(f"\nWARNING: Found {len(stats)} patients, expected {EXPECTED_CASES} for this layout.")
    else:
        print(f"\nDone! Found {len(stats)} patients for the active layout.")
    return


if __name__ == "__main__":
    app.run()
