"""
Visualize the middle-tumour axial slice for selected patients.
One column per patient; rows: scan | GT overlay | per-model prediction overlay.
Supervised models use masks_split4/{sid}_pred.nii.gz.
Zero-shot models use masks_viz/{sid}_t2f_pred.nii.gz.

Usage:
    python plot_tumour_slices.py [--out figures/tumour_slices.pdf]
"""
import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

DATA_ROOT  = Path("/projects/prjs0830/DATA")
SEG_ROOT   = Path("csvs/segmentation")
INPUT_CSV  = Path("csvs/input/bratsafrica.csv")

SUBJECTS = [
    "BraTS-SSA-00025-000",
    "BraTS-SSA-00137-000",
    "BraTS-SSA-00151-000",
]

# Each entry: (display_name, masks_dir, seq_suffix_or_None)
# seq_suffix: if not None, mask filename is {sid}_{seq}_pred.nii.gz (zero-shot models)
#             if None, filename is {sid}_pred.nii.gz (supervised models)
MODELS = [
    ("SAM-Med3D",       SEG_ROOT / "sammed_10pts/masks_viz",   "t2f"),
    ("SAM-Med3D\n+LoRA", SEG_ROOT / "lora_sammed/masks_split4", None),
    ("BrainIAC",        SEG_ROOT / "brainiac/masks_split4",     None),
    ("3D-DINO",         SEG_ROOT / "3dino/masks_split4",        None),
    ("MedSAM2",         SEG_ROOT / "medsam2/masks_viz",         "t2f"),
    ("MedicalSAM2",     SEG_ROOT / "medicalsam2/masks_viz",     "t2f"),
]

GT_COLOR   = np.array([0.10, 0.85, 0.30])
PRED_COLOR = np.array([1.0, 0.15, 0.15])
ALPHA      = 0.45


def load_vol(path: Path) -> np.ndarray:
    return nib.load(path).get_fdata().astype(np.float32)


def middle_tumour_slice(gt: np.ndarray) -> int:
    """Index of the axial slice (axis=2) at the centre of the GT tumour extent."""
    occupied = np.where(np.any(gt > 0, axis=(0, 1)))[0]
    return int(occupied[len(occupied) // 2])


def make_overlay(scan_2d: np.ndarray, mask_2d: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Return an RGB image with the mask region tinted by *color*."""
    lo, hi = np.percentile(scan_2d, [1, 99])
    norm = np.clip((scan_2d - lo) / (hi - lo + 1e-8), 0, 1)
    rgb = np.stack([norm] * 3, axis=-1)
    if mask_2d.any():
        m = mask_2d.astype(bool)
        rgb[m] = (1 - ALPHA) * rgb[m] + ALPHA * color
    return rgb


def main(args: argparse.Namespace) -> None:
    # Build subject_id → img relative path lookup from the CSV
    df_input = pd.read_csv(INPUT_CSV)
    t2f_rows = df_input[df_input["seq"] == "t2f"].set_index("subject_id")

    n_rows = 1 + len(MODELS)   # GT | model …
    n_cols = len(SUBJECTS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7, 9))
    axes = np.atleast_2d(axes)

    # Row labels on the leftmost column
    row_labels = ["Ground Truth"] + [m for m, _, _ in MODELS]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=12, labelpad=15,
                                rotation=90, ha="center", va="center")

    for col, sid in enumerate(SUBJECTS):
        img_rel   = t2f_rows.loc[sid, "img"]
        scan_path = DATA_ROOT / img_rel
        gt_path   = scan_path.parent / f"{sid}-seg.nii.gz"
        scan = load_vol(scan_path)
        gt   = (load_vol(gt_path) > 0).astype(np.uint8)

        z       = middle_tumour_slice(gt)
        scan_sl = scan[:, :, z]
        gt_sl   = gt[:, :, z]

        # GT overlay (row 0)
        axes[0, col].imshow(
            make_overlay(scan_sl, gt_sl, GT_COLOR).transpose(1, 0, 2),
            origin="lower",
        )

        # per-model prediction overlays (rows 1+)
        for row_offset, (_, masks_dir, seq_sfx) in enumerate(MODELS):
            fname   = f"{sid}_{seq_sfx}_pred.nii.gz" if seq_sfx else f"{sid}_pred.nii.gz"
            pred    = (load_vol(masks_dir / fname) > 0.5).astype(np.uint8)
            pred_sl = pred[:, :, z]
            axes[1 + row_offset, col].imshow(
                make_overlay(scan_sl, pred_sl, PRED_COLOR).transpose(1, 0, 2),
                origin="lower",
            )

    # Hide ticks and spines without hiding axis labels (axis("off") also hides ylabel)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    legend_handles = [
        mpatches.Patch(facecolor=GT_COLOR,   label="Ground Truth"),
        mpatches.Patch(facecolor=PRED_COLOR, label="Predicted"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=12, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0.08, 0.03, 1, 1])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {out}  ({out.with_suffix('.png')})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="figures/tumour_slices.pdf")
    main(parser.parse_args())
