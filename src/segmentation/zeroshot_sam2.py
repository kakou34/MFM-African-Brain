import os
import random
import numpy as np
from pathlib import Path
import argparse
import nibabel as nib
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir
from sam2.build_sam import build_sam2_video_predictor_npz

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CONFIG_DIR = Path("MedSAM2/config")


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------
def preprocess_volume(volume, target_size=512):
    """
    volume: np.ndarray (D, H, W)
    Returns normalised tensor (D, 3, target_size, target_size) on CUDA.
    """
    v = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8) * 255.0
    v = v.astype(np.uint8)
    D, H, W = v.shape
    out = np.zeros((D, 3, target_size, target_size), dtype=np.float32)
    for i in range(D):
        img = Image.fromarray(v[i]).convert("RGB").resize((target_size, target_size))
        out[i] = np.array(img).transpose(2, 0, 1)
    out /= 255.0
    t = torch.from_numpy(out).float()
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    t -= mean
    t /= std
    return t  # keep on CPU; init_state handles device placement


def load_and_binarize(seg_path):
    """
    Load a seg NIfTI and binarize all tumour subregions to 1.
    nibabel BraTS layout is (H, W, D); we return (D, H, W) uint8.
    """
    arr = nib.load(seg_path).get_fdata().transpose(2, 0, 1)  # (D, H, W)
    return (arr > 0).astype(np.uint8)


def seg_path_from_img(root, img_relative, subject_id):
    img_dir = (root / img_relative).parent
    return img_dir / f"{subject_id}-seg.nii.gz"


# -----------------------------------------------------------------------------
# Prompt generation
# -----------------------------------------------------------------------------
def bbox_from_mask2d(mask_slice):
    """
    2-D binary mask → [x1, y1, x2, y2] in (col, row) order.
    Returns None when the slice is empty.
    """
    rows = np.where(mask_slice.any(axis=1))[0]
    cols = np.where(mask_slice.any(axis=0))[0]
    if len(rows) == 0:
        return None
    return [float(cols[0]), float(rows[0]), float(cols[-1]), float(rows[-1])]


# -----------------------------------------------------------------------------
# Zero-shot inference
# -----------------------------------------------------------------------------
def run_zero_shot(predictor, images, gt_mask, H_orig, W_orig):
    """
    predictor : SAM2VideoPredictorNPZ (already on GPU, eval mode)
    images    : (D, 3, T, T) float32 CPU tensor (normalised)
    gt_mask   : (D, H_orig, W_orig) uint8 numpy - only used to derive the bbox prompt
    H_orig, W_orig : original slice dimensions (for coordinate normalisation)

    Returns pred_mask (D, H_orig, W_orig) uint8 numpy.
    """
    pred_mask = np.zeros(gt_mask.shape, dtype=np.uint8)

    # Pick the slice with the most GT tumour pixels as the prompt frame
    slice_areas = gt_mask.sum(axis=(1, 2))
    if slice_areas.max() == 0:
        return pred_mask  # no tumour in GT → return empty

    prompt_frame = int(slice_areas.argmax())
    box = bbox_from_mask2d(gt_mask[prompt_frame])
    if box is None:
        return pred_mask

    with torch.inference_mode():
        state = predictor.init_state(
            images=images,
            video_height=H_orig,
            video_width=W_orig,
            offload_video_to_cpu=True,
        )

        predictor.add_new_points_or_box(
            state,
            frame_idx=prompt_frame,
            obj_id=1,
            box=box,
        )

        # Forward: prompt_frame → last slice
        for frame_idx, _, masks in predictor.propagate_in_video(
            state, start_frame_idx=prompt_frame
        ):
            pred_mask[frame_idx] = (masks[0, 0].cpu().numpy() > 0).astype(np.uint8)

        # Backward: prompt_frame → first slice
        for frame_idx, _, masks in predictor.propagate_in_video(
            state, start_frame_idx=prompt_frame, reverse=True
        ):
            pred_mask[frame_idx] = (masks[0, 0].cpu().numpy() > 0).astype(np.uint8)

        predictor.reset_state(state)

    return pred_mask


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def save_pred_nifti(pred_dhw: np.ndarray, orig_img_path: Path, save_path: Path):
    """Save binary prediction (D, H, W) as NIfTI (H, W, D) using original affine."""
    orig = nib.load(orig_img_path)
    hwz  = pred_dhw.transpose(1, 2, 0).astype(np.uint8)
    nib.save(nib.Nifti1Image(hwz, orig.affine, orig.header), str(save_path))


def compute_metrics(pred, gt):
    p = pred.flatten().astype(bool)
    g = gt.flatten().astype(bool)
    inter = (p & g).sum()
    dice  = 2 * inter / (p.sum() + g.sum() + 1e-8)
    iou   = inter / ((p | g).sum() + 1e-8)
    return {"dice": float(dice), "iou": float(iou)}


# -----------------------------------------------------------------------------
# Splits  (identical to classify_disease.py)
# -----------------------------------------------------------------------------
def get_test_splits(df):
    """
    Reproduce the StratifiedShuffleSplit used in classify_disease.py.
    Returns a list of 5 sets of subject_ids that belong to each test fold.
    """
    subjects = (
        df.drop_duplicates("subject_id")[["subject_id", "label"]]
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    X = subjects["subject_id"].values
    y = subjects["label"].values
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=42)
    return [set(X[test_idx]) for _, test_idx in sss.split(X, y)]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(args):
    df = pd.read_csv(args.csv_path)
    root = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    masks_dir = Path(args.masks_dir) if args.masks_dir else None
    if masks_dir:
        masks_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), job_name="sam2"):
        predictor = build_sam2_video_predictor_npz(
            args.config_path, args.checkpoint_path
        )
    predictor.cuda().eval()

    # Reproduce splits
    # test_splits = get_test_splits(df)

    # Run inference per subject × sequence
    records = []
    for subject_id, group in df.groupby("subject_id"):
        label = int(group["label"].iloc[0])

        seg = seg_path_from_img(root, group["img"].iloc[0], subject_id)
        if not seg.exists():
            print(f"[skip] seg not found: {seg}")
            continue
        gt_mask = load_and_binarize(seg)   # (D, H, W)
        _, H_orig, W_orig = gt_mask.shape

        for _, row in group.iterrows():
            seq = row["seq"]
            img_path = root / row["img"]
            print(f"{subject_id} / {seq}", end="  ", flush=True)

            volume = nib.load(img_path).get_fdata().transpose(2, 0, 1)  # (D,H,W)
            images = preprocess_volume(volume, target_size=args.target_size)

            pred = run_zero_shot(predictor, images, gt_mask, H_orig, W_orig)
            m = compute_metrics(pred, gt_mask)

            if masks_dir is not None:
                save_pred_nifti(
                    pred, root / row["img"],
                    masks_dir / f"{subject_id}_{seq}_pred.nii.gz",
                )

            print(f"dice={m['dice']:.3f}  iou={m['iou']:.3f}")
            records.append({
                "subject_id": subject_id,
                "seq": seq,
                "label": label,
                **m,
            })

    df_scans = pd.DataFrame(records)
    scan_csv = out_dir / "per_scan_metrics.csv"
    df_scans.to_csv(scan_csv, index=False)
    print(f"\nSaved per-scan metrics → {scan_csv}")

    # # Aggregate per split × sequence
    # split_records = []
    # for split_idx, test_subjects in enumerate(test_splits):
    #     df_test = df_scans[df_scans["subject_id"].isin(test_subjects)]
    #     for seq, grp in df_test.groupby("seq"):
    #         split_records.append({
    #             "split":      split_idx + 1,
    #             "seq":        seq,
    #             "n_subjects": len(grp),
    #             "dice_mean":  grp["dice"].mean(),
    #             "dice_std":   grp["dice"].std(),
    #             "iou_mean":   grp["iou"].mean(),
    #             "iou_std":    grp["iou"].std(),
    #         })

    # df_splits = pd.DataFrame(split_records)
    # split_csv = out_dir / "split_metrics.csv"
    # df_splits.to_csv(split_csv, index=False)
    # print(f"Saved split metrics  → {split_csv}")

    # summary = (
    #     df_splits.groupby("seq")[["dice_mean", "iou_mean"]]
    #     .agg(["mean", "std"])
    #     .round(3)
    # )
    # print("\nSummary across splits:")
    # print(summary)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot SAM2 tumour segmentation")
    parser.add_argument("--csv_path",        type=str, required=True)
    parser.add_argument("--root_dir",        type=str, required=True)
    parser.add_argument("--out_dir",         type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--config_path",     type=str,
                        default="sam2.1_hiera_t512.yaml")
    parser.add_argument("--target_size",     type=int, default=512)
    parser.add_argument("--masks_dir",       type=str, default=None,
                        help="If set, save per-subject NIfTI predictions here as {subject}_{seq}_pred.nii.gz")
    args = parser.parse_args()
    main(args)
