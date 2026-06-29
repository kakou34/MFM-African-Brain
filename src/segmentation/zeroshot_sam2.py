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

    Returns prob_mask (D, H_orig, W_orig) float32 numpy with sigmoid probabilities.
    """
    prob_mask = np.zeros(gt_mask.shape, dtype=np.float32)

    # Pick the slice with the most GT tumour pixels as the prompt frame
    slice_areas = gt_mask.sum(axis=(1, 2))
    if slice_areas.max() == 0:
        return prob_mask  # no tumour in GT → return empty

    prompt_frame = int(slice_areas.argmax())
    box = bbox_from_mask2d(gt_mask[prompt_frame])
    if box is None:
        return prob_mask

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
            prob_mask[frame_idx] = torch.sigmoid(masks[0, 0]).cpu().numpy()

        # Backward: prompt_frame → first slice
        for frame_idx, _, masks in predictor.propagate_in_video(
            state, start_frame_idx=prompt_frame, reverse=True
        ):
            prob_mask[frame_idx] = torch.sigmoid(masks[0, 0]).cpu().numpy()

        predictor.reset_state(state)

    return prob_mask


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

    # Load model
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), job_name="sam2"):
        predictor = build_sam2_video_predictor_npz(
            args.config_path, args.checkpoint_path
        )
    predictor.cuda().eval()

    records = []
    for subject_id, group in df.groupby("subject_id"):
        label = int(group["label"].iloc[0])

        mask_path = root / group["mask"].iloc[0]
        if not mask_path.exists():
            print(f"[skip] mask not found: {mask_path}")
            continue
        gt_mask = load_and_binarize(mask_path)  # (D, H, W)
        _, H_orig, W_orig = gt_mask.shape

        prob_masks = {}  # seq -> (D, H, W) float32 probability map

        for _, row in group.iterrows():
            seq = row["seq"]
            img_path = root / row["img"]
            print(f"{subject_id} / {seq}", end="  ", flush=True)

            volume = nib.load(img_path).get_fdata().transpose(2, 0, 1)  # (D,H,W)
            images = preprocess_volume(volume, target_size=args.target_size)

            prob = run_zero_shot(predictor, images, gt_mask, H_orig, W_orig)
            prob_masks[seq] = prob

            pred = (prob > 0.5).astype(np.uint8)
            m = compute_metrics(pred, gt_mask)
            print(f"dice={m['dice']:.3f}  iou={m['iou']:.3f}")
            records.append({"subject_id": subject_id, "seq": seq, "label": label, **m})

        if not prob_masks:
            continue

        # Combined mask: average probability across all sequences
        avg_prob = np.mean(list(prob_masks.values()), axis=0)
        combined_pred = (avg_prob > 0.5).astype(np.uint8)
        m_comb = compute_metrics(combined_pred, gt_mask)
        print(f"{subject_id} / combined  dice={m_comb['dice']:.3f}  iou={m_comb['iou']:.3f}")
        records.append({"subject_id": subject_id, "seq": "combined", "label": label, **m_comb})

        # Save FLAIR prediction
        if args.flair_seq in prob_masks:
            flair_rows = group[group["seq"] == args.flair_seq]
            flair_img_path = root / flair_rows.iloc[0]["img"]
            flair_dir = out_dir / args.model_name / args.flair_seq
            flair_dir.mkdir(parents=True, exist_ok=True)
            flair_pred = (prob_masks[args.flair_seq] > 0.5).astype(np.uint8)
            save_pred_nifti(flair_pred, flair_img_path,
                            flair_dir / f"{subject_id}_pred.nii.gz")

        # Save combined prediction
        combined_dir = out_dir / args.model_name / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        ref_img_path = root / group["img"].iloc[0]
        save_pred_nifti(combined_pred, ref_img_path,
                        combined_dir / f"{subject_id}_pred.nii.gz")

    df_scans = pd.DataFrame(records)
    scan_csv = out_dir / "per_scan_metrics.csv"
    df_scans.to_csv(scan_csv, index=False)
    print(f"\nSaved per-scan metrics → {scan_csv}")


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
    parser.add_argument("--model_name",      type=str, required=True,
                        help="Model name used as a subdirectory in out_dir (e.g. sam2_tiny)")
    parser.add_argument("--flair_seq",       type=str, default="t2f",
                        help="Sequence name treated as FLAIR for saving predictions (default: t2f)")
    args = parser.parse_args()
    main(args)
