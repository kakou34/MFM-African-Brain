import os
import random
import numpy as np
from pathlib import Path
import argparse
import nibabel as nib
import pandas as pd
import torch
import torch.nn.functional as F
import medim
from monai.transforms import (
    Compose, LoadImaged, Resized, ResizeWithPadOrCropd, NormalizeIntensityd
)
from sklearn.model_selection import StratifiedShuffleSplit

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_SIZE = 128  # SAM-Med3D native input size


# -----------------------------------------------------------------------------
# MONAI transforms
# -----------------------------------------------------------------------------
def image_transform():
    return Compose([
        LoadImaged(keys=["img"], ensure_channel_first=True),
        Resized(keys=["img"], spatial_size=MODEL_SIZE, mode="trilinear",
                align_corners=False, size_mode="longest"),
        ResizeWithPadOrCropd(keys=["img"], spatial_size=(MODEL_SIZE,) * 3,
                             mode="constant"),
        NormalizeIntensityd(keys=["img"], nonzero=True),
    ])


def mask_transform():
    return Compose([
        LoadImaged(keys=["mask"], ensure_channel_first=True),
        Resized(keys=["mask"], spatial_size=MODEL_SIZE, mode="nearest",
                size_mode="longest"),
        ResizeWithPadOrCropd(keys=["mask"], spatial_size=(MODEL_SIZE,) * 3,
                             mode="constant"),
    ])


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def seg_path_from_img(root, img_relative, subject_id):
    return (root / img_relative).parent / f"{subject_id}-seg.nii.gz"


def load_and_binarize(seg_path):
    """Returns (D, H, W) uint8 in nibabel-transposed convention."""
    arr = nib.load(seg_path).get_fdata().transpose(2, 0, 1)
    return (arr > 0).astype(np.uint8)


def centroid_3d(mask):
    """mask: (D,H,W) uint8. Returns [d,h,w] float or None."""
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    return coords.mean(axis=0).tolist()


def unpad_and_upsample(pred_model, original_size):
    """
    Reverse MONAI's ResizeWithPadOrCropd and upsample to original_size.
    pred_model : (MODEL_SIZE, MODEL_SIZE, MODEL_SIZE) float numpy
    original_size : (D, H, W)
    Returns (D, H, W) uint8 numpy.
    """
    D, H, W = original_size
    scale  = MODEL_SIZE / max(D, H, W)
    D_r    = round(D * scale)
    H_r    = round(H * scale)
    W_r    = round(W * scale)
    pad_d  = (MODEL_SIZE - D_r) // 2
    pad_h  = (MODEL_SIZE - H_r) // 2
    pad_w  = (MODEL_SIZE - W_r) // 2

    cropped = pred_model[pad_d:pad_d + D_r, pad_h:pad_h + H_r, pad_w:pad_w + W_r]
    t = torch.from_numpy(cropped.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(D, H, W), mode="trilinear", align_corners=False)
    return (t[0, 0].numpy() > 0.5).astype(np.uint8)


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
@torch.no_grad()
def run_zero_shot(model, image_tensor, gt_mask_model, n_points=10):
    """
    Multi-click zero-shot inference using n_points random foreground samples.

    model          : SAM_Med3D on DEVICE, eval mode
    image_tensor   : (1, 1, 128, 128, 128) float32 on DEVICE
    gt_mask_model  : (128, 128, 128) uint8 numpy  -- used to sample positive clicks
    n_points       : number of foreground points to sample

    Returns        : (128, 128, 128) uint8 numpy binary mask in model space.
    """
    # --- 1. image embedding ---
    image_embeddings = model.image_encoder(image_tensor)  # (1, 384, 8, 8, 8)

    # --- 2. sample n_points positive clicks from the GT ---
    gt_t      = torch.from_numpy(gt_mask_model).to(DEVICE)   # (128,128,128)
    fg_coords = torch.argwhere(gt_t > 0)                     # (N, 3)
    if len(fg_coords) == 0:
        return np.zeros((MODEL_SIZE,) * 3, dtype=np.uint8)

    N = fg_coords.shape[0]
    if N >= n_points:
        idx = torch.randperm(N)[:n_points]
    else:
        idx = torch.randint(N, (n_points,))
    point = fg_coords[idx].unsqueeze(0).float()              # (1, n_points, 3)
    label = torch.ones(1, n_points, dtype=torch.int64, device=DEVICE)

    # --- 3. empty low-res mask prompt (1, 1, 32, 32, 32) ---
    prev_low_res = torch.zeros(
        1, 1, MODEL_SIZE // 4, MODEL_SIZE // 4, MODEL_SIZE // 4, device=DEVICE
    )

    # --- 4. prompt encoder ---
    sparse_emb, dense_emb = model.prompt_encoder(
        points=[point, label],
        boxes=None,
        masks=prev_low_res,
    )

    # --- 5. mask decoder ---
    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
    )

    # --- 6. upsample to model resolution, sigmoid, threshold ---
    pred = F.interpolate(
        low_res_masks, size=image_tensor.shape[-3:],
        mode="trilinear", align_corners=False,
    )
    pred = torch.sigmoid(pred)[0, 0].cpu().numpy()               # (128,128,128)
    return (pred > 0.5).astype(np.uint8)

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def save_pred_nifti(pred_dhw: np.ndarray, orig_img_path: Path, save_path: Path):
    """Save binary prediction (D, H, W) as NIfTI (H, W, D) using original affine."""
    orig = nib.load(orig_img_path)
    hwz  = pred_dhw.transpose(1, 2, 0).astype(np.uint8)
    nib.save(nib.Nifti1Image(hwz, orig.affine, orig.header), str(save_path))


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
    df      = pd.read_csv(args.csv_path)
    root    = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    masks_dir = Path(args.masks_dir) if args.masks_dir else None
    if masks_dir:
        masks_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SAM-Med3D...")
    ckpt = args.checkpoint_path or \
        "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
    model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt)
    model = model.to(DEVICE).eval()

    img_tfm  = image_transform()
    mask_tfm = mask_transform()

    # test_splits = get_test_splits(df)

    records = []
    for subject_id, group in df.groupby("subject_id"):
        label = int(group["label"].iloc[0])

        seg = seg_path_from_img(root, group["img"].iloc[0], subject_id)
        if not seg.exists():
            print(f"[skip] seg not found: {seg}")
            continue

        # GT in model space (for centroid prompt)
        mask_data    = mask_tfm({"mask": str(seg)})
        gt_mask_128  = (mask_data["mask"][0].numpy() > 0).astype(np.uint8)  # (128,128,128)

        # GT in original space (for comparable metrics)
        gt_mask_orig = load_and_binarize(seg)      # (D, H, W)
        original_size = gt_mask_orig.shape         # (D, H, W)

        for _, row in group.iterrows():
            seq      = row["seq"]
            img_path = root / row["img"]
            print(f"{subject_id} / {seq}", end="  ", flush=True)

            img_data     = img_tfm({"img": str(img_path)})
            image_tensor = img_data["img"].unsqueeze(0).to(DEVICE)  # (1,1,128,128,128)

            prediction_128  = run_zero_shot(model, image_tensor, gt_mask_128, args.num_points)
            prediction_orig = unpad_and_upsample(prediction_128, original_size)

            m = compute_metrics(prediction_orig, gt_mask_orig)

            if masks_dir is not None:
                save_pred_nifti(
                    prediction_orig, root / row["img"],
                    masks_dir / f"{subject_id}_{seq}_pred.nii.gz",
                )

            print(f"dice={m['dice']:.3f}  iou={m['iou']:.3f}")

            records.append({
                "subject_id": subject_id,
                "seq":        seq,
                "label":      label,
                **m,
            })

    df_scans  = pd.DataFrame(records)
    scan_csv  = out_dir / "per_scan_metrics.csv"
    df_scans.to_csv(scan_csv, index=False)
    print(f"\nSaved per-scan metrics → {scan_csv}")

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
    parser = argparse.ArgumentParser(description="Zero-shot SAM-Med3D tumour segmentation")
    parser.add_argument("--csv_path",        type=str, required=True)
    parser.add_argument("--root_dir",        type=str, required=True)
    parser.add_argument("--out_dir",         type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Local .pth path; omit to download from HuggingFace")
    parser.add_argument("--num_points",      type=int, default=10,
                        help="Foreground points sampled randomly per volume for the prompt (default: 10)")
    parser.add_argument("--masks_dir",       type=str, default=None,
                        help="If set, save per-subject NIfTI predictions here as {subject}_{seq}_pred.nii.gz")
    args = parser.parse_args()
    main(args)
