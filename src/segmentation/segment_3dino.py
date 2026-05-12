import sys
from pathlib import Path

DINO3_ROOT = Path(__file__).resolve().parents[2] / "3DINO"
sys.path.append(str(DINO3_ROOT))

import os
import copy
import random
import argparse
from functools import partial

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import monai
import monai.transforms.transform as _mtt
import monai.transforms.compose as _mtc
# MONAI 1.3.2 sets MAX_SEED = 2^32 but numpy/Python 3.13 overflows uint32 at
# that boundary. Patch both modules before any Compose is instantiated.
_mtt.MAX_SEED = int(np.iinfo(np.uint32).max)   # 4294967295
_mtc.MAX_SEED = int(np.iinfo(np.uint32).max)
from monai.data import DataLoader, Dataset
from monai.data.utils import list_data_collate
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureTyped,
    LambdaD,
    LoadImaged,
    NormalizeIntensityd,
    RandFlipd,
    RandRotate90d,
    ResizeWithPadOrCropd,
    Resized,
)
from scipy.ndimage import zoom as nd_zoom
from sklearn.model_selection import StratifiedShuffleSplit

from dinov2.configs import load_and_merge_config_3d
from dinov2.eval.setup import build_model_for_eval
from dinov2.eval.segmentation_3d.segmentation_heads import LinearDecoderHead, UNETRHead

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
monai.utils.set_determinism(seed=SEED)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 96   # must be divisible by patch_size=16; matches feature-extraction scripts
NUM_CLASSES = 2   # background + foreground tumour
INPUT_CHANNELS = 1


# ── Transforms ─────────────────────────────────────────────────────────────────
def _binarize(x):
    return (x > 0).float()


def make_train_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        # Resize longest axis to IMAGE_SIZE, preserving aspect ratio
        Resized(keys=["image"], spatial_size=IMAGE_SIZE, mode="trilinear",
                align_corners=False, size_mode="longest"),
        Resized(keys=["label"], spatial_size=IMAGE_SIZE, mode="nearest",
                size_mode="longest"),
        ResizeWithPadOrCropd(keys=["image", "label"],
                             spatial_size=(IMAGE_SIZE,) * 3, mode="constant"),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        LambdaD(keys=["label"], func=_binarize),
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])


def make_val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Resized(keys=["image"], spatial_size=IMAGE_SIZE, mode="trilinear",
                align_corners=False, size_mode="longest"),
        Resized(keys=["label"], spatial_size=IMAGE_SIZE, mode="nearest",
                size_mode="longest"),
        ResizeWithPadOrCropd(keys=["image", "label"],
                             spatial_size=(IMAGE_SIZE,) * 3, mode="constant"),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        LambdaD(keys=["label"], func=_binarize),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])


# ── Data helpers ───────────────────────────────────────────────────────────────
def seg_path_from_img(root: Path, img_relative: str, subject_id: str) -> Path:
    return (root / img_relative).parent / f"{subject_id}-seg.nii.gz"


def build_datalist(df_subjects: pd.DataFrame, root: Path) -> list:
    records = []
    for _, row in df_subjects.iterrows():
        seg = seg_path_from_img(root, row["img"], row["subject_id"])
        if not seg.exists():
            print(f"  [skip] seg not found: {seg}")
            continue
        img_path = str(root / row["img"])
        records.append({
            "image":      img_path,
            "label":      str(seg),
            "subject_id": row["subject_id"],
            "image_path": img_path,
        })
    return records


# ── Splits identical to the other tasks ────────────────────────────────────────
def get_splits(df: pd.DataFrame):
    subjects = (
        df.drop_duplicates("subject_id")[["subject_id", "label"]]
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    X = subjects["subject_id"].values
    y = subjects["label"].values
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=42)
    return [
        (set(X[train_idx]), set(X[test_idx]))
        for train_idx, test_idx in sss.split(X, y)
    ]


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(pred: np.ndarray, gt: np.ndarray):
    p = pred.flatten().astype(bool)
    g = gt.flatten().astype(bool)
    inter = (p & g).sum()
    dice = 2 * inter / (p.sum() + g.sum() + 1e-8)
    iou = inter / ((p | g).sum() + 1e-8)
    return float(dice), float(iou)


def save_pred_nifti(pred_np: np.ndarray, orig_img_path: str, save_path: Path):
    """Resample binary prediction from model space to original NIfTI space and save."""
    orig = nib.load(orig_img_path)
    orig_shape = orig.shape[:3]
    factors = tuple(o / p for o, p in zip(orig_shape, pred_np.shape))
    resampled = nd_zoom(pred_np.astype(np.float32), factors, order=0)
    nib.save(
        nib.Nifti1Image(resampled.astype(np.uint8), orig.affine, orig.header),
        str(save_path),
    )


# ── Training / evaluation ──────────────────────────────────────────────────────
def train_epoch(seg_model, loader, optimizer, loss_fn, scaler, autocast_ctx):
    seg_model.train()
    # Keep the encoder in eval mode if its parameters are frozen
    if not any(p.requires_grad for p in seg_model.feature_model.parameters()):
        seg_model.feature_model.eval()

    total_loss = 0.0
    for batch in loader:
        x = batch["image"].to(DEVICE)   # (B, 1, H, W, D)
        y = batch["label"].to(DEVICE)   # (B, 1, H, W, D) float 0/1

        optimizer.zero_grad()
        with autocast_ctx():
            logits = seg_model(x)       # (B, 2, H, W, D)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(seg_model, loader, autocast_ctx, masks_dir=None):
    seg_model.eval()
    all_dice, all_iou, per_sample = [], [], []
    for batch in loader:
        x     = batch["image"].to(DEVICE)
        gt_np = batch["label"].cpu().numpy()[:, 0]  # (B, H, W, D)

        with autocast_ctx():
            logits = sliding_window_inference(
                x, (IMAGE_SIZE,) * 3, 1, seg_model, overlap=0.5
            )

        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # (B, H, W, D)
        preds  = (probs > 0.5).astype(np.uint8)
        sids   = batch.get("subject_id",  [None] * preds.shape[0])
        ipaths = batch.get("image_path",  [None] * preds.shape[0])

        for i in range(preds.shape[0]):
            gt_bin = (gt_np[i] > 0).astype(np.uint8)
            d, iou = compute_metrics(preds[i], gt_bin)
            all_dice.append(d)
            all_iou.append(iou)
            sid   = sids[i]   if isinstance(sids,   (list, tuple)) else sids
            ipath = ipaths[i] if isinstance(ipaths, (list, tuple)) else ipaths
            per_sample.append({"subject_id": sid, "dice": d, "iou": iou})
            if masks_dir is not None and sid is not None and ipath is not None:
                save_pred_nifti(preds[i], ipath, masks_dir / f"{sid}_pred.nii.gz")

    return np.mean(all_dice), np.mean(all_iou), all_dice, all_iou, per_sample


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args):
    df = pd.read_csv(args.csv_path)

    # Ensure subject_id column exists (BraTS id format: "<subject>_<seq>")
    if "subject_id" not in df.columns:
        if df["id"].str.contains("_").any():
            df["subject_id"] = df["id"].str.rsplit("_", n=1).str[0]
        else:
            df["subject_id"] = df["id"]

    # Filter to the requested sequence
    df_seq = df[df["seq"] == args.seq].copy()
    if df_seq.empty:
        raise ValueError(
            f"No rows with seq='{args.seq}'. "
            f"Available: {df['seq'].unique().tolist()}"
        )
    print(f"Sequence '{args.seq}': {df_seq['subject_id'].nunique()} subjects")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.root_dir)

    # Load the frozen 3DINO encoder once (shared across splits when frozen)
    print("Loading 3DINO encoder...")
    cfg = load_and_merge_config_3d("train/vit3d_highres")
    feature_model = build_model_for_eval(cfg, args.checkpoint_path)
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float16)
    print(f"  encoder: {cfg.student.arch}, embed_dim={feature_model.num_features}")

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True).to(DEVICE)

    splits = get_splits(df_seq)
    split_records = []

    for split_idx, (train_subjects, test_subjects) in enumerate(splits):
        # Skip splits that are not requested
        if args.split_idx is not None and split_idx + 1 != args.split_idx:
            continue

        print(f"\n{'='*60}")
        print(f"Split {split_idx + 1}/5  |  "
              f"train={len(train_subjects)}  test={len(test_subjects)}")

        df_train = df_seq[df_seq["subject_id"].isin(train_subjects)]
        df_test = df_seq[df_seq["subject_id"].isin(test_subjects)]

        train_list = build_datalist(df_train, root)
        test_list = build_datalist(df_test, root)
        print(f"  volumes with seg: train={len(train_list)}, test={len(test_list)}")

        train_ds = Dataset(data=train_list, transform=make_train_transforms())
        test_ds = Dataset(data=test_list, transform=make_val_transforms())

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=list_data_collate,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=1, shuffle=False,
            num_workers=args.num_workers, collate_fn=list_data_collate,
        )

        # Fresh segmentation head for each split; encoder may be shared or deep-copied
        if args.train_encoder:
            enc = copy.deepcopy(feature_model)
        else:
            enc = feature_model

        if args.segmentation_head == "UNETR":
            seg_model = UNETRHead(enc, INPUT_CHANNELS, IMAGE_SIZE,
                                  NUM_CLASSES, autocast_ctx).to(DEVICE)
        else:
            seg_model = LinearDecoderHead(enc, INPUT_CHANNELS, IMAGE_SIZE,
                                          NUM_CLASSES, autocast_ctx).to(DEVICE)

        if not args.train_encoder:
            seg_model.feature_model.eval()
            for param in seg_model.feature_model.parameters():
                param.requires_grad = False

        trainable = sum(p.numel() for p in seg_model.parameters() if p.requires_grad)
        print(f"  trainable parameters: {trainable:,}")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, seg_model.parameters()),
            lr=args.lr, weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(args.epochs):
            train_loss = train_epoch(
                seg_model, train_loader, optimizer, loss_fn, scaler, autocast_ctx
            )
            scheduler.step()
            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                print(f"  epoch {epoch+1:3d}/{args.epochs} | loss={train_loss:.4f}")

        # Set up mask output directory for this split if requested
        masks_dir = None
        if args.split_idx is not None:
            masks_dir = out_dir / f"masks_split{split_idx + 1}"
            masks_dir.mkdir(parents=True, exist_ok=True)

        dice, iou, per_sub_dice, per_sub_iou, per_sample = evaluate(
            seg_model, test_loader, autocast_ctx, masks_dir=masks_dir
        )
        print(f"  Split {split_idx+1} → dice={dice:.3f} ± {np.std(per_sub_dice):.3f} | "
              f"iou={iou:.3f} ± {np.std(per_sub_iou):.3f}")

        split_records.append({
            "split": split_idx + 1,
            "seq": args.seq,
            "n_train": len(train_list),
            "n_test": len(test_list),
            "dice_mean": dice,
            "dice_std": float(np.std(per_sub_dice)),
            "iou_mean": iou,
            "iou_std": float(np.std(per_sub_iou)),
        })

        # Save per-sample metrics and split membership when running a single split
        if args.split_idx is not None:
            per_sample_csv = out_dir / f"per_sample_metrics_split{split_idx + 1}.csv"
            pd.DataFrame(per_sample).to_csv(per_sample_csv, index=False)
            print(f"  Saved per-sample metrics → {per_sample_csv}")

            split_info = (
                [{"subject_id": s, "role": "train"} for s in sorted(train_subjects)] +
                [{"subject_id": s, "role": "test"}  for s in sorted(test_subjects)]
            )
            split_info_csv = out_dir / f"split_info_split{split_idx + 1}.csv"
            pd.DataFrame(split_info).to_csv(split_info_csv, index=False)
            print(f"  Saved split info → {split_info_csv}")

            if masks_dir is not None:
                print(f"  Saved {len(per_sample)} masks → {masks_dir}/")

        # Release GPU memory before next split
        del seg_model, optimizer, scheduler, scaler
        if args.train_encoder:
            del enc
        torch.cuda.empty_cache()

    df_splits = pd.DataFrame(split_records)
    split_csv = out_dir / "split_metrics.csv"
    df_splits.to_csv(split_csv, index=False)
    print(f"\nSaved split metrics → {split_csv}")

    summary = df_splits[["dice_mean", "iou_mean"]].agg(["mean", "std"]).round(3)
    print("\nSummary across evaluated splits:")
    print(summary.to_string())


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3DINO segmentation on a single MRI sequence with 5 stratified splits"
    )
    parser.add_argument("--csv_path", type=str, required=True,
                        help="CSV with columns: id, img, seq, label, [subject_id]")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory for image paths in the CSV")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for metrics CSV")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to 3DINO pretrained weights (.pth)")
    parser.add_argument("--seq", type=str, default="t2f",
                        help="Sequence name to filter on (default: t2f)")
    parser.add_argument("--segmentation_head", type=str, default="UNETR",
                        choices=["UNETR", "Linear"],
                        help="Segmentation head to attach to the encoder")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs per split")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_encoder", action="store_true",
                        help="Fine-tune the encoder; default is frozen (linear probing)")
    parser.add_argument("--split_idx", type=int, default=None,
                        help="Run only this split (1-indexed, 1-5); saves masks, "
                             "per-sample metrics and split info. Default runs all 5 splits.")
    args = parser.parse_args()
    main(args)
