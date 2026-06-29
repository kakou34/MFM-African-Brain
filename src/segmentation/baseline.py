"""
3D U-Net segmentation baseline trained from scratch.
Matches the same 5-split StratifiedShuffleSplit protocol used across all models.
"""
import os
import random
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom as nd_zoom
import pandas as pd
import torch
import monai
import monai.transforms.transform as _mtt
import monai.transforms.compose as _mtc

_mtt.MAX_SEED = int(np.iinfo(np.uint32).max)
_mtc.MAX_SEED = int(np.iinfo(np.uint32).max)

from monai.data import DataLoader, Dataset
from monai.data.utils import list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    ResizeWithPadOrCropd,
    Resized,
)
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
monai.utils.set_determinism(seed=SEED)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = (128, 128, 128)


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(in_channels: int = 1):
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.1,
    )


# ── Transforms ─────────────────────────────────────────────────────────────────

def make_train_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Resized(keys=["image"], spatial_size=128, mode="trilinear",
                align_corners=False, size_mode="longest"),
        Resized(keys=["label"], spatial_size=128, mode="nearest",
                size_mode="longest"),
        ResizeWithPadOrCropd(keys=["image", "label"],
                             spatial_size=IMAGE_SIZE, mode="constant"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])


def make_val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Resized(keys=["image"], spatial_size=128, mode="trilinear",
                align_corners=False, size_mode="longest"),
        Resized(keys=["label"], spatial_size=128, mode="nearest",
                size_mode="longest"),
        ResizeWithPadOrCropd(keys=["image", "label"],
                             spatial_size=IMAGE_SIZE, mode="constant"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])


# ── Data helpers ───────────────────────────────────────────────────────────────

def filter_valid_subjects(df: pd.DataFrame, seqs: list) -> pd.DataFrame:
    """Keep only subjects that have all requested sequences."""
    counts = (
        df[df["seq"].isin(seqs)]
        .groupby("subject_id")["seq"]
        .apply(lambda x: all(s in x.values for s in seqs))
    )
    valid = counts[counts].index
    return df[df["subject_id"].isin(valid) & df["seq"].isin(seqs)].copy()


def build_datalist(df_subjects: pd.DataFrame, root: Path, seqs: list) -> list:
    records = []
    for subject_id, grp in df_subjects.groupby("subject_id"):
        mask_path = root / grp.iloc[0]["mask"]
        if not mask_path.exists():
            print(f"  [skip] {subject_id}: mask not found: {mask_path}")
            continue

        img_paths, skip = [], False
        for seq in seqs:
            seq_rows = grp[grp["seq"] == seq]
            if seq_rows.empty:
                print(f"  [skip] {subject_id}: missing seq '{seq}'")
                skip = True
                break
            img_path = root / seq_rows.iloc[0]["img"]
            if not img_path.exists():
                print(f"  [skip] {subject_id}: image not found: {img_path}")
                skip = True
                break
            img_paths.append(str(img_path))
        if skip:
            continue

        records.append({
            "subject_id": str(subject_id),
            "image":      img_paths[0] if len(seqs) == 1 else img_paths,
            "label":      str(mask_path),
            "image_path": img_paths[0],
        })
    return records


# ── Splits ─────────────────────────────────────────────────────────────────────

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
    iou  = inter / ((p | g).sum() + 1e-8)
    return float(dice), float(iou)


# ── Training / evaluation / saving ────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0
    for batch in loader:
        x = batch["image"].to(DEVICE)
        y = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_dice, all_iou = [], []
    for batch in loader:
        x  = batch["image"].to(DEVICE)
        gt = batch["label"].cpu().numpy()[:, 0]   # (B, H, W, D)

        logits = sliding_window_inference(x, IMAGE_SIZE, 1, model, overlap=0.5)
        preds  = (torch.sigmoid(logits)[:, 0].cpu().numpy() > 0.5).astype(np.uint8)

        for i in range(preds.shape[0]):
            d, iou = compute_metrics(preds[i], (gt[i] > 0).astype(np.uint8))
            all_dice.append(d)
            all_iou.append(iou)
    return np.mean(all_dice), np.mean(all_iou), all_dice, all_iou


@torch.no_grad()
def save_predictions(model, loader, save_dir: Path):
    """Run inference, compute metrics, and save binary masks as NIfTI files."""
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)
    all_dice, all_iou = [], []
    for batch in loader:
        x  = batch["image"].to(DEVICE)
        gt = batch["label"].cpu().numpy()[:, 0]

        logits = sliding_window_inference(x, IMAGE_SIZE, 1, model, overlap=0.5)
        preds  = (torch.sigmoid(logits)[:, 0].cpu().numpy() > 0.5).astype(np.uint8)

        subject_ids = batch["subject_id"]
        image_paths = batch["image_path"]

        for i in range(preds.shape[0]):
            d, iou = compute_metrics(preds[i], (gt[i] > 0).astype(np.uint8))
            all_dice.append(d)
            all_iou.append(iou)

            orig = nib.load(image_paths[i])
            factors = tuple(o / p for o, p in zip(orig.shape[:3], preds[i].shape))
            resampled = nd_zoom(preds[i].astype(np.float32), factors, order=0)
            nib.save(
                nib.Nifti1Image(resampled.astype(np.uint8), orig.affine, orig.header),
                str(save_dir / f"{subject_ids[i]}_pred.nii.gz"),
            )
    return np.mean(all_dice), np.mean(all_iou), all_dice, all_iou


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    df = pd.read_csv(args.csv_path)

    if "subject_id" not in df.columns:
        if df["id"].str.contains("_").any():
            df["subject_id"] = df["id"].str.rsplit("_", n=1).str[0]
        else:
            df["subject_id"] = df["id"]

    missing_seqs = [s for s in args.seqs if s not in df["seq"].unique()]
    if missing_seqs:
        raise ValueError(
            f"Sequences not found in CSV: {missing_seqs}. "
            f"Available: {df['seq'].unique().tolist()}"
        )

    df_filtered = filter_valid_subjects(df, args.seqs)
    if df_filtered.empty:
        raise ValueError(f"No subjects with all sequences {args.seqs}.")
    n_subjects = df_filtered["subject_id"].nunique()
    print(f"Sequences {args.seqs}: {n_subjects} subjects with all seqs")

    in_channels = len(args.seqs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.root_dir)

    criterion = DiceCELoss(sigmoid=True, lambda_dice=0.5, lambda_ce=0.5).to(DEVICE)

    splits = get_splits(df_filtered)
    split_records = []

    for split_idx, (train_subjects, test_subjects) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"Split {split_idx+1}/5  |  "
              f"train={len(train_subjects)}  test={len(test_subjects)}")

        df_train = df_filtered[df_filtered["subject_id"].isin(train_subjects)]
        df_test  = df_filtered[df_filtered["subject_id"].isin(test_subjects)]

        train_list = build_datalist(df_train, root, args.seqs)
        test_list  = build_datalist(df_test,  root, args.seqs)
        print(f"  volumes with seg: train={len(train_list)}, test={len(test_list)}")

        train_ds = Dataset(data=train_list, transform=make_train_transforms())
        test_ds  = Dataset(data=test_list,  transform=make_val_transforms())

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=list_data_collate,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=1, shuffle=False,
            num_workers=args.num_workers, collate_fn=list_data_collate,
        )

        model = build_model(in_channels=in_channels).to(DEVICE)
        print(f"  trainable params: {sum(p.numel() for p in model.parameters()):,}")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
            scheduler.step()
            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                print(f"  epoch {epoch+1:3d}/{args.epochs} | loss={train_loss:.4f}")

        if split_idx == 3:
            seq_tag = "_".join(args.seqs)
            pred_dir = out_dir / "baseline" / seq_tag / f"split{split_idx + 1}"
            print(f"  saving predictions → {pred_dir}")
            dice, iou, per_sub_dice, per_sub_iou = save_predictions(model, test_loader, pred_dir)
        else:
            dice, iou, per_sub_dice, per_sub_iou = evaluate(model, test_loader)
        print(f"  Split {split_idx+1} → "
              f"dice={dice:.3f} ± {np.std(per_sub_dice):.3f} | "
              f"iou={iou:.3f}  ± {np.std(per_sub_iou):.3f}")

        split_records.append({
            "split":     split_idx + 1,
            "seqs":      "+".join(args.seqs),
            "n_train":   len(train_list),
            "n_test":    len(test_list),
            "dice_mean": dice,
            "dice_std":  float(np.std(per_sub_dice)),
            "iou_mean":  iou,
            "iou_std":   float(np.std(per_sub_iou)),
        })

        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()

    df_splits = pd.DataFrame(split_records)
    split_csv = out_dir / "split_metrics.csv"
    df_splits.to_csv(split_csv, index=False)
    print(f"\nSaved split metrics → {split_csv}")

    summary = df_splits[["dice_mean", "iou_mean"]].agg(["mean", "std"]).round(3)
    print("\nSummary across 5 splits:")
    print(summary.to_string())


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D U-Net segmentation baseline trained from scratch"
    )
    parser.add_argument("--csv_path",    type=str, required=True,
                        help="CSV with columns: id, img, seq, label, [subject_id]")
    parser.add_argument("--root_dir",    type=str, required=True,
                        help="Root directory for image paths in the CSV")
    parser.add_argument("--out_dir",     type=str, required=True,
                        help="Output directory for metrics CSV")
    parser.add_argument("--seqs",        type=str, nargs="+", default=["t2f"],
                        help="Sequence(s) to train on; multiple → multi-channel input (default: t2f)")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch_size",  type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr",          type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
