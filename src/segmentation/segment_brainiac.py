import os
import copy
import random
import argparse

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import monai
import monai.transforms.transform as _mtt
import monai.transforms.compose as _mtc
# MONAI 1.3.2 / Python 3.13 overflow fix (same as segment_3dino.py)
_mtt.MAX_SEED = int(np.iinfo(np.uint32).max)
_mtc.MAX_SEED = int(np.iinfo(np.uint32).max)
from monai.data import DataLoader, Dataset
from monai.data.utils import list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import UNETR
from monai.transforms import (
    Compose,
    EnsureTyped,
    LambdaD,
    LoadImaged,
    NormalizeIntensityd,
    Rand3DElasticd,
    RandBiasFieldd,
    RandFlipd,
    RandGaussianNoised,
    RandRotated,
    ResizeWithPadOrCropd,
    Resized,
)
from scipy.ndimage import zoom as nd_zoom
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
monai.utils.set_determinism(seed=SEED)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = (96, 96, 96)


# ── Model ──────────────────────────────────────────────────────────────────────
class BrainIACSegmentationModel(nn.Module):
    """
    UNETR decoder with the BrainIAC ViT-B encoder initialised from a SimCLR
    checkpoint, exactly as specified in the BrainIAC downstream-task docs.
    """
    def __init__(self, simclr_ckpt_path: str, freeze_encoder: bool = True):
        super().__init__()

        self.unetr = UNETR(
            in_channels=1,
            out_channels=1,
            img_size=IMAGE_SIZE,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )

        ckpt = torch.load(simclr_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        backbone_sd = {k[9:]: v for k, v in state_dict.items()
                       if k.startswith("backbone.")}

        # MONAI 1.3.2 creates position_embeddings of shape [1, 512, 768] for
        # img_size=96 while the checkpoint stores [1, 216, 768] (6^3 patches).
        # Load everything with strict=False to skip that one mismatch, then
        # directly replace the parameter with the checkpoint's tensor so the
        # forward pass receives the correct 216-token embedding for 96^3 inputs.
        self.unetr.vit.load_state_dict(backbone_sd, strict=False)
        pos_key = "patch_embedding.position_embeddings"
        if pos_key in backbone_sd:
            self.unetr.vit.patch_embedding.position_embeddings = nn.Parameter(
                backbone_sd[pos_key]
            )

        print("BrainIAC weights loaded into UNETR encoder.")

        if freeze_encoder:
            for param in self.unetr.vit.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.unetr(x)   # (B, 1, H, W, D) logits


# ── Transforms ─────────────────────────────────────────────────────────────────
def _binarize(x):
    return (x > 0).float()


def make_train_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Resized(keys=["image"], spatial_size=96, mode="trilinear",
                align_corners=False, size_mode="longest"),
        Resized(keys=["label"], spatial_size=96, mode="nearest",
                size_mode="longest"),
        ResizeWithPadOrCropd(keys=["image", "label"],
                             spatial_size=IMAGE_SIZE, mode="constant"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        LambdaD(keys=["label"], func=_binarize),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotated(keys=["image", "label"], range_x=0.1, prob=0.5,
                    mode=["bilinear", "nearest"]),
        Rand3DElasticd(keys=["image", "label"], sigma_range=(5, 8),
                       magnitude_range=(100, 200), prob=0.2,
                       mode=["bilinear", "nearest"]),
        RandBiasFieldd(keys=["image"], prob=0.3),
        RandGaussianNoised(keys=["image"], prob=0.2),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])


def make_val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Resized(keys=["image"], spatial_size=96, mode="trilinear",
                align_corners=False, size_mode="longest"),
        Resized(keys=["label"], spatial_size=96, mode="nearest",
                size_mode="longest"),
        ResizeWithPadOrCropd(keys=["image", "label"],
                             spatial_size=IMAGE_SIZE, mode="constant"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        LambdaD(keys=["label"], func=_binarize),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])


# ── Data helpers ───────────────────────────────────────────────────────────────
def filter_valid_subjects(df: pd.DataFrame, seq: str) -> pd.DataFrame:
    """Keep only subjects that have the requested sequence."""
    valid = df[df["seq"] == seq]["subject_id"].unique()
    return df[df["subject_id"].isin(valid) & (df["seq"] == seq)].copy()


def build_datalist(df_subjects: pd.DataFrame, root: Path, seq: str) -> list:
    records = []
    for subject_id, grp in df_subjects.groupby("subject_id"):
        mask_path = root / grp.iloc[0]["mask"]
        if not mask_path.exists():
            print(f"  [skip] {subject_id}: mask not found: {mask_path}")
            continue
        seq_rows = grp[grp["seq"] == seq]
        if seq_rows.empty:
            print(f"  [skip] {subject_id}: missing seq '{seq}'")
            continue
        img_path = root / seq_rows.iloc[0]["img"]
        if not img_path.exists():
            print(f"  [skip] {subject_id}: image not found: {img_path}")
            continue
        records.append({
            "subject_id": str(subject_id),
            "image":      str(img_path),
            "label":      str(mask_path),
            "image_path": str(img_path),
        })
    return records


# ── Splits (identical to all other experiments) ────────────────────────────────
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
def train_epoch(model, loader, optimizer, dice_loss, scaler):
    model.train()
    if not any(p.requires_grad for p in model.unetr.vit.parameters()):
        model.unetr.vit.eval()

    total_loss = 0.0
    for batch in loader:
        x = batch["image"].to(DEVICE)   # (B, 1, H, W, D)
        y = batch["label"].to(DEVICE)   # (B, 1, H, W, D) float 0/1

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = dice_loss(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, masks_dir=None):
    model.eval()
    all_dice, all_iou, per_sample = [], [], []
    for batch in loader:
        x   = batch["image"].to(DEVICE)
        gt  = batch["label"].cpu().numpy()[:, 0]   # (B, H, W, D)

        logits = sliding_window_inference(x, IMAGE_SIZE, 1, model, overlap=0.5)
        preds  = (torch.sigmoid(logits)[:, 0].cpu().numpy() > 0.5).astype(np.uint8)
        sids   = batch.get("subject_id",  [None] * preds.shape[0])
        ipaths = batch.get("image_path",  [None] * preds.shape[0])

        for i in range(preds.shape[0]):
            d, iou = compute_metrics(preds[i], (gt[i] > 0).astype(np.uint8))
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

    if "subject_id" not in df.columns:
        if df["id"].str.contains("_").any():
            df["subject_id"] = df["id"].str.rsplit("_", n=1).str[0]
        else:
            df["subject_id"] = df["id"]

    if args.seq not in df["seq"].unique():
        raise ValueError(
            f"Sequence '{args.seq}' not found in CSV. "
            f"Available: {df['seq'].unique().tolist()}"
        )

    df_seq = filter_valid_subjects(df, args.seq)
    if df_seq.empty:
        raise ValueError(f"No subjects with seq='{args.seq}'.")
    print(f"Sequence '{args.seq}': {df_seq['subject_id'].nunique()} subjects")

    # Extra training subjects (never used for testing)
    df_extra_train = None
    if args.extra_train_csv:
        df_extra = pd.read_csv(args.extra_train_csv)
        if "subject_id" not in df_extra.columns:
            if df_extra["id"].str.contains("_").any():
                df_extra["subject_id"] = df_extra["id"].str.rsplit("_", n=1).str[0]
            else:
                df_extra["subject_id"] = df_extra["id"]
        df_extra_train = filter_valid_subjects(df_extra, args.seq)
        n_extra = df_extra_train["subject_id"].nunique()
        print(f"Extra training subjects: {n_extra}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.root_dir)

    dice_loss = DiceCELoss(sigmoid=True, lambda_dice=0.5, lambda_ce=0.5).to(DEVICE)

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
        df_test  = df_seq[df_seq["subject_id"].isin(test_subjects)]

        train_list = build_datalist(df_train, root, args.seq)
        test_list  = build_datalist(df_test,  root, args.seq)

        if df_extra_train is not None:
            df_extra_split = df_extra_train[
                ~df_extra_train["subject_id"].isin(test_subjects)
            ]
            extra_list = build_datalist(df_extra_split, root, args.seq)
            print(f"  extra training subjects added: {len(extra_list)}")
            train_list = train_list + extra_list

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

        model = BrainIACSegmentationModel(
            simclr_ckpt_path=args.checkpoint_path,
            freeze_encoder=not args.train_encoder,
        ).to(DEVICE)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  trainable parameters: {trainable:,}")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(args.epochs):
            train_loss = train_epoch(
                model, train_loader, optimizer, dice_loss, scaler
            )
            scheduler.step()
            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                print(f"  epoch {epoch+1:3d}/{args.epochs} | loss={train_loss:.4f}")

        # Always save predictions for split 4; also save for a specific requested split
        if split_idx == 3:
            masks_dir = out_dir / "brainiac" / args.seq / f"split{split_idx + 1}"
            masks_dir.mkdir(parents=True, exist_ok=True)
            print(f"  saving predictions → {masks_dir}")
        elif args.split_idx is not None:
            masks_dir = out_dir / f"masks_split{split_idx + 1}"
            masks_dir.mkdir(parents=True, exist_ok=True)
        else:
            masks_dir = None

        dice, iou, per_sub_dice, per_sub_iou, per_sample = evaluate(
            model, test_loader, masks_dir=masks_dir
        )
        print(f"  Split {split_idx+1} → "
              f"dice={dice:.3f} ± {np.std(per_sub_dice):.3f} | "
              f"iou={iou:.3f}  ± {np.std(per_sub_iou):.3f}")

        split_records.append({
            "split":     split_idx + 1,
            "seq":       args.seq,
            "n_train":   len(train_list),
            "n_test":    len(test_list),
            "dice_mean": dice,
            "dice_std":  float(np.std(per_sub_dice)),
            "iou_mean":  iou,
            "iou_std":   float(np.std(per_sub_iou)),
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

        del model, optimizer, scheduler, scaler
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
        description="BrainIAC UNETR segmentation on a single MRI sequence with 5 stratified splits"
    )
    parser.add_argument("--csv_path",        type=str, required=True,
                        help="CSV with columns: id, img, seq, label, [subject_id]")
    parser.add_argument("--extra_train_csv", type=str, default=None,
                        help="Optional CSV of extra subjects added only to the training loader")
    parser.add_argument("--root_dir",        type=str, required=True,
                        help="Root directory for image paths in the CSV")
    parser.add_argument("--out_dir",         type=str, required=True,
                        help="Output directory for metrics CSV")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to BrainIAC SimCLR pretrained weights (.ckpt)")
    parser.add_argument("--seq",             type=str, default="t2f",
                        help="Sequence name to filter on (default: t2f)")
    parser.add_argument("--epochs",          type=int, default=100,
                        help="Training epochs per split")
    parser.add_argument("--batch_size",      type=int, default=4)
    parser.add_argument("--num_workers",     type=int, default=4)
    parser.add_argument("--lr",              type=float, default=5e-4,
                        help="Learning rate (BrainIAC default: 5e-4)")
    parser.add_argument("--train_encoder",   action="store_true",
                        help="Fine-tune the ViT encoder; default is frozen")
    parser.add_argument("--split_idx",       type=int, default=None,
                        help="Run only this split (1-indexed, 1-5); saves masks, "
                             "per-sample metrics and split info. Default runs all 5 splits.")
    args = parser.parse_args()
    main(args)
