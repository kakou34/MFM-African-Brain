"""
DenseNet121 classification baseline trained from scratch on raw MRI volumes.
Matches the same 5-split StratifiedShuffleSplit protocol used across all models.
"""
import os
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import monai
import monai.transforms.transform as _mtt
import monai.transforms.compose as _mtc

_mtt.MAX_SEED = int(np.iinfo(np.uint32).max)
_mtc.MAX_SEED = int(np.iinfo(np.uint32).max)

from monai.networks.nets import DenseNet121
from monai.transforms import (
    Compose,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    ResizeWithPadOrCropd,
    Resized,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
monai.utils.set_determinism(seed=SEED)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = (128, 128, 128)


# ── Transforms ─────────────────────────────────────────────────────────────────

def make_train_transforms():
    return Compose([
        LoadImaged(keys=["image"], ensure_channel_first=True),
        Resized(keys=["image"], spatial_size=128, mode="trilinear",
                align_corners=False, size_mode="longest"),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=IMAGE_SIZE, mode="constant"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])


def make_val_transforms():
    return Compose([
        LoadImaged(keys=["image"], ensure_channel_first=True),
        Resized(keys=["image"], spatial_size=128, mode="trilinear",
                align_corners=False, size_mode="longest"),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=IMAGE_SIZE, mode="constant"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])


# ── Dataset ────────────────────────────────────────────────────────────────────

class SubjectDataset(torch.utils.data.Dataset):
    """
    One item per subject. Handles three multi-sequence modes:
      'single' – one image per subject → (1, H, W, D)
      'mean'   – all seqs stacked     → (n_seqs, 1, H, W, D); logits averaged at forward
      'concat' – all seqs as channels → (n_seqs, H, W, D)
    """
    def __init__(self, records, effective_mode, seqs, transform):
        self.records = records
        self.mode = effective_mode
        self.seqs = seqs
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        label = torch.tensor(rec["label"], dtype=torch.float32)

        if self.mode == "concat":
            imgs = [
                self.transform({"image": rec["imgs"][s]})["image"]  # (1, H, W, D)
                for s in self.seqs
            ]
            return torch.cat(imgs, dim=0), label  # (n_seqs, H, W, D)

        if self.mode == "mean":
            imgs = [
                self.transform({"image": rec["imgs"][s]})["image"]  # (1, H, W, D)
                for s in self.seqs
            ]
            return torch.stack(imgs, dim=0), label  # (n_seqs, 1, H, W, D)

        # single
        return self.transform({"image": rec["imgs"][self.seqs[0]]})["image"], label


# ── Data helpers ────────────────────────────────────────────────────────────────

def prepare_data(df, args):
    """Parse CSV and return (df, effective_mode, seqs, in_channels)."""
    if "subject_id" not in df.columns:
        if df["id"].str.contains("_").any():
            df["subject_id"] = df["id"].str.rsplit("_", n=1).str[0]
            df["seq"]        = df["id"].str.rsplit("_", n=1).str[1]
        else:
            df["subject_id"] = df["id"]

    if "seq" not in df.columns:
        df["seq"] = "single"
        return df, "single", ["single"], 1

    seqs = sorted(df["seq"].unique().tolist())
    mode = args.mode

    if mode == "mean":
        return df, "mean", seqs, 1

    if mode == "concat":
        return df, "concat", seqs, len(seqs)

    if mode not in seqs:
        raise ValueError(f"Sequence '{mode}' not found. Available: {seqs}")
    df = df[df["seq"] == mode].copy()
    return df, "single", [mode], 1


def build_records(df, root, effective_mode, seqs):
    """Build per-subject record list with resolved image paths."""
    records = []
    for subj_id, group in df.groupby("subject_id"):
        label = int(group["label"].iloc[0])
        imgs = {}
        valid = True
        for seq in seqs:
            rows = group[group["seq"] == seq]
            if rows.empty:
                print(f"  [skip] {subj_id}: missing seq '{seq}'")
                valid = False
                break
            imgs[seq] = str(root / rows["img"].iloc[0])
        if valid:
            records.append({"subject_id": subj_id, "label": label, "imgs": imgs})
    return records


def get_splits(df):
    subjects = (
        df.drop_duplicates("subject_id")[["subject_id", "label"]]
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    X = subjects["subject_id"].values
    y = subjects["label"].values
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=42)
    return [
        (set(X[ti]), set(X[vi]))
        for ti, vi in sss.split(X, y)
    ]


# ── Forward pass ───────────────────────────────────────────────────────────────

def _forward(model, imgs, effective_mode):
    if effective_mode == "mean":
        # imgs: (B, n_seqs, 1, H, W, D) → average logits over sequences
        n_seqs = imgs.shape[1]
        logits = torch.stack(
            [model(imgs[:, s].to(DEVICE)) for s in range(n_seqs)], dim=0
        ).mean(dim=0)
    else:
        logits = model(imgs.to(DEVICE))
    return logits


# ── Training / evaluation ──────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, effective_mode):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        labels = labels.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = _forward(model, imgs, effective_mode)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, effective_mode):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        logits = _forward(model, imgs, effective_mode)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy().flatten())
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > 0.5).astype(int)
    bal_acc = balanced_accuracy_score(all_labels, preds)
    f1      = f1_score(all_labels, preds)
    auc     = roc_auc_score(all_labels, all_probs)
    return bal_acc, f1, auc


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    df   = pd.read_csv(args.csv_path)
    root = Path(args.root_dir)

    df, effective_mode, seqs, in_channels = prepare_data(df, args)
    print(f"Mode: {args.mode} → effective={effective_mode} | "
          f"seqs={seqs} | in_channels={in_channels} | "
          f"subjects={df['subject_id'].nunique()}")

    splits  = get_splits(df)
    records = build_records(df, root, effective_mode, seqs)
    subj_to_rec = {r["subject_id"]: r for r in records}

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for split_idx, (train_subjects, test_subjects) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"Split {split_idx+1}/5 | train={len(train_subjects)} test={len(test_subjects)}")

        train_recs = [subj_to_rec[s] for s in train_subjects if s in subj_to_rec]
        test_recs  = [subj_to_rec[s] for s in test_subjects  if s in subj_to_rec]
        print(f"  records: train={len(train_recs)} test={len(test_recs)}")

        train_ds = SubjectDataset(train_recs, effective_mode, seqs, make_train_transforms())
        test_ds  = SubjectDataset(test_recs,  effective_mode, seqs, make_val_transforms())

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
        )

        model = DenseNet121(
            spatial_dims=3, in_channels=in_channels, out_channels=1
        ).to(DEVICE)
        print(f"  trainable params: {sum(p.numel() for p in model.parameters()):,}")

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(args.epochs):
            loss = train_epoch(model, train_loader, optimizer, criterion, scaler, effective_mode)
            scheduler.step()
            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                print(f"  epoch {epoch+1:3d}/{args.epochs} | loss={loss:.4f}")

        bal_acc, f1, auc = evaluate(model, test_loader, effective_mode)
        print(f"  Split {split_idx+1} → bal_acc={bal_acc:.3f}  f1={f1:.3f}  auc={auc:.3f}")

        results.append({
            "split":   split_idx + 1,
            "mode":    args.mode,
            "bal_acc": bal_acc,
            "f1":      f1,
            "auc":     auc,
        })

        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()

    df_res = pd.DataFrame(results)
    df_res.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(df_res[["bal_acc", "f1", "auc"]].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DenseNet121 classification baseline trained from scratch"
    )
    parser.add_argument("--csv_path",    type=str, required=True,
                        help="CSV with id, img, label, [seq], [subject_id]")
    parser.add_argument("--root_dir",    type=str, required=True,
                        help="Root directory for image paths in the CSV")
    parser.add_argument("--output_csv",  type=str, default="results_baseline_cls.csv")
    parser.add_argument("--mode",        type=str, default="mean",
                        help="'mean', 'concat', or a sequence name (e.g. 't2f')")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch_size",  type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    main(args)
