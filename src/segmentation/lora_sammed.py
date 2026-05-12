"""
LoRA fine-tuning of SAM-Med3D for 3D brain tumour segmentation.

Only the image encoder's attention layers receive LoRA adapters;
the prompt encoder and mask decoder remain completely frozen.
A GT centroid point prompt is used during both training and evaluation.

Usage:
    python src/segmentation/lora_sammed.py \
        --csv_path csvs/input/bratsafrica.csv \
        --root_dir /projects/prjs0830/DATA \
        --out_dir csvs/results/seg_lora_sammed \
        --seq t2f
"""

import os
import math
import random
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
import monai.transforms.transform as _mtt
import monai.transforms.compose as _mtc
# MONAI 1.3.2 / Python 3.13 overflow fix
_mtt.MAX_SEED = int(np.iinfo(np.uint32).max)
_mtc.MAX_SEED = int(np.iinfo(np.uint32).max)
from monai.data import DataLoader, Dataset
from monai.data.utils import list_data_collate
from monai.losses import DiceCELoss
from monai.transforms import (
    Compose,
    EnsureTyped,
    LambdaD,
    LoadImaged,
    NormalizeIntensityd,
    RandFlipd,
    RandRotated,
    ResizeWithPadOrCropd,
    Resized,
)
from scipy.ndimage import zoom as nd_zoom
from sklearn.model_selection import StratifiedShuffleSplit

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
monai.utils.set_determinism(seed=SEED)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = (128, 128, 128)   # SAM-Med3D native input size


# ── LoRA ───────────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """Wraps nn.Linear with frozen base weight + trainable low-rank adapters."""
    def __init__(self, linear: nn.Linear, rank: int):
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.scale = 1.0 / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.linear(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


_ATTN_KEYWORDS = ('attn', 'attention', 'self_attn')


def apply_lora(model: nn.Module, rank: int = 4) -> int:
    """Replace every nn.Linear inside attention sub-modules with LoRALinear."""
    to_replace = []
    for mod_name, module in model.named_modules():
        if any(kw in mod_name.lower() for kw in _ATTN_KEYWORDS):
            for child_name, child in list(module.named_children()):
                if isinstance(child, nn.Linear):
                    to_replace.append((module, child_name, child))
    for parent, child_name, child in to_replace:
        setattr(parent, child_name, LoRALinear(child, rank))
    return len(to_replace)


def lora_state(model: nn.Module) -> dict:
    return {k: v.clone() for k, v in model.state_dict().items() if 'lora_' in k}


def reset_lora(model: nn.Module, state: dict):
    current = model.state_dict()
    current.update(state)
    model.load_state_dict(current)


# ── Model ──────────────────────────────────────────────────────────────────────
class SAMMed3DLoRA(nn.Module):
    """
    SAM-Med3D with LoRA adapters on the image encoder attention layers only.
    The prompt encoder and mask decoder remain completely frozen.

    During forward, a GT-derived centroid point prompt must be supplied.
    Gradients flow through the mask decoder's use of image_embedding back
    to the LoRA parameters; the decoder's own weights are never updated.
    """
    def __init__(self, lora_rank: int = 4, checkpoint_path: str = None):
        super().__init__()
        import medim
        ckpt = "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
        sam = medim.create_model('SAM-Med3D', pretrained=True, checkpoint_path=ckpt)

        self.image_encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder   = sam.mask_decoder
        del sam

        # Freeze everything, then re-enable LoRA parameters only
        for p in self.parameters():
            p.requires_grad_(False)

        n_lora = apply_lora(self.image_encoder, rank=lora_rank)
        print(f"LoRA applied: {n_lora} attention linears in image encoder (rank={lora_rank})")

    def forward(
        self,
        image: torch.Tensor,         # (B, 1, 128, 128, 128)
        point_coords: torch.Tensor,  # (B, N, 3)  voxel coordinates
        point_labels: torch.Tensor,  # (B, N)     1=foreground, 0=background
    ) -> torch.Tensor:               # returns (B, 1, 128, 128, 128) logits
        image_embedding = self.image_encoder(image)   # (B, 384, 8, 8, 8)

        # Prompt encoder is frozen and not on the gradient path of image_embedding
        with torch.no_grad():
            sparse_emb, dense_emb = self.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )
            image_pe = self.prompt_encoder.get_dense_pe()

        # Mask decoder weights are frozen but gradients must still flow through
        # image_embedding to reach the LoRA parameters → no torch.no_grad() here
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )

        return F.interpolate(
            low_res_masks,
            size=image.shape[2:],
            mode='trilinear',
            align_corners=False,
        )


# ── Prompt helpers ─────────────────────────────────────────────────────────────
def get_random_fg_prompts(
    label_batch: torch.Tensor,
    device: torch.device,
    n_points: int = 10,
):
    """
    Sample up to n_points random foreground voxels from each label volume.
    If the mask has fewer positive voxels than n_points, all of them are used
    (with replacement sampling to always return exactly n_points).

    label_batch : (B, 1, H, W, D) float
    Returns
        point_coords : (B, n_points, 3) – sampled voxel indices
        point_labels : (B, n_points)    – all ones (foreground)
    """
    B = label_batch.shape[0]
    coords = []
    for b in range(B):
        mask = label_batch[b, 0]                       # (H, W, D)
        pos  = mask.nonzero(as_tuple=False).float()    # (N, 3)
        if pos.shape[0] == 0:
            # No foreground: fall back to volume centre repeated n_points times
            centre = torch.tensor(
                [s / 2.0 for s in mask.shape], dtype=torch.float32
            )
            sampled = centre.unsqueeze(0).expand(n_points, -1)
        elif pos.shape[0] >= n_points:
            idx = torch.randperm(pos.shape[0])[:n_points]
            sampled = pos[idx]                         # (n_points, 3)
        else:
            # Fewer voxels than requested: sample with replacement
            idx = torch.randint(pos.shape[0], (n_points,))
            sampled = pos[idx]                         # (n_points, 3)
        coords.append(sampled)                         # (n_points, 3)
    point_coords = torch.stack(coords, dim=0).to(device)              # (B, n_points, 3)
    point_labels = torch.ones(B, n_points, dtype=torch.long, device=device)
    return point_coords, point_labels


# ── Transforms ─────────────────────────────────────────────────────────────────
def _binarize(x):
    return (x > 0).float()


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
        LambdaD(keys=["label"], func=_binarize),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotated(keys=["image", "label"], range_x=0.1, prob=0.5,
                    mode=["bilinear", "nearest"]),
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
def train_epoch(model, loader, optimizer, dice_loss, scaler, n_points):
    model.train()
    # Frozen components stay in eval mode so their BN statistics are not updated
    model.prompt_encoder.eval()
    model.mask_decoder.eval()

    total_loss = 0.0
    for batch in loader:
        x = batch["image"].to(DEVICE)   # (B, 1, 128, 128, 128)
        y = batch["label"].to(DEVICE)   # (B, 1, 128, 128, 128)

        point_coords, point_labels = get_random_fg_prompts(y, DEVICE, n_points)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(x, point_coords, point_labels)
            loss   = dice_loss(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, n_points, masks_dir=None):
    model.eval()
    all_dice, all_iou, per_sample = [], [], []
    for batch in loader:
        x  = batch["image"].to(DEVICE)
        y  = batch["label"].to(DEVICE)
        gt = y.cpu().numpy()[:, 0]     # (B, H, W, D)

        point_coords, point_labels = get_random_fg_prompts(y, DEVICE, n_points)

        with torch.cuda.amp.autocast():
            logits = model(x, point_coords, point_labels)

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

    # Build model once; LoRA weights are reset to this initial state each split
    model = SAMMed3DLoRA(
        lora_rank=args.lora_rank,
        checkpoint_path=args.checkpoint_path,
    ).to(DEVICE)

    initial_lora = lora_state(model)
    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {trainable_total:,}")
    print(f"Prompt: {args.num_points} random foreground points per volume")

    dice_loss    = DiceCELoss(sigmoid=True, lambda_dice=0.5, lambda_ce=0.5).to(DEVICE)
    splits       = get_splits(df_seq)
    split_records = []

    for split_idx, (train_subjects, test_subjects) in enumerate(splits):
        # Skip splits that are not requested
        if args.split_idx is not None and split_idx + 1 != args.split_idx:
            continue

        print(f"\n{'='*60}")
        print(f"Split {split_idx + 1}/5  |  "
              f"train={len(train_subjects)}  test={len(test_subjects)}")

        reset_lora(model, initial_lora)

        df_train = df_seq[df_seq["subject_id"].isin(train_subjects)]
        df_test  = df_seq[df_seq["subject_id"].isin(test_subjects)]

        train_list = build_datalist(df_train, root)
        test_list  = build_datalist(df_test,  root)
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
                model, train_loader, optimizer, dice_loss, scaler, args.num_points
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
            model, test_loader, args.num_points, masks_dir=masks_dir
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

        del optimizer, scheduler, scaler
        torch.cuda.empty_cache()

    df_splits = pd.DataFrame(split_records)
    split_csv  = out_dir / "split_metrics.csv"
    df_splits.to_csv(split_csv, index=False)
    print(f"\nSaved split metrics → {split_csv}")

    summary = df_splits[["dice_mean", "iou_mean"]].agg(["mean", "std"]).round(3)
    print("\nSummary across evaluated splits:")
    print(summary.to_string())


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM-Med3D LoRA segmentation – 5 stratified splits"
    )
    parser.add_argument("--csv_path",        type=str, required=True,
                        help="CSV with columns: id, img, seq, label, [subject_id]")
    parser.add_argument("--root_dir",        type=str, required=True,
                        help="Root directory for image paths in the CSV")
    parser.add_argument("--out_dir",         type=str, required=True,
                        help="Output directory for metrics CSV")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="SAM-Med3D checkpoint (.pth); defaults to turbo from HuggingFace")
    parser.add_argument("--seq",             type=str, default="t2f",
                        help="Sequence to filter on (default: t2f)")
    parser.add_argument("--lora_rank",       type=int, default=4,
                        help="LoRA rank (default: 4)")
    parser.add_argument("--num_points",      type=int, default=10,
                        help="Foreground points sampled randomly per volume for the prompt (default: 10)")
    parser.add_argument("--epochs",          type=int, default=100,
                        help="Training epochs per split (default: 100)")
    parser.add_argument("--batch_size",      type=int, default=2,
                        help="Batch size; 128³ volumes are memory-heavy (default: 2)")
    parser.add_argument("--num_workers",     type=int, default=4)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--split_idx",       type=int, default=None,
                        help="Run only this split (1-indexed, 1-5); saves masks, "
                             "per-sample metrics and split info. Default runs all 5 splits.")
    args = parser.parse_args()
    main(args)
