"""
LoRA fine-tuning for brain MRI classification.

For multi-sequence datasets (BraTS-Africa): all sequences of a subject are
forwarded through the shared backbone, their feature vectors are concatenated,
and the result is fed to the classifier.
For single-sequence datasets (NigerianBrain): standard single-forward pass.

Usage:
    python src/lora/lora.py \
        --model brainiac \
        --csv_path csvs/input/bratsafrica.csv \
        --root_dir /projects/prjs0830/DATA \
        --output_csv csvs/results/lora_bratsafrica_brainiac.csv \
        --checkpoint_path /projects/0/prjs0830/checkpoints/BrainIAC.ckpt
"""

import sys
import math
import copy
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
import SimpleITK as sitk
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─────────────────────────────────────────────────────────────────────────────
# LoRA
# ─────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a frozen base weight and trainable low-rank adapters."""
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


ATTN_KEYWORDS = ('attn', 'attention', 'self_attn')


def apply_lora(model: nn.Module, rank: int = 4) -> int:
    """Replace every nn.Linear inside an attention sub-module with a LoRALinear."""
    to_replace = []
    for mod_name, module in model.named_modules():
        if any(kw in mod_name.lower() for kw in ATTN_KEYWORDS):
            for child_name, child in list(module.named_children()):
                if isinstance(child, nn.Linear):
                    to_replace.append((module, child_name, child))
    for parent, child_name, child in to_replace:
        setattr(parent, child_name, LoRALinear(child, rank))
    return len(to_replace)


def freeze_non_lora(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad_('lora_' in name)


def lora_state(model: nn.Module) -> dict:
    """Return a copy of only the LoRA parameter values."""
    return {k: v.clone() for k, v in model.state_dict().items() if 'lora_' in k}


def reset_lora(model: nn.Module, state: dict):
    """Restore LoRA parameters to a previously saved state (for per-split resets)."""
    current = model.state_dict()
    current.update(state)
    model.load_state_dict(current)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def make_monai_preprocess(target_size: int = 96):
    """Returns a function path -> tensor (1, D, H, W) using MONAI transforms."""
    from monai.transforms import (
        Compose, LoadImage, Resize, ResizeWithPadOrCrop, NormalizeIntensity,
    )
    t = Compose([
        LoadImage(image_only=True, ensure_channel_first=True),
        Resize(spatial_size=target_size, size_mode='longest',
               mode='trilinear', align_corners=False),
        ResizeWithPadOrCrop(spatial_size=(target_size, target_size, target_size)),
        NormalizeIntensity(nonzero=True),
    ])
    return lambda path: t(str(path))


def make_sam2_preprocess(target_size: int = 512):
    """Returns a function path -> tensor (D, 3, H, W) matching SAM2 preprocessing."""
    IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(path: str) -> torch.Tensor:
        volume = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
        v_min, v_max = volume.min(), volume.max()
        if v_max > v_min:
            volume = (volume - v_min) / (v_max - v_min) * 255.0
        volume = volume.astype(np.uint8)
        D = volume.shape[0]
        out = np.empty((D, 3, target_size, target_size), dtype=np.float32)
        for i in range(D):
            arr = np.array(
                Image.fromarray(volume[i]).convert('RGB').resize((target_size, target_size))
            ).transpose(2, 0, 1).astype(np.float32) / 255.0
            out[i] = (arr - IMG_MEAN[:, None, None]) / IMG_STD[:, None, None]
        return torch.from_numpy(out)

    return preprocess


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

SubjectInfo = tuple  # (subject_id: str, label: int, seqs: list[(seq_name, img_path)])


def build_subject_list(df: pd.DataFrame) -> list:
    """
    Parse a feature-extraction-style CSV into a list of SubjectInfo tuples.
    Derives subject_id and seq from the id column when not explicitly present
    (BraTS-style ids: "<subject>_<seq>").
    """
    df = df.copy()
    if 'subject_id' not in df.columns:
        if df['id'].str.contains('_').any():
            split = df['id'].str.rsplit('_', n=1)
            df['subject_id'] = split.str[0]
            df['seq'] = split.str[1]
        else:
            df['subject_id'] = df['id']

    subjects = []
    for sid, group in df.groupby('subject_id', sort=True):
        label = int(group['label'].iloc[0])
        if 'seq' in group.columns:
            seqs = [(r['seq'], r['img']) for _, r in group.sort_values('seq').iterrows()]
        else:
            seqs = [('img', r['img']) for _, r in group.iterrows()]
        subjects.append((sid, label, seqs))
    return subjects


class SubjectDataset(Dataset):
    def __init__(self, subjects: list, root_dir: str, preprocess_fn):
        self.subjects = subjects
        self.root_dir = Path(root_dir)
        self.preprocess = preprocess_fn

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        sid, label, seqs = self.subjects[idx]
        imgs = [self.preprocess(self.root_dir / img_path) for _, img_path in seqs]
        return {
            'subject_id': sid,
            'label': torch.tensor(label, dtype=torch.float32),
            'imgs': imgs,
        }


def collate_subjects(batch):
    """Stack subjects into batched tensors, one tensor per sequence position."""
    subject_ids = [b['subject_id'] for b in batch]
    labels = torch.stack([b['label'] for b in batch])
    n_seqs = len(batch[0]['imgs'])
    imgs_per_seq = [torch.stack([b['imgs'][i] for b in batch]) for i in range(n_seqs)]
    return {'subject_id': subject_ids, 'label': labels, 'imgs': imgs_per_seq}


# ─────────────────────────────────────────────────────────────────────────────
# Backbone wrappers
# ─────────────────────────────────────────────────────────────────────────────

class BrainIACBackbone(nn.Module):
    """MONAI ViT-B/16, outputs CLS token (B, 768)."""
    feat_dim = 768

    def __init__(self, ckpt_path: str):
        super().__init__()
        from monai.networks.nets import ViT
        self.vit = ViT(
            in_channels=1, img_size=(96, 96, 96), patch_size=(16, 16, 16),
            hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12, save_attn=True,
        )
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('state_dict', ckpt)
        sd = {k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')}
        self.vit.load_state_dict(sd, strict=True)

    def forward(self, x):
        return self.vit(x)[0][:, 0]   # CLS token → (B, 768)


class SAMMed3DBackbone(nn.Module):
    """SAM-Med3D image encoder with global average pooling, outputs (B, 384)."""
    feat_dim = 384

    def __init__(self):
        super().__init__()
        import medim
        sam = medim.create_model(
            'SAM-Med3D', pretrained=True,
            checkpoint_path='https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth',
        )
        self.enc = copy.deepcopy(sam.image_encoder)
        del sam
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        return self.pool(self.enc(x)).flatten(1)   # (B, 384)


class DINOBackbone(nn.Module):
    """3D DINOv2 backbone, output dim probed at construction time."""

    def __init__(self, ckpt_path: str):
        super().__init__()
        DINO_ROOT = Path(__file__).resolve().parents[2] / '3DINO'
        sys.path.insert(0, str(DINO_ROOT))
        from dinov2.eval.setup import build_model_for_eval
        from dinov2.configs import load_and_merge_config_3d
        cfg = load_and_merge_config_3d('train/vit3d_highres')
        self.model = build_model_for_eval(cfg, ckpt_path)
        # Probe feature dim
        with torch.no_grad():
            dummy_device = next(self.model.parameters()).device
            dummy = torch.zeros(1, 1, 96, 96, 96, device=dummy_device)
            out = self.model(dummy)
        self.feat_dim = out.shape[-1]

    def forward(self, x):
        return self.model(x)


class MedSAM2Backbone(nn.Module):
    """MedSAM2 Hiera image encoder, outputs (B, 256)."""
    feat_dim = 256

    def __init__(self, config_path: str, ckpt_path: str):
        super().__init__()
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from sam2.build_sam import build_sam2_video_predictor_npz
        CONFIG_DIR = Path('MedSAM2/config')
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(CONFIG_DIR), job_name='sam2_lora'):
            predictor = build_sam2_video_predictor_npz(config_path, ckpt_path)
            self.enc = predictor.image_encoder
        del predictor

    def forward(self, x, slice_chunk: int = 8):
        # x: (B, D, 3, H, W)
        B, D, C, H, W = x.shape
        dtype = next(self.enc.parameters()).dtype
        flat = x.view(B * D, C, H, W).to(DEVICE, dtype=dtype)
        chunks = []
        for start in range(0, B * D, slice_chunk):
            chunk_feats = self.enc(flat[start:start + slice_chunk])['vision_features']
            chunks.append(chunk_feats.mean((-2, -1)))  # (chunk, 256)
        feats = torch.cat(chunks, dim=0)               # (B*D, 256)
        return feats.view(B, D, -1).mean(1).float()    # (B, 256)


class MedicalSAM2Backbone(nn.Module):
    """MedicalSAM2 Hiera image encoder, outputs (B, 256)."""
    feat_dim = 256

    def __init__(self, config_path: str, ckpt_path: str):
        super().__init__()
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from sam2.build_sam import build_sam2_video_predictor
        CONFIG_DIR = Path('MedSAM2/config')
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(CONFIG_DIR), job_name='sam2_lora'):
            predictor = build_sam2_video_predictor(config_path, ckpt_path)
            self.enc = predictor.image_encoder
        del predictor

    def forward(self, x, slice_chunk: int = 8):
        B, D, C, H, W = x.shape
        dtype = next(self.enc.parameters()).dtype
        flat = x.view(B * D, C, H, W).to(DEVICE, dtype=dtype)
        chunks = []
        for start in range(0, B * D, slice_chunk):
            chunk_feats = self.enc(flat[start:start + slice_chunk])['vision_features']
            chunks.append(chunk_feats.mean((-2, -1)))
        feats = torch.cat(chunks, dim=0)
        return feats.view(B, D, -1).mean(1).float()


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

class Classifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _forward_batch(backbone, batch, n_seqs):
    """Encode all sequences and concatenate into a single feature vector per subject."""
    seq_feats = [backbone(x.to(DEVICE)) for x in batch['imgs']]
    return torch.cat(seq_feats, dim=1) if n_seqs > 1 else seq_feats[0]


def run_experiment(
    subjects: list,
    backbone: nn.Module,
    preprocess_fn,
    n_seqs: int,
    feat_dim: int,
    n_splits: int = 5,
    lr: float = 1e-4,
    epochs: int = 50,
    batch_size: int = 4,
    out_path: str = 'results_lora.csv',
):
    labels_arr = np.array([s[1] for s in subjects])
    X_idx = np.arange(len(subjects))

    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=42)
    criterion = nn.BCEWithLogitsLoss()

    # Save initial LoRA state so we can reset between splits
    initial_lora = lora_state(backbone)

    results = []

    for split_idx, (train_idx, test_idx) in enumerate(splitter.split(X_idx, labels_arr)):
        print(f'\n── Split {split_idx + 1}/{n_splits} '
              f'(train={len(train_idx)}, test={len(test_idx)}) ──')

        # Reset LoRA weights to pre-training state
        reset_lora(backbone, initial_lora)

        classifier = Classifier(input_dim=feat_dim * n_seqs).to(DEVICE)

        trainable = (
            [p for n, p in backbone.named_parameters() if 'lora_' in n]
            + list(classifier.parameters())
        )
        optimizer = optim.Adam(trainable, lr=lr)

        train_ds = SubjectDataset([subjects[i] for i in train_idx], args_root_dir, preprocess_fn)
        test_ds  = SubjectDataset([subjects[i] for i in test_idx],  args_root_dir, preprocess_fn)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate_subjects, num_workers=0, drop_last=True)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                                  collate_fn=collate_subjects, num_workers=0)

        scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == 'cuda'))
        for epoch in range(epochs):
            backbone.train()
            classifier.train()
            for batch in train_loader:
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
                    feats = _forward_batch(backbone, batch, n_seqs)
                    labels = batch['label'].to(DEVICE).unsqueeze(1)
                    loss = criterion(classifier(feats), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        backbone.eval()
        classifier.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                feats = _forward_batch(backbone, batch, n_seqs)
                probs = torch.sigmoid(classifier(feats)).cpu().numpy().flatten()
                all_probs.extend(probs)
                all_labels.extend(batch['label'].numpy())

        all_probs  = np.array(all_probs)
        all_labels = np.array(all_labels)
        preds = (all_probs > 0.5).astype(int)

        bal_acc = balanced_accuracy_score(all_labels, preds)
        f1      = f1_score(all_labels, preds)
        auc     = roc_auc_score(all_labels, all_probs)
        print(f'  bal_acc={bal_acc:.3f}  f1={f1:.3f}  auc={auc:.3f}')
        results.append({'split': split_idx + 1, 'bal_acc': bal_acc, 'f1': f1, 'auc': auc})

    df_results = pd.DataFrame(results)
    df_results.to_csv(out_path, index=False)
    print(f'\nSaved to {out_path}')
    print(df_results[['bal_acc', 'f1', 'auc']].agg(['mean', 'std']).round(3).to_string())
    return results, df_results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

# Module-level variable so run_experiment can access root_dir without threading
# through every call; set by main() before training starts.
args_root_dir: str = ''


def main(args):
    global args_root_dir
    args_root_dir = args.root_dir

    # ── Build backbone ───────────────────────────────────────────────────────
    if args.model == 'brainiac':
        backbone = BrainIACBackbone(args.checkpoint_path)
        preprocess_fn = make_monai_preprocess(target_size=96)
    elif args.model == 'sammed':
        backbone = SAMMed3DBackbone()
        preprocess_fn = make_monai_preprocess(target_size=128)
    elif args.model == '3dino':
        backbone = DINOBackbone(args.checkpoint_path)
        preprocess_fn = make_monai_preprocess(target_size=96)
    elif args.model == 'medsam2':
        backbone = MedSAM2Backbone(args.config_path, args.checkpoint_path)
        preprocess_fn = make_sam2_preprocess(target_size=512)
    elif args.model == 'medicalsam2':
        backbone = MedicalSAM2Backbone(args.config_path, args.checkpoint_path)
        preprocess_fn = make_sam2_preprocess(target_size=512)
    else:
        raise ValueError(f'Unknown model: {args.model}')

    feat_dim = backbone.feat_dim

    # ── Freeze backbone, apply LoRA ──────────────────────────────────────────
    for p in backbone.parameters():
        p.requires_grad_(False)

    n_lora = apply_lora(backbone, rank=args.lora_rank)
    print(f'[{args.model}] LoRA applied to {n_lora} attention linears (rank={args.lora_rank})')

    backbone = backbone.to(DEVICE)

    # ── Load data ────────────────────────────────────────────────────────────
    df = pd.read_csv(args.csv_path)
    subjects = build_subject_list(df)
    if args.seq is not None:
        filtered = []
        for sid, label, seqs in subjects:
            match = [(name, path) for name, path in seqs if name == args.seq]
            if match:
                filtered.append((sid, label, match))
            else:
                print(f'  [warn] subject {sid} has no sequence "{args.seq}", skipping')
        subjects = filtered
    n_seqs = len(subjects[0][2])   # number of sequences per subject
    print(f'Subjects: {len(subjects)}  |  sequences per subject: {n_seqs}'
          f'  |  classifier input: {feat_dim * n_seqs}')

    # ── Train ────────────────────────────────────────────────────────────────
    run_experiment(
        subjects=subjects,
        backbone=backbone,
        preprocess_fn=preprocess_fn,
        n_seqs=n_seqs,
        feat_dim=feat_dim,
        n_splits=args.num_splits,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        out_path=args.output_csv,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LoRA fine-tuning for brain MRI classification')
    parser.add_argument('--model', required=True,
                        choices=['sammed', 'brainiac', '3dino', 'medsam2', 'medicalsam2'])
    parser.add_argument('--csv_path',       required=True)
    parser.add_argument('--root_dir',       required=True)
    parser.add_argument('--output_csv',     default='results_lora.csv')
    parser.add_argument('--checkpoint_path', default=None,
                        help='Required for brainiac, 3dino, medsam2, medicalsam2')
    parser.add_argument('--config_path',    default=None,
                        help='Hydra config yaml name for medsam2/medicalsam2')
    parser.add_argument('--seq',            default=None,
                        help='Use a single sequence (e.g. t2f) instead of concatenating all')
    parser.add_argument('--lora_rank',  type=int,   default=4)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=4)
    parser.add_argument('--num_splits', type=int,   default=5)
    args = parser.parse_args()
    main(args)
