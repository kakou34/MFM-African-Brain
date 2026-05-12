"""
Count trainable parameters for each experiment configuration used in the paper.

Classification
  • Linear probing  – MLP head on frozen features
  • LoRA            – LoRA adapters (rank=4) + MLP head
  • Baseline        – DenseNet121 from scratch

Segmentation
  • BrainIAC        – UNETR decoder, ViT encoder frozen
  • 3D-DINO         – UNETRHead decoder, ViT encoder frozen
  • SAM-Med3D+LoRA  – LoRA adapters in image encoder only (rank=4)
  • Baseline        – 3D U-Net from scratch

Usage:
    python count_experiment_params.py
"""

import sys
import math
from functools import partial
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent
DINO3_ROOT = REPO_ROOT / "3DINO"
sys.path.insert(0, str(DINO3_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ───────────────────────────────────────────────────────────────────

def n_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fmt(n):
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def report(label, fn):
    try:
        n = fn()
        print(f"  {label:<55s}  {n:>12,}  ({fmt(n)})")
    except Exception as e:
        print(f"  {label:<55s}  ERROR: {e}")


# ── LoRA (identical to src/lora/lora.py and src/segmentation/lora_sammed.py) ──

_ATTN_KEYWORDS = ("attn", "attention", "self_attn")


class _LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int):
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.linear(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) / self.lora_A.shape[0]


def _apply_lora(model: nn.Module, rank: int = 4) -> int:
    to_replace = []
    for mod_name, module in model.named_modules():
        if any(kw in mod_name.lower() for kw in _ATTN_KEYWORDS):
            for child_name, child in list(module.named_children()):
                if isinstance(child, nn.Linear):
                    to_replace.append((module, child_name, child))
    for parent, child_name, child in to_replace:
        setattr(parent, child_name, _LoRALinear(child, rank))
    return len(to_replace)


# ── MLP head (matches EnhancedLinearClassifier in classify_disease.py) ────────

def _mlp_head(input_dim: int, hidden_dim: int = 128) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),     nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(hidden_dim // 2, 1),
    )


# ── Feature dimensions per backbone (single sequence) ─────────────────────────

FEAT_DIMS = {
    "SAM-Med3D":    384,
    "BrainIAC":     768,
    "MedSAM2":      256,
    "MedicalSAM2":  256,
}


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

# ── Linear probing ────────────────────────────────────────────────────────────

def linprobe_params(feat_dim: int) -> int:
    return n_trainable(_mlp_head(feat_dim))


def linprobe_3dino_params() -> int:
    from dinov2.configs import load_and_merge_config_3d
    from dinov2.models import build_model_from_cfg
    cfg = load_and_merge_config_3d("train/vit3d_highres")
    model, _ = build_model_from_cfg(cfg, only_teacher=True)
    return n_trainable(_mlp_head(model.num_features))


# ── LoRA fine-tuning ──────────────────────────────────────────────────────────

def lora_cls_brainiac() -> int:
    from monai.networks.nets import ViT
    backbone = ViT(
        in_channels=1, img_size=(96, 96, 96), patch_size=(16, 16, 16),
        hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12, save_attn=True,
    )
    for p in backbone.parameters():
        p.requires_grad_(False)
    _apply_lora(backbone, rank=4)
    return n_trainable(backbone) + n_trainable(_mlp_head(FEAT_DIMS["BrainIAC"]))


def lora_cls_sammed() -> int:
    import medim
    sam = medim.create_model("SAM-Med3D", pretrained=False)
    encoder = sam.image_encoder
    for p in encoder.parameters():
        p.requires_grad_(False)
    _apply_lora(encoder, rank=4)
    return n_trainable(encoder) + n_trainable(_mlp_head(FEAT_DIMS["SAM-Med3D"]))


def lora_cls_3dino() -> int:
    from dinov2.configs import load_and_merge_config_3d
    from dinov2.models import build_model_from_cfg
    cfg = load_and_merge_config_3d("train/vit3d_highres")
    model, _ = build_model_from_cfg(cfg, only_teacher=True)
    feat_dim = model.num_features
    for p in model.parameters():
        p.requires_grad_(False)
    _apply_lora(model, rank=4)
    return n_trainable(model) + n_trainable(_mlp_head(feat_dim))


def _lora_cls_sam2(config_name: str, feat_dim: int) -> int:
    CONFIG_DIR = REPO_ROOT / "local" / "MedSAM2" / "config"
    from hydra.core.global_hydra import GlobalHydra
    from hydra import initialize_config_dir
    from sam2.build_sam import build_sam2
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), job_name="lora_count"):
        model = build_sam2(config_name, ckpt_path=None, device="cpu")
    encoder = model.image_encoder
    for p in encoder.parameters():
        p.requires_grad_(False)
    _apply_lora(encoder, rank=4)
    return n_trainable(encoder) + n_trainable(_mlp_head(feat_dim))


# ── Baseline classification (DenseNet121) ─────────────────────────────────────

def baseline_cls_params() -> int:
    from monai.networks.nets import DenseNet121
    return n_trainable(DenseNet121(spatial_dims=3, in_channels=1, out_channels=1))


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

# ── BrainIAC: UNETR decoder, frozen ViT encoder ───────────────────────────────

def brainiac_seg_params() -> int:
    from monai.networks.nets import UNETR
    unetr = UNETR(
        in_channels=1, out_channels=1, img_size=(96, 96, 96),
        feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12,
        norm_name="instance", res_block=True, dropout_rate=0.0,
    )
    for p in unetr.vit.parameters():
        p.requires_grad_(False)
    return n_trainable(unetr)


# ── 3D-DINO: UNETRHead decoder, frozen encoder ────────────────────────────────

def dino_seg_params() -> int:
    from dinov2.configs import load_and_merge_config_3d
    from dinov2.models import build_model_from_cfg
    from dinov2.eval.segmentation_3d.segmentation_heads import UNETRHead
    cfg = load_and_merge_config_3d("train/vit3d_highres")
    encoder, _ = build_model_from_cfg(cfg, only_teacher=True)
    for p in encoder.parameters():
        p.requires_grad_(False)
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=False)
    seg = UNETRHead(encoder, input_channels=1, image_size=96,
                   num_classes=2, autocast_ctx=autocast_ctx)
    return n_trainable(seg)


# ── SAM-Med3D+LoRA: image encoder LoRA adapters only ─────────────────────────

def sammed_lora_seg_params() -> int:
    import medim
    sam = medim.create_model("SAM-Med3D", pretrained=False)
    for p in sam.parameters():
        p.requires_grad_(False)
    _apply_lora(sam.image_encoder, rank=4)
    return n_trainable(sam)


# ── Baseline segmentation: 3D U-Net from scratch ─────────────────────────────

def baseline_seg_params() -> int:
    from monai.networks.nets import UNet
    return n_trainable(UNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
        num_res_units=2, dropout=0.1,
    ))


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    SEP = "=" * 75

    print(f"\n{SEP}")
    print("  CLASSIFICATION — trainable parameters")
    print(SEP)

    print("\n  Linear probing (MLP head on frozen features, single sequence)")
    for name, dim in FEAT_DIMS.items():
        report(f"    {name}  (feat_dim={dim})", lambda d=dim: linprobe_params(d))
    report("    3D-DINO  (feat_dim probed from model)", linprobe_3dino_params)

    print("\n  LoRA fine-tuning (LoRA adapters rank=4 + MLP head, single sequence)")
    report("    BrainIAC",    lora_cls_brainiac)
    report("    SAM-Med3D",   lora_cls_sammed)
    report("    3D-DINO",     lora_cls_3dino)
    report("    MedSAM2",     lambda: _lora_cls_sam2("sam2.1_hiera_t512.yaml", feat_dim=256))
    report("    MedicalSAM2", lambda: _lora_cls_sam2("sam2_hiera_t.yaml",      feat_dim=256))

    print("\n  Baseline (DenseNet121, single sequence)")
    report("    DenseNet121", baseline_cls_params)

    print(f"\n{SEP}")
    print("  SEGMENTATION — trainable parameters")
    print(SEP)

    print()
    report("  BrainIAC   — UNETR decoder (ViT encoder frozen)",          brainiac_seg_params)
    report("  3D-DINO    — UNETRHead decoder (ViT encoder frozen)",       dino_seg_params)
    report("  SAM-Med3D+LoRA — image encoder LoRA adapters (rank=4)",     sammed_lora_seg_params)
    report("  Baseline   — 3D U-Net from scratch",                        baseline_seg_params)

    print()
