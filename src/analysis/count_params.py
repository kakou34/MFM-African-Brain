"""
Count total parameters for each model used in the project.
Models: SAM-Med3D, BrainIAC, 3D-DINO, Med-SAM2, Medical-SAM2
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DINO3_ROOT = REPO_ROOT / "3DINO"
sys.path.insert(0, str(DINO3_ROOT))

CONFIG_DIR = REPO_ROOT / "local" / "MedSAM2" / "config"

import torch
from hydra.core.global_hydra import GlobalHydra


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_sammed():
    import medim
    model = medim.create_model("SAM-Med3D", pretrained=False)
    return count_params(model)


def count_brainiac():
    from monai.networks.nets import ViT
    model = ViT(
        in_channels=1,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        save_attn=True,
    )
    return count_params(model)


def count_3dino():
    from dinov2.configs import load_and_merge_config_3d
    from dinov2.models import build_model_from_cfg
    cfg = load_and_merge_config_3d("train/vit3d_highres")
    model, _ = build_model_from_cfg(cfg, only_teacher=True)
    return count_params(model)


def count_sam2(config_name):
    from sam2.build_sam import build_sam2
    GlobalHydra.instance().clear()
    from hydra import initialize_config_dir
    with initialize_config_dir(config_dir=str(CONFIG_DIR), job_name="sam2_count"):
        model = build_sam2(config_name, ckpt_path=None, device="cpu")
    return count_params(model)


def fmt(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    return f"{n/1e3:.2f}K"


def report(name, fn):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    try:
        total, trainable = fn()
        print(f"  Total params      : {total:>15,}  ({fmt(total)})")
        print(f"  Trainable params  : {trainable:>15,}  ({fmt(trainable)})")
    except Exception as e:
        print(f"  ERROR: {e}")


if __name__ == "__main__":
    report("SAM-Med3D", count_sammed)
    report("BrainIAC (ViT-B backbone)", count_brainiac)
    report("3D-DINO (ViT-L teacher)", count_3dino)
    report("Med-SAM2 (sam2.1_hiera_t512)", lambda: count_sam2("sam2.1_hiera_t512.yaml"))
    report("Medical-SAM2 (sam2_hiera_t)", lambda: count_sam2("sam2_hiera_t.yaml"))
    print()
