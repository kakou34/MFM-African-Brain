"""
Violin plots of per-scan Dice for MedSAM2 and MedicalSAM2 on BraTS Africa and EGD,
for 'combined' and 'T2-FLAIR' sequences, with individual subject scores overlaid.
One subplot per model; violins within each subplot are grouped by sequence and
coloured by dataset.

Usage:
    python plot_zeroshot_dice.py [--out figures/zeroshot_dice.pdf]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

RESULTS_ROOT = Path("path/to/DATA")

DATASETS = [
    ("BraTS-Africa", "BraTS-Africa"),
    ("EGD-150",          "EGD"),
]

MODELS = [
    ("MedSAM2",     "medsam2"),
    ("Medical-SAM2", "medicalsam2"),
]

SEQS       = ["combined", "t2f"]
SEQ_LABELS = ["Combined", "FLAIR"]

# Okabe-Ito palette — perceptually uniform and print-safe
COLORS = ["#66c2a5", "#fc8d62"]   # one colour per dataset

BOX_W   = 0.22
GROUP_W = 0.50


def load_data(dataset_dir, model_dir):
    path = RESULTS_ROOT / dataset_dir / "Results" / model_dir / "per_scan_metrics.csv"
    df = pd.read_csv(path)
    return {seq: df.loc[df["seq"] == seq, "dice"].values for seq in SEQS}


def draw_panel(ax, model_data, offsets, group_x, ylabel=False):
    """Draw one panel for a single model; violins grouped by sequence, coloured by dataset."""
    rng = np.random.default_rng(42)

    for i, (ds_label, _) in enumerate(DATASETS):
        color = COLORS[i]
        positions = group_x + offsets[i]
        vals_per_seq = [model_data[ds_label][seq] for seq in SEQS]

        parts = ax.violinplot(
            vals_per_seq,
            positions=positions,
            widths=BOX_W * 0.85,
            showmedians=True,
            showextrema=True,
        )

        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.6)
            body.set_edgecolor(color)
            body.set_linewidth(0.8)
            body.set_zorder(5)

        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)
        parts["cmedians"].set_zorder(4)
        for key in ("cmins", "cmaxes", "cbars"):
            parts[key].set_color(color)
            parts[key].set_linewidth(0.8)
            parts[key].set_zorder(5)

        jitter_w = BOX_W * 0.30
        for pos, vals in zip(positions, vals_per_seq):
            jitter = rng.uniform(-jitter_w, jitter_w, size=len(vals))
            ax.scatter(pos + jitter, vals, s=5, color=color, alpha=0.95,
                       linewidths=0, zorder=3)

    if ylabel:
        ax.set_ylabel("Dice score", fontsize=14)
    ax.set_ylim(-0.02, 1.08)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(group_x)
    ax.set_xticklabels(SEQ_LABELS, fontsize=14)
    ax.tick_params(axis="x", length=0)
    ax.spines[["top", "right"]].set_visible(False)


def main(args):
    # all_data[ds_label][model_label][seq] = np.ndarray of dice scores
    all_data = {ds_label: {} for ds_label, _ in DATASETS}
    for ds_label, ds_dir in DATASETS:
        for model_label, model_dir in MODELS:
            all_data[ds_label][model_label] = load_data(ds_dir, model_dir)

    # Restructure to model_data[model_label][ds_label][seq]
    model_data = {
        model_label: {ds_label: all_data[ds_label][model_label] for ds_label, _ in DATASETS}
        for model_label, _ in MODELS
    }

    group_x = np.arange(len(SEQS), dtype=float)
    offsets = np.linspace(-GROUP_W / 2 + BOX_W / 2,
                           GROUP_W / 2 - BOX_W / 2,
                           len(DATASETS))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for ax, (model_label, _) in zip(axes, MODELS):
        draw_panel(ax, model_data[model_label], offsets, group_x, ylabel=(ax is axes[0]))
        ax.set_title(model_label, fontsize=13, fontweight="bold")

    legend_handles = [
        Patch(facecolor=COLORS[i], alpha=0.75, label=ds_label)
        for i, (ds_label, _) in enumerate(DATASETS)
    ]
    fig.legend(handles=legend_handles, fontsize=14, framealpha=0.9,
               loc="lower center", ncol=len(DATASETS), bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {out}  ({out.with_suffix('.png')})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures/zeroshot_dice_violin.pdf")
    main(parser.parse_args())
