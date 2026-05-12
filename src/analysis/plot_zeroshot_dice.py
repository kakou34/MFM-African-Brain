"""
Box plots of per-scan Dice for zero-shot segmentation models across all sequences,
with individual subject scores overlaid as jittered points.
Models: SAM-Med3D (10 pts), MedSAM2, MedicalSAM2.

Usage:
    python plot_zeroshot_dice.py [--out figures/zeroshot_dice.pdf]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

SEG = Path("csvs/segmentation")

MODELS = [
    ("SAM-Med3D", SEG / "sammed_10pts/per_scan_metrics.csv"),
    ("MedSAM2",            SEG / "medsam2/per_scan_metrics.csv"),
    ("MedicalSAM2",        SEG / "medicalsam2/per_scan_metrics.csv"),
]

SEQS       = ["t1c", "t1n", "t2f", "t2w"]
SEQ_LABELS = ["T1Gd", "T1w", "T2-FLAIR", "T2w"]

# Okabe-Ito palette — perceptually uniform and print-safe
COLORS = ["#0072B2", "#009E73", "#D55E00"]

BOX_W   = 0.22   # width of each box
GROUP_W = 0.60   # total width reserved per sequence group (tighter within group)


def main(args):
    # Load all per-scan dice values: {model_name: {seq: np.ndarray}}
    data = {}
    for name, path in MODELS:
        df = pd.read_csv(path)
        data[name] = {seq: df.loc[df["seq"] == seq, "dice"].values for seq in SEQS}

    n_models = len(MODELS)
    n_seqs   = len(SEQS)
    rng      = np.random.default_rng(42)

    # Centre positions for each sequence group
    group_x  = np.arange(n_seqs, dtype=float)
    offsets  = np.linspace(-GROUP_W / 2 + BOX_W / 2,
                            GROUP_W / 2 - BOX_W / 2,
                            n_models)

    fig, ax = plt.subplots(figsize=(10, 5))

    legend_handles = []

    for i, (name, _) in enumerate(MODELS):
        color = COLORS[i]
        positions = group_x + offsets[i]

        vals_per_seq = [data[name][seq] for seq in SEQS]

        bp = ax.boxplot(
            vals_per_seq,
            positions=positions,
            widths=BOX_W * 0.85,
            patch_artist=True,
            notch=False,
            showfliers=False,       # we draw the points ourselves
            medianprops=dict(color="white", linewidth=2),
            boxprops=dict(facecolor=color, alpha=0.55, linewidth=0.8),
            whiskerprops=dict(linewidth=0.8, color=color),
            capprops=dict(linewidth=0.8, color=color),
            zorder=3,
        )

        # Jittered strip of individual points
        jitter_w = BOX_W * 0.30
        for pos, vals in zip(positions, vals_per_seq):
            jitter = rng.uniform(-jitter_w, jitter_w, size=len(vals))
            ax.scatter(
                pos + jitter, vals,
                s=5, color=color, alpha=0.45,
                linewidths=0, zorder=4,
            )

        # One proxy patch for the legend
        from matplotlib.patches import Patch
        legend_handles.append(Patch(facecolor=color, alpha=0.75, label=name))

    ax.set_ylabel("Dice score", fontsize=12)
    ax.set_ylim(-0.02, 1.08)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)

    ax.set_xticks(group_x)
    ax.set_xticklabels(SEQ_LABELS, fontsize=12)
    ax.tick_params(axis="x", length=0)

    ax.spines[["top", "right"]].set_visible(False)

    fig.legend(handles=legend_handles, fontsize=10, framealpha=0.9,
               loc="lower center", ncol=n_models, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {out}  ({out.with_suffix('.png')})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures/zeroshot_dice.pdf")
    main(parser.parse_args())
