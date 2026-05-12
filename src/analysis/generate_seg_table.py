"""
Generate the segmentation results table (Dice mean ± std) for the paper.
One column per BraTS-Africa sequence, one row per model.

Usage:
    python generate_seg_table.py [--out tables/segmentation_results.md]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEG = Path("csvs/segmentation")
LORA_SEG = Path("csvs/segmentation/lora_sammed")


def load_dice(path, seq) -> str:
    """Return 'mean ± std' for dice_mean across 5 splits for a given sequence.

    If `path` is a directory, looks for split_metrics_<seq>.csv inside it.
    Otherwise treats `path` as a file and filters by the seq column.
    Returns '-' if the file is absent or the sequence has no data.
    """
    if path is None:
        return "-"
    # Support per-sequence files (baseline)
    resolved = path / f"split_metrics_{seq}.csv" if path.is_dir() else path
    if not resolved.exists():
        return "-"
    df = pd.read_csv(resolved)
    if "seq" in df.columns:
        df = df[df["seq"] == seq]
    if df.empty:
        return "-"
    vals = df.groupby("split")["dice_mean"].mean().values
    return f"{vals.mean():.3f} ± {vals.std():.3f}"


SEQS = ["t1c", "t1n", "t2f", "t2w"]
SEQ_LABELS = {"t1c": "T1c", "t1n": "T1n", "t2f": "T2-FLAIR", "t2w": "T2w"}

# (display name, path, is_separator_before)
# Pass a directory for models with per-sequence files (baseline);
# pass a file path for models with a single multi-sequence CSV.
MODELS = [
    # ── Baseline ─────────────────────────────────────────────────────────────
    ("Baseline (U-Net)",        SEG / "baseline",                            False),
    # ── Zero-shot SAM models ──────────────────────────────────────────────────
    ("SAM-Med3D (1 pt)",        SEG / "sammed/split_metrics.csv",            True),
    ("SAM-Med3D (10 pts)",      SEG / "sammed_10pts/split_metrics.csv",      False),
    ("MedSAM2",                 SEG / "medsam2/split_metrics.csv",           False),
    ("MedicalSAM2",             SEG / "medicalsam2/split_metrics.csv",       False),
    # ── Finetuned models ──────────────────────────────────────────────────────
    ("SAM-Med3D + LoRA",        LORA_SEG / "split_metrics.csv",              True),
    ("BrainIAC",                SEG / "brainiac/split_metrics.csv",          False),
    ("3D-DINO",                 SEG / "3dino/split_metrics.csv",             False),
]


def build_rows() -> list[tuple]:
    rows = []
    for display, path, sep in MODELS:
        cells = [load_dice(path, seq) for seq in SEQS]
        rows.append((display, cells, sep))
    return rows


def render_markdown(rows: list[tuple]) -> str:
    col_labels = [SEQ_LABELS[s] for s in SEQS]
    header = "| Model | " + " | ".join(col_labels) + " |"
    sep    = "| :--- |" + " :---: |" * len(SEQS)

    lines = [
        "## Segmentation Results — Dice (mean ± std over 5 splits)",
        "",
        header,
        sep,
    ]
    for name, cells, add_sep in rows:
        if add_sep:
            lines.append("| | | | | |")
        lines.append("| " + name + " | " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def main(args):
    rows = build_rows()

    # Plain-text preview
    col_w = 22
    header_plain = f"{'Model':<30}" + "".join(f"{SEQ_LABELS[s]:<{col_w}}" for s in SEQS)
    print(header_plain)
    print("-" * (30 + col_w * len(SEQS)))
    for i, (name, cells, sep) in enumerate(rows):
        if sep:
            print()
        print(f"{name:<30}" + "".join(f"{c:<{col_w}}" for c in cells))

    md = render_markdown(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"\nMarkdown table saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tables/segmentation_results.md",
                        help="Output path for the markdown file")
    main(parser.parse_args())
