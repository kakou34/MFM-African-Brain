"""
Generate the classification results table (balanced accuracy mean ± std)
for inclusion in the paper. Reads all results CSVs and produces a markdown file.

Usage:
    python generate_cls_table.py [--out tables/classification_results.md]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("csvs/results")
LORA    = Path("csvs/results_lora")


def load(path) -> str:
    """Return 'mean ± std' for bal_acc over splits, or '-' if file is absent."""
    if path is None or not path.exists():
        return "-"
    vals = pd.read_csv(path)["bal_acc"].values
    return f"{vals.mean():.3f} ± {vals.std():.3f}"


# (display name, key used in filenames, is_lora)
MODELS = [
    ("Baseline (DenseNet121)",  "baseline",    False),
    ("SAM-Med3D",               "sammed",      False),
    ("BrainIAC",                "brainiac",    False),
    ("3D-DINO",                 "3dino",       False),
    ("MedSAM2",                 "medsam2",     False),
    ("MedicalSAM2",             "medicalsam2", False),
    ("SAM-Med3D + LoRA",        "sammed",      True),
    ("BrainIAC + LoRA",         "brainiac",    True),
    ("3D-DINO + LoRA",          "3dino",       True),
]


def build_rows() -> list[tuple]:
    rows = []
    for display, key, is_lora in MODELS:
        if is_lora:
            nb          = load(LORA / f"lora_nigerianbrain_{key}.csv")
            brats_mean   = "-"   # LoRA ran on t2f only for BraTS-Africa
            brats_concat = "-"
            brats_t2f   = load(LORA / f"lora_bratsafrica_{key}.csv")
        else:
            nb          = load(RESULTS / f"nigerianbrain_{key}.csv")
            brats_mean   = load(RESULTS / f"bratsafrica_{key}_mean.csv")
            brats_concat = load(RESULTS / f"bratsafrica_{key}_concat.csv")
            brats_t2f   = load(RESULTS / f"bratsafrica_{key}_t2f.csv")

        rows.append((display, nb, brats_mean, brats_concat, brats_t2f))
    return rows


def render_markdown(rows: list[tuple]) -> str:
    col_headers = [
        "Model",
        "NigerianBrain\n(Dementia vs Healthy)",
        "BraTS-Africa\n(Mean)",
        "BraTS-Africa\n(Concat)",
        "BraTS-Africa\n(T2-FLAIR)",
    ]

    header = "| " + " | ".join(col_headers) + " |"
    sep    = "|" + "|".join([":-----------:" if i > 0 else " Model " for i in range(len(col_headers))]) + "|"
    sep    = "| :--- | :---: | :---: | :---: | :---: |"

    lines = [
        "## Classification Results — Balanced Accuracy (mean ± std over 5 splits)",
        "",
        "| Model | NigerianBrain (Dementia vs Healthy) | BraTS-Africa (Mean) | BraTS-Africa (Concat) | BraTS-Africa (T2-FLAIR) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for i, (name, nb, bm, bc, bt) in enumerate(rows):
        if i == 6:  # separator before LoRA block
            lines.append("| | | | | |")
        lines.append(f"| {name} | {nb} | {bm} | {bc} | {bt} |")

    return "\n".join(lines) + "\n"


def main(args):
    rows = build_rows()

    # Print a plain-text preview
    print(f"{'Model':<28} {'NigerianBrain':<20} {'BraTS Mean':<20} {'BraTS Concat':<20} {'BraTS t2f':<20}")
    print("-" * 108)
    for i, (name, nb, bm, bc, bt) in enumerate(rows):
        if i == 6:
            print()
        print(f"{name:<28} {nb:<20} {bm:<20} {bc:<20} {bt:<20}")

    md = render_markdown(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"\nMarkdown table saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tables/classification_results.md",
                        help="Output path for the markdown file")
    main(parser.parse_args())
