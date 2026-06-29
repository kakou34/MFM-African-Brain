"""
Generate the classification results table (AUC mean ± std)
for inclusion in the paper. Reads all results CSVs and produces a markdown file.

Usage:
    python generate_cls_table.py [--out tables/classification_results.md]
"""
import argparse
from pathlib import Path

import pandas as pd

RESULTS = Path("csvs/results")
LORA    = Path("csvs/results_lora")


def load(path) -> str:
    """Return 'mean ± std' for AUC over splits, or '-' if file is absent."""
    if path is None or not path.exists():
        return "-"
    vals = pd.read_csv(path)["auc"].values
    return f"{vals.mean():.2f} ± {vals.std():.2f}"


# (display name, key used in filenames, is_lora)
# SAM-Med3D (sammed) excluded
MODELS = [
    ("Baseline (DenseNet121)",  "baseline",    False),
    ("BrainIAC",                "brainiac",    False),
    ("3D-DINO",                 "3dino",       False),
    ("MedSAM2",                 "medsam2",     False),
    ("MedicalSAM2",             "medicalsam2", False),
    ("BrainIAC + LoRA",         "brainiac",    True),
    ("3D-DINO + LoRA",          "3dino",       True),
]


def build_rows() -> list[tuple]:
    rows = []
    for display, key, is_lora in MODELS:
        if is_lora:
            nb     = load(LORA / f"lora_nigerianbrain_{key}.csv")
            oasis4 = load(LORA / f"lora_oasis4_{key}_subsampled.csv")
        else:
            nb     = load(RESULTS / f"nigerianbrain_{key}.csv")
            oasis4 = load(RESULTS / f"oasis4_{key}_subsampled.csv")

        rows.append((display, nb, oasis4))
    return rows


def render_markdown(rows: list[tuple]) -> str:
    lines = [
        "## Classification Results — AUC (mean ± std over 5 splits)",
        "",
        "| Model | NigerianBrain (Dementia vs Healthy) | OASIS-4 |",
        "| :--- | :---: | :---: |",
    ]
    for i, (name, nb, oasis4) in enumerate(rows):
        if i == 5:  # separator before LoRA block
            lines.append("| | | |")
        lines.append(f"| {name} | {nb} | {oasis4} |")

    return "\n".join(lines) + "\n"


def main(args):
    rows = build_rows()

    # Print a plain-text preview
    print(f"{'Model':<28} {'NigerianBrain':<25} {'OASIS-4':<25}")
    print("-" * 80)
    for i, (name, nb, oasis4) in enumerate(rows):
        if i == 5:
            print()
        print(f"{name:<28} {nb:<25} {oasis4:<25}")

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
