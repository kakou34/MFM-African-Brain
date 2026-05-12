import numpy as np
from pathlib import Path
import argparse
import nibabel as nib
import pandas as pd


def seg_path(root: Path, img_relative: str, subject_id: str) -> Path:
    img_dir = (root / img_relative).parent
    return img_dir / f"{subject_id}-seg.nii.gz"


def binseg_path(seg: Path) -> Path:
    # e.g. BraTS-SSA-00002-000-seg.nii.gz -> BraTS-SSA-00002-000-binseg.nii.gz
    return seg.parent / seg.name.replace("-seg.nii.gz", "-binseg.nii.gz")


def binarize_and_save(seg: Path, out: Path):
    img = nib.load(seg)
    binary = (img.get_fdata() > 0).astype(np.uint8)
    nib.save(nib.Nifti1Image(binary, img.affine, img.header), out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binarize tumour segmentation masks")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--root_dir", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    root = Path(args.root_dir)

    seen = set()
    for _, row in df.iterrows():
        seg = seg_path(root, row["img"], row["subject_id"])
        if seg in seen:
            continue
        seen.add(seg)

        out = binseg_path(seg)
        binarize_and_save(seg, out)
        print(f"Saved {out}")
