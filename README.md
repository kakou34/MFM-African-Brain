# Can Medical Foundation Models Read the African Brain?

This repository benchmarks medical vision foundation models for brain MRI **classification** and **tumour segmentation** on two African datasets: **NigerianBrain** (single-sequence T1) and **BraTS-Africa** (multi-sequence: T1c, T1n, T2-FLAIR, T2w). We evaluate four foundation models — BrainIAC, 3D-DINO, MedSAM2, and MedicalSAM2 — under three adaptation regimes: linear probing, LoRA fine-tuning, and supervised fine-tuning with a decoder head.

---

## Repository Structure

```
africai/
├── 3DINO/                        # 3D-DINO submodule
├── src/
│   ├── feature_extraction/       # Extract frozen features from each model
│   ├── linear_probing/           # Linear probe and DenseNet121 baseline
│   ├── lora/                     # LoRA fine-tuning for classification
│   ├── segmentation/             # Segmentation experiments (all models)
│   └── analysis/                 # Result tables and figures
├── csvs/
│   ├── input/                    # nigerianbrain.csv, bratsafrica.csv
│   ├── output/                   # Extracted feature CSVs
│   ├── results/                  # Classification results
│   ├── results_lora/             # LoRA classification results
│   └── segmentation/             # Per-model Dice scores
├── tables/                       # Generated markdown result tables
├── figures/                      # Generated figures
├── requirements.txt
└── slurm_*.sh                    # SLURM job scripts (one per experiment)
```

---

## Installation

**Requirements:** Python 3.13, CUDA 12.8+

```bash
git clone --recurse-submodules <repo_link>
cd repo
pip install -r requirements.txt
```

If you cloned without `--recurse-submodules`, initialise the 3D-DINO submodule manually:

```bash
git submodule update --init --recursive
```

### Model Checkpoints

Download the checkpoints and place them in a directory of your choice (referred to as `$CKPT` below):

| Model | Checkpoint file |
|-------|----------------|
| [BrainIAC](https://github.com/AIM-KannLab/BrainIAC) | `BrainIAC.ckpt` |
| [3D-DINO](https://github.com/AICONSlab/3DINO) | `3dino_vit_weights.pth` |
| [MedSAM2](https://github.com/bowang-lab/MedSAM2) | `MedSAM2_latest.pt` |
| [MedicalSAM2](https://github.com/ImprintLab/Medical-SAM2) | `MedicalSAM2_pretrain.pth` |

---

## Data Preparation

Each dataset is described by a CSV file. Place them under `csvs/input/`.

**Single-sequence dataset** (e.g. NigerianBrain — `nigerianbrain.csv`):

| column | description |
|--------|-------------|
| `id` | subject identifier |
| `cohort` | cohort name |
| `img` | relative path to NIfTI file from `root_dir` |
| `label` | binary label (0/1) |

**Multi-sequence dataset** (e.g. BraTS-Africa — `bratsafrica.csv`):

| column | description |
|--------|-------------|
| `id` | row identifier (`{subject_id}_{seq}`) |
| `subject_id` | subject identifier |
| `seq` | MRI sequence (`t1c`, `t1n`, `t2f`, `t2w`) |
| `cohort` | cohort name |
| `img` | relative path to NIfTI file from `root_dir` |
| `label` | binary label (0/1) |

Set `$ROOT` to the directory that contains your data (e.g. `/path/to/data`).

---

## Experiments

All experiments use stratified 5-fold splits (no subject leakage). Evaluation metrics are **balanced accuracy** (classification) and **Dice coefficient** (segmentation).

---

### Classification

#### Baseline — DenseNet121 trained from scratch

```bash
# NigerianBrain
python src/linear_probing/baseline.py \
    --csv_path csvs/input/nigerianbrain.csv \
    --root_dir $ROOT \
    --output_csv csvs/results/nigerianbrain_baseline.csv \
    --mode mean \
    --epochs 50 --lr 1e-3 --batch_size 2 --num_workers 8

# BraTS-Africa (mode: mean | concat | t2f)
python src/linear_probing/baseline.py \
    --csv_path csvs/input/bratsafrica.csv \
    --root_dir $ROOT \
    --output_csv csvs/results/bratsafrica_baseline_t2f.csv \
    --mode t2f \
    --epochs 50 --lr 1e-3 --batch_size 2 --num_workers 8
```

`--mode` options:
- `mean` — average logits across all sequences
- `concat` — stack sequences as channels
- `t2f` / `t1c` / `t1n` / `t2w` — single sequence

---

#### Linear Probing (frozen features + MLP head)

**Step 1 — Extract features:**

```bash

# BrainIAC
python src/feature_extraction/brainiac.py \
    --csv_path csvs/input/nigerianbrain.csv \
    --root_dir $ROOT \
    --save_path csvs/output/nigerianbrain/nigerianbrain_brainiac.csv \
    --checkpoint_path $CKPT/BrainIAC.ckpt

# 3D-DINO
python src/feature_extraction/3dino.py \
    --csv_path csvs/input/nigerianbrain.csv \
    --root_dir $ROOT \
    --save_path csvs/output/nigerianbrain/nigerianbrain_3dino.csv \
    --checkpoint_path $CKPT/3dino_vit_weights.pth

# MedSAM2
python src/feature_extraction/medsam2.py \
    --csv_path csvs/input/nigerianbrain.csv \
    --root_dir $ROOT \
    --save_path csvs/output/nigerianbrain/nigerianbrain_medsam2.csv \
    --checkpoint_path $CKPT/MedSAM2_latest.pt \
    --config_path sam2.1_hiera_t512.yaml

# MedicalSAM2
python src/feature_extraction/medicalsam2.py \
    --csv_path csvs/input/nigerianbrain.csv \
    --root_dir $ROOT \
    --save_path csvs/output/nigerianbrain/nigerianbrain_medicalsam2.csv \
    --checkpoint_path $CKPT/MedicalSAM2_pretrain.pth \
    --config_path sam2_hiera_t.yaml
```

Repeat with `bratsafrica.csv` and save to `csvs/output/brats/`.

**Step 2 — Train linear probe:**

```bash
# NigerianBrain (single sequence → mode=mean)
python src/linear_probing/classify_disease.py \
    --features_csv csvs/output/nigerianbrain/nigerianbrain_sammed.csv \
    --output_csv   csvs/results/nigerianbrain_sammed.csv \
    --mode mean

# BraTS-Africa (run for each mode)
for MODE in mean concat t2f; do
    python src/linear_probing/classify_disease.py \
        --features_csv csvs/output/brats/bratsafrica_sammed.csv \
        --output_csv   csvs/results/bratsafrica_sammed_${MODE}.csv \
        --mode $MODE
done
```

Replace `sammed` with `brainiac`, `3dino`, `medsam2`, or `medicalsam2` for other models.

---

#### LoRA Fine-tuning (classification)

LoRA adapters (rank 4) are added to the attention layers of the frozen image encoder, then a linear classifier is trained on top.

```bash

# NigerianBrain — BrainIAC
python src/lora/lora.py \
    --model brainiac \
    --csv_path csvs/input/nigerianbrain.csv \
    --root_dir $ROOT \
    --output_csv csvs/results_lora/lora_nigerianbrain_brainiac.csv \
    --checkpoint_path $CKPT/BrainIAC.ckpt

# NigerianBrain — 3D-DINO
python src/lora/lora.py \
    --model 3dino \
    --csv_path csvs/input/nigerianbrain.csv \
    --root_dir $ROOT \
    --output_csv csvs/results_lora/lora_nigerianbrain_3dino.csv \
    --checkpoint_path $CKPT/3dino_vit_weights.pth

# NigerianBrain — MedSAM2
python src/lora/lora.py \
    --model medsam2 \
    --csv_path csvs/input/nigerianbrain.csv \
    --root_dir $ROOT \
    --output_csv csvs/results_lora/lora_nigerianbrain_medsam2.csv \
    --checkpoint_path $CKPT/MedSAM2_latest.pt \
    --config_path sam2.1_hiera_t512.yaml

# NigerianBrain — MedicalSAM2
python src/lora/lora.py \
    --model medicalsam2 \
    --csv_path csvs/input/nigerianbrain.csv \
    --root_dir $ROOT \
    --output_csv csvs/results_lora/lora_nigerianbrain_medicalsam2.csv \
    --checkpoint_path $CKPT/MedicalSAM2_pretrain.pth \
    --config_path sam2_hiera_t.yaml
```

For BraTS-Africa, add `--seq t2f` to all commands above and replace `nigerianbrain` paths with `bratsafrica`.

Key optional arguments: `--lora_rank 4`, `--lr 1e-4`, `--epochs 50`, `--batch_size 4`, `--num_splits 5`.

---

### Segmentation

All segmentation experiments run on BraTS-Africa using the T2-FLAIR (`t2f`) sequence by default. Results (per-split Dice scores) are written as CSVs to `--out_dir`.

---

#### Baseline — 3D U-Net trained from scratch

```bash
python src/segmentation/baseline.py \
    --csv_path csvs/input/bratsafrica.csv \
    --root_dir $ROOT \
    --out_dir  csvs/segmentation/baseline \
    --seq t2f \
    --epochs 50 --batch_size 2 --lr 1e-4 --num_workers 8
```

Run once per sequence (`t1c`, `t1n`, `t2f`, `t2w`) to benchmark all modalities.


---

#### BrainIAC with UNETR Decoder

The BrainIAC ViT-B encoder is initialised from a SimCLR checkpoint and paired with a UNETR decoder head. The encoder is frozen by default; pass `--train_encoder` for full fine-tuning.

```bash
python src/segmentation/segment_brainiac.py \
    --csv_path        csvs/input/bratsafrica.csv \
    --root_dir        $ROOT \
    --out_dir         csvs/segmentation/brainiac \
    --checkpoint_path $CKPT/BrainIAC.ckpt \
    --seq t2f \
    --epochs 50 --batch_size 2 --lr 1e-4 --num_workers 8
```

---

#### 3D-DINO with UNETR Decoder

The 3D-DINO encoder is paired with a UNETR segmentation head. Supports `UNETR` (default) or `Linear` decoder via `--segmentation_head`. The encoder is frozen by default; pass `--train_encoder` for full fine-tuning.

```bash
python src/segmentation/segment_3dino.py \
    --csv_path        csvs/input/bratsafrica.csv \
    --root_dir        $ROOT \
    --out_dir         csvs/segmentation/3dino \
    --checkpoint_path $CKPT/3dino_vit_weights.pth \
    --seq t2f \
    --segmentation_head UNETR \
    --epochs 50 --batch_size 2 --lr 1e-4 --num_workers 8
```

---


## Generating Result Tables

After running experiments, generate summary markdown tables:

```bash
# Classification results
python src/analysis/generate_cls_table.py

# Segmentation results
python src/analysis/generate_seg_table.py
```

Tables are written to `tables/classification_results.md` and `tables/segmentation_results.md`.


