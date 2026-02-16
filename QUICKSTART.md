# Quick Start Guide

## Installation

```bash
conda create -n lung_ct python=3.8
conda activate lung_ct
pip install -r requirements.txt
```

## Expected Data Layout

```text
data/
  train/
    case_001_ct.nii.gz
    case_001_airway.nii.gz
    case_001_nodule.nii.gz
  val/
    case_002_ct.nii.gz
    case_002_airway.nii.gz
    case_002_nodule.nii.gz
```

## Train

```bash
python scripts/train_airway.py \
  --config configs/airway_unet_config.yaml \
  --data_dir ./data \
  --output_dir ./experiments/airway

python scripts/train_nodule.py \
  --config configs/nodule_vnet_config.yaml \
  --data_dir ./data \
  --output_dir ./experiments/nodule
```

## Inference

```bash
python scripts/inference.py \
  --input ./data/val/case_002_ct.nii.gz \
  --airway_model ./experiments/airway/checkpoints/checkpoint_best.pth \
  --nodule_model ./experiments/nodule/checkpoints/checkpoint_best.pth \
  --output_dir ./results/case_002
```

## Evaluate

```bash
python scripts/evaluate.py \
  --pred_dir ./results \
  --gt_dir ./data/val \
  --output_json ./results/eval_report.json
```
