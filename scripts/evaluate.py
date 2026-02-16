"""Evaluate segmentation predictions against ground-truth masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from src.models.metrics import dice_coefficient, hausdorff_distance_95, iou_score, sensitivity_recall


def load_mask(path: Path) -> np.ndarray:
    arr = nib.load(str(path)).get_fdata()
    return (arr > 0.5).astype(np.uint8)


def evaluate_case(pred_path: Path, gt_path: Path) -> dict:
    pred = load_mask(pred_path)
    gt = load_mask(gt_path)

    metrics = {
        "dice": float(dice_coefficient(np.array(pred), np.array(gt))),
        "iou": float(iou_score(np.array(pred), np.array(gt))),
        "sensitivity": float(sensitivity_recall(np.array(pred), np.array(gt))),
        "hd95": float(hausdorff_distance_95(pred, gt)),
    }
    return metrics


def main(args: argparse.Namespace) -> None:
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    pred_files = sorted(pred_dir.glob(args.pred_glob))
    if not pred_files:
        raise ValueError(f"No prediction files matching '{args.pred_glob}' in {pred_dir}")

    per_case = {}
    for pred_path in pred_files:
        gt_path = gt_dir / pred_path.name.replace(args.pred_suffix, args.gt_suffix)
        if not gt_path.exists():
            print(f"Skipping {pred_path.name}: missing GT {gt_path.name}")
            continue
        per_case[pred_path.name] = evaluate_case(pred_path, gt_path)

    if not per_case:
        raise ValueError("No matched prediction/ground-truth pairs found")

    keys = ["dice", "iou", "sensitivity", "hd95"]
    summary = {k: float(np.mean([m[k] for m in per_case.values()])) for k in keys}

    report = {"num_cases": len(per_case), "summary": summary, "per_case": per_case}
    print(json.dumps(report["summary"], indent=2))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3D segmentation predictions")
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--pred_glob", type=str, default="*airway_segmentation.nii.gz")
    parser.add_argument("--pred_suffix", type=str, default="airway_segmentation.nii.gz")
    parser.add_argument("--gt_suffix", type=str, default="_airway.nii.gz")
    parser.add_argument("--output_json", type=str, default="")

    main(parser.parse_args())
