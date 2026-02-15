"""
Evaluation metrics for medical image segmentation.

Clinical validation metrics:
- Dice Coefficient (overlap)
- IoU / Jaccard Index
- Sensitivity / Recall (true positive rate)
- Specificity (true negative rate)
- Precision (positive predictive value)
- Hausdorff Distance (surface distance)
- Average Surface Distance
"""

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt


def dice_coefficient(pred, target, smooth=1e-5):
    """
    Compute Dice coefficient (F1 score for segmentation).

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)

    Args:
        pred: Binary predictions (B, 1, D, H, W) or (D, H, W)
        target: Binary ground truth (same shape as pred)
        smooth: Smoothing factor to avoid division by zero

    Returns:
        dice: Dice coefficient (scalar or per-sample)
    """
    pred = pred.view(-1).float()
    target = target.view(-1).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice.item() if dice.numel() == 1 else dice


def iou_score(pred, target, smooth=1e-5):
    """
    Compute Intersection over Union (Jaccard Index).

    IoU = |X ∩ Y| / |X ∪ Y|

    Args:
        pred: Binary predictions
        target: Binary ground truth
        smooth: Smoothing factor

    Returns:
        iou: IoU score
    """
    pred = pred.view(-1).float()
    target = target.view(-1).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.item() if iou.numel() == 1 else iou


def sensitivity_recall(pred, target, smooth=1e-5):
    """
    Compute Sensitivity (Recall, True Positive Rate).

    Sensitivity = TP / (TP + FN)

    Measures ability to detect positive cases.
    Important for clinical applications (don't miss lesions).

    Args:
        pred: Binary predictions
        target: Binary ground truth
        smooth: Smoothing factor

    Returns:
        sensitivity: Sensitivity score
    """
    pred = pred.view(-1).float()
    target = target.view(-1).float()

    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()

    sensitivity = (tp + smooth) / (tp + fn + smooth)

    return sensitivity.item() if sensitivity.numel() == 1 else sensitivity


def specificity(pred, target, smooth=1e-5):
    """
    Compute Specificity (True Negative Rate).

    Specificity = TN / (TN + FP)

    Measures ability to correctly identify negative cases.

    Args:
        pred: Binary predictions
        target: Binary ground truth
        smooth: Smoothing factor

    Returns:
        specificity: Specificity score
    """
    pred = pred.view(-1).float()
    target = target.view(-1).float()

    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()

    spec = (tn + smooth) / (tn + fp + smooth)

    return spec.item() if spec.numel() == 1 else spec


def precision(pred, target, smooth=1e-5):
    """
    Compute Precision (Positive Predictive Value).

    Precision = TP / (TP + FP)

    Measures accuracy of positive predictions.
    Important to minimize false alarms.

    Args:
        pred: Binary predictions
        target: Binary ground truth
        smooth: Smoothing factor

    Returns:
        precision: Precision score
    """
    pred = pred.view(-1).float()
    target = target.view(-1).float()

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()

    prec = (tp + smooth) / (tp + fp + smooth)

    return prec.item() if prec.numel() == 1 else prec


def hausdorff_distance_95(pred, target):
    """
    Compute 95th percentile Hausdorff Distance.

    Measures maximum surface distance between segmentations.
    95th percentile is more robust to outliers than max HD.

    HD(X, Y) = max(h(X, Y), h(Y, X))
    where h(X, Y) = max_{x in X} min_{y in Y} ||x - y||

    Args:
        pred: Binary predictions (numpy array)
        target: Binary ground truth (numpy array)

    Returns:
        hd95: 95th percentile Hausdorff distance in mm (assumes 1mm spacing)
    """
    # Convert to numpy if needed
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()

    # Squeeze if needed
    pred = np.squeeze(pred)
    target = np.squeeze(target)

    # Check if either mask is empty
    if pred.sum() == 0 or target.sum() == 0:
        return float('inf')

    # Compute distance transforms
    # Distance from each voxel to nearest surface
    dt_pred = distance_transform_edt(~pred.astype(bool))
    dt_target = distance_transform_edt(~target.astype(bool))

    # Get surface voxels
    surface_pred = pred & (dt_pred <= 1)
    surface_target = target & (dt_target <= 1)

    # Compute distances from pred surface to target
    distances_pred_to_target = dt_target[surface_pred]

    # Compute distances from target surface to pred
    distances_target_to_pred = dt_pred[surface_target]

    # Combine all distances
    all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])

    if len(all_distances) == 0:
        return 0.0

    # Return 95th percentile
    hd95 = np.percentile(all_distances, 95)

    return float(hd95)


def average_surface_distance(pred, target):
    """
    Compute Average Surface Distance (ASD).

    Mean distance between surfaces of two segmentations.
    More stable than Hausdorff distance.

    Args:
        pred: Binary predictions (numpy array)
        target: Binary ground truth (numpy array)

    Returns:
        asd: Average surface distance in mm
    """
    # Convert to numpy if needed
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()

    # Squeeze if needed
    pred = np.squeeze(pred)
    target = np.squeeze(target)

    # Check if either mask is empty
    if pred.sum() == 0 or target.sum() == 0:
        return float('inf')

    # Compute distance transforms
    dt_pred = distance_transform_edt(~pred.astype(bool))
    dt_target = distance_transform_edt(~target.astype(bool))

    # Get surface voxels
    surface_pred = pred & (dt_pred <= 1)
    surface_target = target & (dt_target <= 1)

    # Compute distances
    distances_pred_to_target = dt_target[surface_pred]
    distances_target_to_pred = dt_pred[surface_target]

    # Average of all surface distances
    all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])

    if len(all_distances) == 0:
        return 0.0

    asd = np.mean(all_distances)

    return float(asd)


def compute_all_metrics(pred, target, spacing=(1.0, 1.0, 1.0)):
    """
    Compute all segmentation metrics.

    Args:
        pred: Binary predictions (torch.Tensor or numpy.ndarray)
        target: Binary ground truth (same type as pred)
        spacing: Voxel spacing in mm (for distance metrics)

    Returns:
        metrics: Dictionary with all computed metrics
    """
    # Convert to binary if needed
    if torch.is_tensor(pred):
        pred_bin = (pred > 0.5).float()
    else:
        pred_bin = (pred > 0.5).astype(np.float32)

    metrics = {}

    # Overlap metrics (work with torch tensors)
    if not torch.is_tensor(pred_bin):
        pred_tensor = torch.from_numpy(pred_bin)
        target_tensor = torch.from_numpy(target)
    else:
        pred_tensor = pred_bin
        target_tensor = target

    metrics['dice'] = dice_coefficient(pred_tensor, target_tensor)
    metrics['iou'] = iou_score(pred_tensor, target_tensor)
    metrics['sensitivity'] = sensitivity_recall(pred_tensor, target_tensor)
    metrics['specificity'] = specificity(pred_tensor, target_tensor)
    metrics['precision'] = precision(pred_tensor, target_tensor)

    # Distance metrics (work with numpy arrays)
    if torch.is_tensor(pred_bin):
        pred_np = pred_bin.cpu().numpy()
        target_np = target.cpu().numpy()
    else:
        pred_np = pred_bin
        target_np = target

    try:
        metrics['hausdorff_95'] = hausdorff_distance_95(pred_np, target_np)
        metrics['avg_surface_distance'] = average_surface_distance(pred_np, target_np)
    except Exception as e:
        metrics['hausdorff_95'] = float('inf')
        metrics['avg_surface_distance'] = float('inf')

    return metrics


def test_metrics():
    """Test metric functions."""

    print("Testing segmentation metrics...")

    # Create dummy data - perfect prediction
    pred = torch.ones(1, 1, 32, 32, 32)
    target = torch.ones(1, 1, 32, 32, 32)

    metrics = compute_all_metrics(pred, target)

    print("\nPerfect prediction metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    assert metrics['dice'] == 1.0, "Dice should be 1.0 for perfect prediction"
    assert metrics['sensitivity'] == 1.0, "Sensitivity should be 1.0"

    # Test with partial overlap
    pred_partial = torch.zeros(1, 1, 32, 32, 32)
    pred_partial[:, :, :16, :, :] = 1  # Half overlap

    metrics_partial = compute_all_metrics(pred_partial, target)

    print("\nPartial overlap metrics:")
    for name, value in metrics_partial.items():
        print(f"  {name}: {value:.4f}")

    assert 0 < metrics_partial['dice'] < 1, "Dice should be between 0 and 1"

    print("\n✓ All metric tests passed")


if __name__ == "__main__":
    test_metrics()
