"""
Loss functions for medical image segmentation.

Includes:
- Dice Loss (soft Dice coefficient)
- Focal Loss (handles class imbalance)
- Tversky Loss (controls FP/FN balance)
- Combined Loss (Dice + BCE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.

    Directly optimizes Dice coefficient (overlap metric).
    Good for imbalanced datasets where positive class is rare.

    Loss = 1 - Dice = 1 - (2 * |X ∩ Y|) / (|X| + |Y|)
    """

    def __init__(self, smooth=1e-5, apply_sigmoid=True):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
            apply_sigmoid: Apply sigmoid to predictions (set False if using BCEWithLogitsLoss)
        """
        super().__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid

    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (B, 1, D, H, W) - logits or probabilities
            target: Ground truth (B, 1, D, H, W) - binary 0/1

        Returns:
            loss: Scalar Dice loss
        """
        if self.apply_sigmoid:
            pred = torch.sigmoid(pred)

        # Flatten spatial dimensions
        pred = pred.view(-1)
        target = target.view(-1)

        # Compute Dice coefficient
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focuses training on hard examples by down-weighting easy examples.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where p_t is the model's estimated probability for the correct class.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor for positive class (0-1)
            gamma: Focusing parameter (0 = standard CE, higher = more focus on hard examples)
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (B, 1, D, H, W) - logits
            target: Ground truth (B, 1, D, H, W) - binary 0/1
        """
        # Apply sigmoid
        p = torch.sigmoid(pred)

        # Compute BCE loss
        bce_loss = F.binary_cross_entropy(p, target, reduction='none')

        # Compute p_t
        p_t = p * target + (1 - p) * (1 - target)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)

        # Compute focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss.

    Allows controlling the balance between false positives and false negatives.

    TL = 1 - (TP) / (TP + α*FP + β*FN)

    α=β=0.5: Dice loss
    α>β: Penalizes FP more (higher precision)
    β>α: Penalizes FN more (higher recall)
    """

    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-5, apply_sigmoid=True):
        """
        Args:
            alpha: Weight for false positives (0-1)
            beta: Weight for false negatives (0-1), typically alpha + beta = 1
            smooth: Smoothing factor
            apply_sigmoid: Apply sigmoid to predictions
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid

    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (B, 1, D, H, W)
            target: Ground truth (B, 1, D, H, W)
        """
        if self.apply_sigmoid:
            pred = torch.sigmoid(pred)

        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)

        # Calculate components
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()

        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        return 1 - tversky


class CombinedLoss(nn.Module):
    """
    Combined Dice + BCE loss.

    Combines the benefits of both losses:
    - Dice: Focuses on overlap, handles imbalance
    - BCE: Provides stable gradients, pixel-level supervision

    Loss = λ * Dice + (1-λ) * BCE
    """

    def __init__(self, dice_weight=0.5, smooth=1e-5):
        """
        Args:
            dice_weight: Weight for Dice loss (0-1), BCE weight = 1 - dice_weight
            smooth: Smoothing factor for Dice
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = 1 - dice_weight

        self.dice = DiceLoss(smooth=smooth, apply_sigmoid=False)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (B, 1, D, H, W) - logits
            target: Ground truth (B, 1, D, H, W) - binary 0/1
        """
        dice_loss = self.dice(pred, target)
        bce_loss = self.bce(pred, target)

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


def test_losses():
    """Test loss functions."""

    # Create dummy data
    batch_size = 2
    pred = torch.randn(batch_size, 1, 32, 32, 32)  # Logits
    target = torch.randint(0, 2, (batch_size, 1, 32, 32, 32)).float()

    print("Testing loss functions with dummy data...")
    print(f"Pred shape: {pred.shape}, Target shape: {target.shape}")
    print()

    # Test Dice Loss
    dice_loss = DiceLoss()
    loss = dice_loss(pred, target)
    print(f"Dice Loss: {loss.item():.4f}")
    assert 0 <= loss.item() <= 1, "Dice loss should be in [0, 1]"

    # Test Focal Loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal_loss(pred, target)
    print(f"Focal Loss: {loss.item():.4f}")
    assert loss.item() >= 0, "Focal loss should be non-negative"

    # Test Tversky Loss
    tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)  # Favor recall
    loss = tversky_loss(pred, target)
    print(f"Tversky Loss: {loss.item():.4f}")
    assert 0 <= loss.item() <= 1, "Tversky loss should be in [0, 1]"

    # Test Combined Loss
    combined_loss = CombinedLoss(dice_weight=0.5)
    loss = combined_loss(pred, target)
    print(f"Combined Loss: {loss.item():.4f}")
    assert loss.item() >= 0, "Combined loss should be non-negative"

    print()
    print("✓ All loss tests passed")

    # Test gradient flow
    print("\nTesting gradient flow...")
    pred.requires_grad = True
    loss = combined_loss(pred, target)
    loss.backward()
    print(f"Gradient shape: {pred.grad.shape}")
    print(f"Gradient mean: {pred.grad.mean().item():.6f}")
    print("✓ Gradient flow test passed")


if __name__ == "__main__":
    test_losses()
