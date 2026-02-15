"""
3D U-Net implementation for airway segmentation.

Based on: Çiçek et al., "3D U-Net: Learning Dense Volumetric Segmentation 
from Sparse Annotation" MICCAI 2016

Features:
- Skip connections for feature propagation
- Batch normalization for training stability
- Deep supervision (optional)
- Memory-efficient implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """Two consecutive 3D convolutions with BN and ReLU."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downsampling with maxpool then double conv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upsampling with transpose conv and double conv with skip connection."""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: Input from previous layer (lower resolution)
            x2: Skip connection from encoder (higher resolution)
        """
        x1 = self.up(x1)

        # Handle size mismatch due to padding
        diff_d = x2.size()[2] - x1.size()[2]
        diff_h = x2.size()[3] - x1.size()[3]
        diff_w = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2,
                        diff_d // 2, diff_d - diff_d // 2])

        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric segmentation.

    Args:
        in_channels: Number of input channels (1 for CT)
        out_channels: Number of output channels (1 for binary segmentation)
        features: List of feature dimensions at each level [32, 64, 128, 256, 512]
        trilinear: Use trilinear upsampling (faster) vs transpose conv (learnable)
        deep_supervision: Enable auxiliary outputs at each decoder level

    Input shape: (B, C, D, H, W) where D=depth, H=height, W=width
    Output shape: (B, 1, D, H, W) with sigmoid activation
    """

    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256, 512],
                 trilinear=True, deep_supervision=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deep_supervision = deep_supervision

        # Initial convolution
        self.inc = DoubleConv3D(in_channels, features[0])

        # Encoder (downsampling path)
        self.down1 = Down3D(features[0], features[1])
        self.down2 = Down3D(features[1], features[2])
        self.down3 = Down3D(features[2], features[3])

        # Bottleneck
        factor = 2 if trilinear else 1
        self.down4 = Down3D(features[3], features[4] // factor)

        # Decoder (upsampling path)
        self.up1 = Up3D(features[4], features[3] // factor, trilinear)
        self.up2 = Up3D(features[3], features[2] // factor, trilinear)
        self.up3 = Up3D(features[2], features[1] // factor, trilinear)
        self.up4 = Up3D(features[1], features[0], trilinear)

        # Output convolution
        self.outc = nn.Conv3d(features[0], out_channels, kernel_size=1)

        # Deep supervision auxiliary outputs
        if deep_supervision:
            self.dsv1 = nn.Conv3d(features[3] // factor, out_channels, kernel_size=1)
            self.dsv2 = nn.Conv3d(features[2] // factor, out_channels, kernel_size=1)
            self.dsv3 = nn.Conv3d(features[1] // factor, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through 3D U-Net.

        Args:
            x: Input tensor (B, 1, D, H, W)

        Returns:
            If deep_supervision=False: logits (B, 1, D, H, W)
            If deep_supervision=True: dict with main output and auxiliary outputs
        """
        # Encoder
        x1 = self.inc(x)      # features[0]
        x2 = self.down1(x1)   # features[1]
        x3 = self.down2(x2)   # features[2]
        x4 = self.down3(x3)   # features[3]
        x5 = self.down4(x4)   # features[4] (bottleneck)

        # Decoder with skip connections
        x = self.up1(x5, x4)  # features[3] // 2
        if self.deep_supervision:
            dsv1 = self.dsv1(x)

        x = self.up2(x, x3)   # features[2] // 2
        if self.deep_supervision:
            dsv2 = self.dsv2(x)

        x = self.up3(x, x2)   # features[1] // 2
        if self.deep_supervision:
            dsv3 = self.dsv3(x)

        x = self.up4(x, x1)   # features[0]

        # Output
        logits = self.outc(x)

        if self.deep_supervision and self.training:
            # Return main output + auxiliary outputs for loss computation
            return {
                'output': logits,
                'dsv1': F.interpolate(dsv1, size=logits.shape[2:], mode='trilinear', align_corners=True),
                'dsv2': F.interpolate(dsv2, size=logits.shape[2:], mode='trilinear', align_corners=True),
                'dsv3': F.interpolate(dsv3, size=logits.shape[2:], mode='trilinear', align_corners=True)
            }
        else:
            return logits

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_unet3d():
    """Test 3D U-Net with dummy input."""

    # Create model
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        features=[32, 64, 128, 256, 512],
        deep_supervision=False
    )

    print(f"Model parameters: {model.count_parameters():,}")

    # Dummy input (batch=2, channels=1, depth=64, height=128, width=128)
    x = torch.randn(2, 1, 64, 128, 128)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, "Output shape mismatch"

    print("✓ UNet3D test passed")

    # Test with deep supervision
    model_ds = UNet3D(in_channels=1, out_channels=1, deep_supervision=True)
    model_ds.train()

    with torch.no_grad():
        outputs = model_ds(x)

    print(f"\nDeep supervision outputs:")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")

    print("✓ Deep supervision test passed")


if __name__ == "__main__":
    test_unet3d()
