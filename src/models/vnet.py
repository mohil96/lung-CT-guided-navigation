"""
V-Net implementation for nodule detection.

Based on: Milletari et al., "V-Net: Fully Convolutional Neural Networks 
for Volumetric Medical Image Segmentation" 3DV 2016

Features:
- Residual connections
- PReLU activation
- Dice loss compatibility
- Compact architecture for patch-based processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with residual connection."""

    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()

        layers = []
        for i in range(num_convs):
            if i == 0:
                layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.PReLU())

        self.conv = nn.Sequential(*layers)

        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.residual(x)


class DownBlock(nn.Module):
    """Downsampling block with strided convolution."""

    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()

        self.down = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(out_channels, out_channels, num_convs)

    def forward(self, x):
        x = self.down(x)
        return self.conv_block(x)


class UpBlock(nn.Module):
    """Upsampling block with transpose convolution and skip connection."""

    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(out_channels * 2, out_channels, num_convs)

    def forward(self, x, skip):
        """
        Args:
            x: Input from previous layer
            skip: Skip connection from encoder
        """
        x = self.up(x)

        # Handle size mismatch
        diff_d = skip.size()[2] - x.size()[2]
        diff_h = skip.size()[3] - x.size()[3]
        diff_w = skip.size()[4] - x.size()[4]

        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                      diff_h // 2, diff_h - diff_h // 2,
                      diff_d // 2, diff_d - diff_d // 2])

        # Concatenate skip connection
        x = torch.cat([skip, x], dim=1)
        return self.conv_block(x)


class VNet(nn.Module):
    """
    V-Net for volumetric medical image segmentation.

    Optimized for patch-based nodule detection with compact architecture.

    Args:
        in_channels: Number of input channels (1 for CT)
        out_channels: Number of output channels (1 for binary segmentation)
        features: Base number of features (default: 16)

    Input shape: (B, 1, D, H, W) - typically 64x64x64 patches
    Output shape: (B, 1, D, H, W)
    """

    def __init__(self, in_channels=1, out_channels=1, features=16):
        super().__init__()

        # Initial convolution
        self.input_conv = ConvBlock(in_channels, features, num_convs=1)

        # Encoder path
        self.down1 = DownBlock(features, features * 2, num_convs=2)
        self.down2 = DownBlock(features * 2, features * 4, num_convs=3)
        self.down3 = DownBlock(features * 4, features * 8, num_convs=3)

        # Bottleneck
        self.bottleneck = DownBlock(features * 8, features * 16, num_convs=3)

        # Decoder path
        self.up3 = UpBlock(features * 16, features * 8, num_convs=3)
        self.up2 = UpBlock(features * 8, features * 4, num_convs=3)
        self.up1 = UpBlock(features * 4, features * 2, num_convs=2)
        self.up0 = UpBlock(features * 2, features, num_convs=1)

        # Output
        self.output_conv = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through V-Net.

        Args:
            x: Input tensor (B, 1, D, H, W)

        Returns:
            logits: Output tensor (B, 1, D, H, W)
        """
        # Initial convolution
        x0 = self.input_conv(x)

        # Encoder with skip connections
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        # Bottleneck
        x4 = self.bottleneck(x3)

        # Decoder with skip connections
        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)

        # Output
        logits = self.output_conv(x)

        return logits

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_vnet():
    """Test V-Net with dummy input."""

    # Create model
    model = VNet(
        in_channels=1,
        out_channels=1,
        features=16
    )

    print(f"Model parameters: {model.count_parameters():,}")

    # Dummy input (batch=2, channels=1, 64x64x64 patch)
    x = torch.randn(2, 1, 64, 64, 64)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, "Output shape mismatch"

    print("✓ VNet test passed")

    # Test memory usage with different patch sizes
    print("\nMemory test with different patch sizes:")
    patch_sizes = [32, 48, 64, 96]

    for size in patch_sizes:
        x_test = torch.randn(1, 1, size, size, size)
        with torch.no_grad():
            out = model(x_test)
        print(f"  Patch {size}³: Input {x_test.numel():,} -> Output {out.numel():,} elements")

    print("✓ Memory test passed")


if __name__ == "__main__":
    test_vnet()
