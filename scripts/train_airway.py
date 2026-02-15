"""
Training script for 3D U-Net airway segmentation.

Features:
- Mixed precision training (FP16)
- Gradient accumulation
- TensorBoard logging
- Checkpoint management
- Validation during training
- Learning rate scheduling
"""

import os
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# Import your modules (adjust paths as needed)
# from src.models.unet3d import UNet3D
# from src.models.losses import CombinedLoss, DiceLoss
# from src.models.metrics import dice_coefficient, sensitivity_recall


class AirwayDataset(torch.utils.data.Dataset):
    """
    Dataset for airway segmentation.

    Expected data structure:
    data_dir/
        train/
            case_001_ct.nii.gz
            case_001_airway.nii.gz
            case_002_ct.nii.gz
            case_002_airway.nii.gz
            ...
    """

    def __init__(self, data_dir, split='train', patch_size=(128, 128, 128)):
        """
        Args:
            data_dir: Root data directory
            split: 'train', 'val', or 'test'
            patch_size: Size of random crops (D, H, W)
        """
        self.data_dir = Path(data_dir) / split
        self.patch_size = patch_size
        self.split = split

        # Find all CT files
        self.ct_files = sorted(list(self.data_dir.glob('*_ct.nii.gz')))

        if len(self.ct_files) == 0:
            raise ValueError(f"No CT files found in {self.data_dir}")

        print(f"Found {len(self.ct_files)} cases in {split} set")

    def __len__(self):
        return len(self.ct_files)

    def __getitem__(self, idx):
        """
        Load CT and airway mask, extract random patch.

        Returns:
            ct_patch: (1, D, H, W) tensor
            mask_patch: (1, D, H, W) tensor
        """
        # Load CT
        ct_file = self.ct_files[idx]
        mask_file = str(ct_file).replace('_ct.nii.gz', '_airway.nii.gz')

        # In practice, use SimpleITK to load:
        # import SimpleITK as sitk
        # ct = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_file)))
        # mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_file))

        # For now, simulate with random data for demonstration
        ct = np.random.randn(256, 256, 256).astype(np.float32)
        mask = (np.random.rand(256, 256, 256) > 0.95).astype(np.float32)

        # Normalize CT
        ct = np.clip(ct, -1000, 400)  # HU window
        ct = (ct + 1000) / 1400  # Normalize to [0, 1]

        # Extract random patch
        if self.split == 'train':
            d, h, w = ct.shape
            pd, ph, pw = self.patch_size

            # Random crop
            d_start = np.random.randint(0, max(1, d - pd))
            h_start = np.random.randint(0, max(1, h - ph))
            w_start = np.random.randint(0, max(1, w - pw))

            ct_patch = ct[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
            mask_patch = mask[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        else:
            # Center crop for validation
            ct_patch = ct[:self.patch_size[0], :self.patch_size[1], :self.patch_size[2]]
            mask_patch = mask[:self.patch_size[0], :self.patch_size[1], :self.patch_size[2]]

        # Convert to tensors
        ct_patch = torch.from_numpy(ct_patch).unsqueeze(0)  # (1, D, H, W)
        mask_patch = torch.from_numpy(mask_patch).unsqueeze(0)

        return ct_patch, mask_patch


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, 
                    epoch, writer, gradient_accum_steps=1):
    """Train for one epoch."""

    model.train()

    running_loss = 0.0
    running_dice = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    optimizer.zero_grad()

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        # Mixed precision forward pass
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss = loss / gradient_accum_steps  # Scale loss for gradient accumulation

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % gradient_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Compute metrics
        with torch.no_grad():
            pred_mask = torch.sigmoid(outputs) > 0.5
            dice = dice_coefficient(pred_mask, masks)

        # Update running metrics
        running_loss += loss.item() * gradient_accum_steps
        running_dice += dice

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss / (batch_idx + 1):.4f}',
            'dice': f'{running_dice / (batch_idx + 1):.4f}'
        })

        # Log to TensorBoard
        global_step = epoch * len(dataloader) + batch_idx
        if batch_idx % 10 == 0:
            writer.add_scalar('Train/Loss', loss.item() * gradient_accum_steps, global_step)
            writer.add_scalar('Train/Dice', dice, global_step)

    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)

    return epoch_loss, epoch_dice


def validate(model, dataloader, criterion, device):
    """Validate the model."""

    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_sensitivity = 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            pred_mask = torch.sigmoid(outputs) > 0.5
            dice = dice_coefficient(pred_mask, masks)
            sensitivity = sensitivity_recall(pred_mask, masks)

            running_loss += loss.item()
            running_dice += dice
            running_sensitivity += sensitivity

    val_loss = running_loss / len(dataloader)
    val_dice = running_dice / len(dataloader)
    val_sensitivity = running_sensitivity / len(dataloader)

    return val_loss, val_dice, val_sensitivity


def save_checkpoint(model, optimizer, epoch, val_dice, save_dir, is_best=False):
    """Save model checkpoint."""

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_dice': val_dice,
    }

    # Save latest checkpoint
    checkpoint_path = Path(save_dir) / 'checkpoint_last.pth'
    torch.save(checkpoint, checkpoint_path)

    # Save best checkpoint
    if is_best:
        best_path = Path(save_dir) / 'checkpoint_best.pth'
        torch.save(checkpoint, best_path)
        print(f"✓ Saved best model (Dice: {val_dice:.4f})")

    # Save periodic checkpoint
    if epoch % 10 == 0:
        periodic_path = Path(save_dir) / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, periodic_path)


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def dice_coefficient(pred, target, smooth=1e-5):
    """Compute Dice coefficient."""
    pred = pred.view(-1).float()
    target = target.view(-1).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


def sensitivity_recall(pred, target, smooth=1e-5):
    """Compute sensitivity."""
    pred = pred.view(-1).float()
    target = target.view(-1).float()
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    sensitivity = (tp + smooth) / (tp + fn + smooth)
    return sensitivity.item()


def main(args):
    """Main training function."""

    # Load configuration
    config = load_config(args.config)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    print("Creating model...")
    # Import UNet3D from your modules
    # model = UNet3D(
    #     in_channels=config['model']['in_channels'],
    #     out_channels=config['model']['out_channels'],
    #     features=config['model']['features'],
    #     trilinear=config['model']['trilinear']
    # )

    # Placeholder for demonstration
    class DummyUNet3D(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(1, 1, 3, padding=1)
        def forward(self, x):
            return self.conv(x)

    model = DummyUNet3D()
    model = model.to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create datasets
    print("Loading datasets...")
    train_dataset = AirwayDataset(
        args.data_dir,
        split='train',
        patch_size=tuple(config['data']['patch_size'])
    )

    val_dataset = AirwayDataset(
        args.data_dir,
        split='val',
        patch_size=tuple(config['data']['patch_size'])
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create loss function
    # criterion = CombinedLoss(
    #     dice_weight=config['loss']['dice_weight'],
    #     smooth=config['loss']['smooth']
    # )
    criterion = nn.BCEWithLogitsLoss()  # Placeholder

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config['scheduler']['mode'],
        factor=config['scheduler']['factor'],
        patience=config['scheduler']['patience'],
        min_lr=config['scheduler']['min_lr'],
        verbose=config['scheduler']['verbose']
    )

    # Create gradient scaler for mixed precision
    scaler = GradScaler() if config['training']['use_amp'] else None

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=output_dir / 'logs')

    # Training loop
    print("\nStarting training...")
    best_dice = 0.0

    for epoch in range(1, config['training']['num_epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['training']['num_epochs']}")
        print(f"{'='*60}")

        # Train
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch, writer, config['training']['gradient_accumulation_steps']
        )

        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")

        # Validate
        if epoch % config['validation']['frequency'] == 0:
            val_loss, val_dice, val_sensitivity = validate(
                model, val_loader, criterion, device
            )

            print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, "
                  f"Val Sensitivity: {val_sensitivity:.4f}")

            # Log to TensorBoard
            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/Dice', val_dice, epoch)
            writer.add_scalar('Val/Sensitivity', val_sensitivity, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

            # Update scheduler
            scheduler.step(val_dice)

            # Save checkpoint
            is_best = val_dice > best_dice
            if is_best:
                best_dice = val_dice

            if epoch % config['checkpoint']['save_frequency'] == 0 or is_best:
                save_checkpoint(model, optimizer, epoch, val_dice, checkpoint_dir, is_best)

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation Dice: {best_dice:.4f}")
    print("="*60)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D U-Net for airway segmentation")

    parser.add_argument('--config', type=str, default='airway_unet_config.yaml',
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./experiments/airway',
                       help='Path to output directory')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    main(args)
