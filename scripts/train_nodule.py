"""Train V-Net for nodule segmentation."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import SegmentationDataset, dataset_summary
from src.models.losses import CombinedLoss, DiceLoss, FocalLoss, TverskyLoss
from src.models.metrics import dice_coefficient, sensitivity_recall
from src.models.vnet import VNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_loss(loss_cfg: Dict) -> nn.Module:
    loss_type = loss_cfg["type"].lower()

    if loss_type == "combinedloss":
        return CombinedLoss(dice_weight=loss_cfg.get("dice_weight", 0.5), smooth=loss_cfg.get("smooth", 1e-5))
    if loss_type == "diceloss":
        return DiceLoss(smooth=loss_cfg.get("smooth", 1e-5), apply_sigmoid=True)
    if loss_type == "focalloss":
        return FocalLoss(alpha=loss_cfg.get("alpha", 0.25), gamma=loss_cfg.get("gamma", 2.0))
    if loss_type == "tverskyloss":
        return TverskyLoss(
            alpha=loss_cfg.get("alpha", 0.5),
            beta=loss_cfg.get("beta", 0.5),
            smooth=loss_cfg.get("smooth", 1e-5),
            apply_sigmoid=True,
        )
    if loss_type == "bcewithlogitsloss":
        return nn.BCEWithLogitsLoss()

    raise ValueError(f"Unsupported loss type: {loss_cfg['type']}")


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, val_dice: float, path: Path) -> None:
    checkpoint = {
        "epoch": epoch,
        "val_dice": float(val_dice),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return int(ckpt["epoch"]) + 1, float(ckpt.get("val_dice", 0.0))


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    gradient_accum_steps: int,
    epoch: int,
    writer: SummaryWriter,
    use_amp: bool,
) -> tuple[float, float]:
    model.train()

    running_loss = 0.0
    running_dice = 0.0

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Train {epoch}")

    for batch_idx, (images, masks) in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
            scaled_loss = loss / gradient_accum_steps

        scaler.scale(scaled_loss).backward()

        if (batch_idx + 1) % gradient_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            preds = torch.sigmoid(logits) > 0.5
            dice = dice_coefficient(preds, masks)

        running_loss += loss.item()
        running_dice += dice

        global_step = (epoch - 1) * len(dataloader) + batch_idx
        if batch_idx % 10 == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/dice", dice, global_step)

        pbar.set_postfix(loss=f"{running_loss / (batch_idx + 1):.4f}", dice=f"{running_dice / (batch_idx + 1):.4f}")

    return running_loss / len(dataloader), running_dice / len(dataloader)


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float, float]:
    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_sens = 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Val"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, masks)
            preds = torch.sigmoid(logits) > 0.5

            running_loss += loss.item()
            running_dice += dice_coefficient(preds, masks)
            running_sens += sensitivity_recall(preds, masks)

    n = len(dataloader)
    return running_loss / n, running_dice / n, running_sens / n


def main(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config["training"].get("use_amp", True) and device.type == "cuda")

    model = VNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        features=config["model"].get("features", 16),
    ).to(device)

    criterion = build_loss(config["loss"])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
        betas=tuple(config["optimizer"].get("betas", [0.9, 0.999])),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["scheduler"].get("T_max", config["training"]["num_epochs"])),
        eta_min=float(config["scheduler"].get("eta_min", 1e-6)),
    )

    train_dataset = SegmentationDataset(
        data_dir=args.data_dir,
        split="train",
        mask_suffix="_nodule.nii.gz",
        patch_size=config["data"]["patch_size"],
        target_spacing=config["data"]["spacing"],
        intensity_range=config["data"]["intensity_range"],
        seed=args.seed,
    )
    val_dataset = SegmentationDataset(
        data_dir=args.data_dir,
        split="val",
        mask_suffix="_nodule.nii.gz",
        patch_size=config["data"]["patch_size"],
        target_spacing=config["data"]["spacing"],
        intensity_range=config["data"]["intensity_range"],
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=bool(config["training"].get("pin_memory", False)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=bool(config["training"].get("pin_memory", False)),
    )

    print("Train dataset:", dataset_summary(train_dataset))
    print("Val dataset:", dataset_summary(val_dataset))
    print(f"Model params: {model.count_parameters():,}")
    print(f"Using device={device}, amp={use_amp}")

    scaler = GradScaler(enabled=use_amp)
    writer = SummaryWriter(log_dir=output_dir / config.get("tensorboard", {}).get("log_dir", "logs"))

    start_epoch = 1
    best_dice = 0.0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        start_epoch, best_dice = load_checkpoint(resume_path, model, optimizer, device)
        print(f"Resumed from {resume_path} at epoch {start_epoch} (best dice={best_dice:.4f})")

    num_epochs = int(config["training"]["num_epochs"])
    save_freq = int(config.get("checkpoint", {}).get("save_frequency", 10))
    grad_accum = int(config["training"].get("gradient_accumulation_steps", 1))
    val_freq = int(config.get("validation", {}).get("frequency", 1))

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss, train_dice = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            grad_accum,
            epoch,
            writer,
            use_amp,
        )
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_dice={train_dice:.4f}")

        if epoch % val_freq == 0:
            val_loss, val_dice, val_sens = validate(model, val_loader, criterion, device)

            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/dice", val_dice, epoch)
            writer.add_scalar("val/sensitivity", val_sens, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            print(f"Epoch {epoch}: val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_sens={val_sens:.4f}")

            save_checkpoint(model, optimizer, epoch, val_dice, ckpt_dir / "checkpoint_last.pth")
            if val_dice > best_dice:
                best_dice = val_dice
                save_checkpoint(model, optimizer, epoch, val_dice, ckpt_dir / "checkpoint_best.pth")
                print(f"Saved best checkpoint at epoch {epoch} (dice={best_dice:.4f})")

        scheduler.step()

        if epoch % save_freq == 0:
            save_checkpoint(model, optimizer, epoch, best_dice, ckpt_dir / f"checkpoint_epoch_{epoch}.pth")

    writer.close()
    print(f"Training complete. Best validation dice: {best_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train V-Net for nodule segmentation")
    parser.add_argument("--config", type=str, default="configs/nodule_vnet_config.yaml")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="experiments/nodule")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())
