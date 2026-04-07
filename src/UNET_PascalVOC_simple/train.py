import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader

from src.UNET_PascalVOC_simple.model import UNet
from src.UNET_PascalVOC_simple.dataset import PascalVOCSegmentation, NUM_CLASSES
from src.utils.aws_utils import sync_to_s3, terminate_self
from src.utils.device_utils import get_device
from src.utils.logging_utils import get_logger


def pixel_accuracy(preds, targets):
    """Compute overall pixel accuracy."""
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total


def mean_iou(preds, targets, num_classes):
    """Compute mean Intersection over Union across all classes present in the batch."""
    ious = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return sum(ious) / len(ious) if ious else 0.0


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for batch_idx, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # (B, C, H, W)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        total_acc += pixel_accuracy(preds, masks)
        num_batches += 1

        if (batch_idx + 1) % 20 == 0:
            logger.info(
                "Epoch %d | Batch %d/%d | Loss: %.4f",
                epoch, batch_idx + 1, len(loader), loss.item(),
            )

    avg_loss = running_loss / num_batches
    avg_acc = total_acc / num_batches
    return avg_loss, avg_acc


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    total_acc = 0.0
    total_miou = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total_acc += pixel_accuracy(preds, masks)
            total_miou += mean_iou(preds, masks, num_classes)
            num_batches += 1

    avg_loss = running_loss / num_batches
    avg_acc = total_acc / num_batches
    avg_miou = total_miou / num_batches
    return avg_loss, avg_acc, avg_miou


def main():
    parser = argparse.ArgumentParser(description="U-Net Pascal VOC Segmentation Training")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./checkpoints/unet_voc")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="unet-voc-segmentation")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--s3-bucket", type=str, default="", help="S3 bucket to sync checkpoints to")
    parser.add_argument("--s3-prefix", type=str, default="", help="S3 key prefix (e.g. unet-run-001)")
    parser.add_argument("--self-terminate", action="store_true", help="Terminate EC2 instance when training finishes")
    args = parser.parse_args()

    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("U-Net Pascal VOC Semantic Segmentation Training")
    logger.info("=" * 60)

    device = get_device()
    logger.info("Using device: %s", device)

    # --- Dataset ---
    logger.info("Downloading and preparing Pascal VOC 2012 dataset...")
    train_dataset = PascalVOCSegmentation(
        root=args.data_dir, image_set="train", image_size=args.image_size,
    )
    val_dataset = PascalVOCSegmentation(
        root=args.data_dir, image_set="val", image_size=args.image_size,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
    )

    logger.info("Train samples: %d | Val samples: %d", len(train_dataset), len(val_dataset))

    # --- Model ---
    logger.info("Building U-Net (in_channels=3, num_classes=%d)...", NUM_CLASSES)
    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total parameters: %d (%.1fM)", total_params, total_params / 1e6)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    # --- Weights & Biases ---
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
        wandb.watch(model, log="all", log_freq=100)

    # --- Training loop ---
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info("Starting training for %d epochs...", args.epochs)

    best_miou = 0.0
    for epoch in range(1, args.epochs + 1):
        logger.info("-" * 40)
        logger.info("Epoch %d/%d (lr=%.6f)", epoch, args.epochs, optimizer.param_groups[0]["lr"])

        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
        )
        elapsed = time.time() - start

        val_loss, val_acc, val_miou = evaluate(
            model, val_loader, criterion, device, NUM_CLASSES,
        )

        scheduler.step(val_miou)

        logger.info(
            "Train Loss: %.4f | Train Acc: %.2f%% | "
            "Val Loss: %.4f | Val Acc: %.2f%% | Val mIoU: %.4f | Time: %.1fs",
            train_loss, train_acc * 100, val_loss, val_acc * 100, val_miou, elapsed,
        )

        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc * 100,
                "val/loss": val_loss,
                "val/accuracy": val_acc * 100,
                "val/mIoU": val_miou,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time": elapsed,
            })

        if val_miou > best_miou:
            best_miou = val_miou
            path = os.path.join(args.save_dir, "best_model.pt")
            torch.save(model.state_dict(), path)
            logger.info("New best model saved (mIoU=%.4f)", val_miou)

    # --- Final save ---
    final_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    logger.info("Final model saved to %s", final_path)
    logger.info("Training complete. Best val mIoU: %.4f", best_miou)

    if args.wandb:
        wandb.finish()

    # --- Sync checkpoints to S3 ---
    if args.s3_bucket:
        prefix = args.s3_prefix or f"unet-{time.strftime('%Y%m%d-%H%M%S')}"
        logger.info("Syncing checkpoints to s3://%s/%s/...", args.s3_bucket, prefix)
        sync_to_s3(args.save_dir, args.s3_bucket, prefix)

    # --- Self-terminate EC2 instance ---
    if args.self_terminate:
        logger.info("Self-terminating EC2 instance...")
        terminate_self()


if __name__ == "__main__":
    main()
