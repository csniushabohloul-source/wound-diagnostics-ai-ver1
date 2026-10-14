import os
import sys
from pathlib import Path

# Ensure root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import time
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from src.config import PROJECT_ROOT, CHECKPOINTS_DIR, RESULTS_DIR, NUM_CLASSES, LEARNING_RATE
from src.data.dataset import ProcessedWoundDataset
from src.models.lightweight_unet import CostEffectiveMultiTaskWoundUNet
from src.models.losses import CombinedWoundLoss
from src.utils.metrics import mean_iou, dice_coefficient


def train_cost_effective_model(
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 1e-3,
    device: str = "cpu"
):
    """
    Trains the lightweight Cost-Effective U-Net on patient-split clinical data.
    Logs loss, validation Dice score, and saves best model checkpoints.
    """
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Starting Training: Cost-Effective Edge AI Wound Model")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 60)

    # 1. Dataset & Patient-level Split
    train_ds, val_ds = ProcessedWoundDataset.create_patient_split(val_ratio=0.3, seed=42)
    print(f"Loaded Patient-split: {len(train_ds)} train samples, {len(val_ds)} val samples.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Model, Loss, Optimizer
    model = CostEffectiveMultiTaskWoundUNet(in_channels=3, num_classes=NUM_CLASSES, base_channels=32)
    model.to(device)

    criterion = CombinedWoundLoss(num_classes=NUM_CLASSES, alpha=0.5, beta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Tracking metrics
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_miou": []
    }

    best_dice = -1.0
    best_checkpoint_path = CHECKPOINTS_DIR / "best_cost_effective_unet.pt"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / max(len(train_ds), 1)
        scheduler.step()

        # Validation loop
        model.eval()
        val_running_loss = 0.0
        dices = []
        mious = []

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)
                val_running_loss += loss.item() * images.size(0)

                preds = torch.argmax(logits, dim=1)
                
                # Compute Dice and mIoU
                for b in range(images.size(0)):
                    p_mask = (preds[b] > 0).float()
                    t_mask = (masks[b] > 0).float()
                    dices.append(dice_coefficient(p_mask, t_mask))
                    mious.append(mean_iou(preds[b], masks[b], num_classes=NUM_CLASSES))

        epoch_val_loss = val_running_loss / max(len(val_ds), 1)
        epoch_val_dice = float(np.mean(dices)) if dices else 0.0
        epoch_val_miou = float(np.mean(mious)) if mious else 0.0

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_dice"].append(epoch_val_dice)
        history["val_miou"].append(epoch_val_miou)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Dice: {epoch_val_dice*100:.2f}% | "
            f"Val mIoU: {epoch_val_miou*100:.2f}%"
        )

        # Checkpoint if best
        if epoch_val_dice > best_dice:
            best_dice = epoch_val_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": best_dice,
                "val_miou": epoch_val_miou,
                "num_classes": NUM_CLASSES
            }, best_checkpoint_path)

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"Training completed in {total_time:.1f}s!")
    print(f"Best Validation Dice: {best_dice*100:.2f}%")
    print(f"Model saved to: {best_checkpoint_path}")

    # Plot and save learning curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    epochs_range = range(1, epochs + 1)

    ax1.plot(epochs_range, history["train_loss"], label="Train Loss", color="#e74c3c", linewidth=2)
    ax1.plot(epochs_range, history["val_loss"], label="Val Loss", color="#3498db", linewidth=2, linestyle="--")
    ax1.set_title("Training and Validation Loss", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(epochs_range, [d * 100 for d in history["val_dice"]], label="Val Dice Score %", color="#2ecc71", linewidth=2)
    ax2.plot(epochs_range, [m * 100 for m in history["val_miou"]], label="Val mIoU %", color="#9b59b6", linewidth=2, linestyle=":")
    ax2.set_title("Validation Segmentation Metrics", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Score %", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    curve_plot_path = RESULTS_DIR / "training_curves.png"
    plt.savefig(str(curve_plot_path))
    plt.close()
    print(f"Learning curves plot saved to: {curve_plot_path}")

    return model, history


if __name__ == "__main__":
    train_cost_effective_model(epochs=12, batch_size=4)
