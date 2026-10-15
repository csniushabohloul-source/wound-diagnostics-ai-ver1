import os
import sys
import time
import random
from pathlib import Path

# Ensure root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from src.config import CHECKPOINTS_DIR, RESULTS_DIR, NUM_CLASSES, TISSUE_CLASSES
from src.models.lightweight_unet import CostEffectiveMultiTaskWoundUNet
from src.models.losses import CombinedWoundLoss
from src.utils.metrics import dice_coefficient, mean_iou


class MultiDatasetWoundDataset(Dataset):
    """
    Combined High-Accuracy Clinical Wound Dataset:
    Aggregates FUSeg (MICCAI Challenge), AZH Wound Center, and LUTSeg longitudinal multi-tissue visits.
    Caches resized tensors in memory for fast, bottleneck-free CPU training.
    """

    def __init__(
        self,
        target_size: tuple = (160, 160),
        is_train: bool = True,
        max_fuseg: int = 250,
        max_azh: int = 150,
        augment: bool = True
    ):
        self.target_size = target_size
        self.is_train = is_train
        self.augment = augment and is_train

        self.images = []       # list of (3, H, W) float32 tensors [0, 1]
        self.binary_masks = [] # list of (1, H, W) float32 tensors {0, 1}
        self.tissue_masks = [] # list of (H, W) long tensors {0..5}
        self.has_tissue = []   # list of bool: True if fine-grained tissue annotations exist
        self.risk_labels = []  # list of int: 0 (Low), 1 (Moderate), 2 (High)

        self._load_all_data(max_fuseg, max_azh)

    def _load_all_data(self, max_fuseg: int, max_azh: int):
        print(f"Loading clinical wound datasets (is_train={self.is_train}, target_size={self.target_size})...")
        t0 = time.time()

        # 1. Load LUTSeg (Fine-grained multi-tissue expert annotations)
        proc_img_dir = PROJECT_ROOT / "data" / "processed" / "images"
        proc_mask_dir = PROJECT_ROOT / "data" / "processed" / "masks"
        
        lutseg_files = sorted(list(proc_img_dir.glob("*.npy")))
        if lutseg_files:
            # Deterministic split by patient ID to prevent leakage
            patients = sorted(list(set(f.stem.split("_")[0] + "_" + f.stem.split("_")[1] for f in lutseg_files)))
            random.seed(42)
            shuffled_patients = list(patients)
            random.shuffle(shuffled_patients)
            split = int(len(shuffled_patients) * 0.75)
            train_pats = set(shuffled_patients[:split])
            val_pats = set(shuffled_patients[split:])

            for f in lutseg_files:
                pat_id = f.stem.split("_")[0] + "_" + f.stem.split("_")[1]
                if self.is_train and pat_id not in train_pats:
                    continue
                if not self.is_train and pat_id not in val_pats:
                    continue

                img_np = np.load(f)  # (256, 256, 3) uint8
                mask_np = np.load(proc_mask_dir / f.name) # (256, 256) uint8

                # Resize to target
                img_res = cv2.resize(img_np, self.target_size, interpolation=cv2.INTER_LINEAR)
                mask_res = cv2.resize(mask_np, self.target_size, interpolation=cv2.INTER_NEAREST)

                bin_mask = (mask_res > 0).astype(np.float32)

                # Clinical Risk calculation based on tissue breakdown
                total_wound = np.sum(bin_mask)
                if total_wound > 0:
                    necrotic_ratio = np.sum(mask_res == 4) / total_wound
                    slough_ratio = np.sum(mask_res == 2) / total_wound
                    if necrotic_ratio > 0.15 or slough_ratio > 0.35:
                        risk = 2 # High
                    elif slough_ratio > 0.10:
                        risk = 1 # Moderate
                    else:
                        risk = 0 # Low
                else:
                    risk = 0

                img_tensor = torch.from_numpy(np.transpose(img_res.astype(np.float32) / 255.0, (2, 0, 1)))
                bin_tensor = torch.from_numpy(bin_mask).unsqueeze(0)
                tis_tensor = torch.from_numpy(mask_res.astype(np.int64))

                self.images.append(img_tensor)
                self.binary_masks.append(bin_tensor)
                self.tissue_masks.append(tis_tensor)
                self.has_tissue.append(True)
                self.risk_labels.append(risk)

        print(f"  -> Loaded {len(self.images)} LUTSeg samples.")

        # 2. Load FUSeg (MICCAI Challenge Binary Wound Segmentation)
        fuseg_split = "train" if self.is_train else "validation"
        fuseg_img_dir = PROJECT_ROOT / "data" / "raw" / "fuseg" / fuseg_split / "images"
        fuseg_lbl_dir = PROJECT_ROOT / "data" / "raw" / "fuseg" / fuseg_split / "labels"

        if fuseg_img_dir.exists():
            fuseg_imgs = sorted(list(fuseg_img_dir.glob("*.png")))
            limit = max_fuseg if self.is_train else min(60, len(fuseg_imgs))
            count_f = 0
            for img_p in fuseg_imgs[:limit]:
                lbl_p = fuseg_lbl_dir / img_p.name
                if not lbl_p.exists():
                    continue

                img_bgr = cv2.imread(str(img_p))
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                mask_raw = cv2.imread(str(lbl_p), 0)
                if mask_raw is None:
                    continue

                img_res = cv2.resize(img_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
                mask_res = cv2.resize(mask_raw, self.target_size, interpolation=cv2.INTER_NEAREST)

                bin_mask = (mask_res > 127).astype(np.float32)

                # Generate pseudo-tissue guidance inside wound bed based on color
                # Granulation (3: red/pink), Slough (2: yellow), Necrotic (4: dark)
                tis_mask = np.zeros(self.target_size, dtype=np.int64)
                wound_pts = (bin_mask > 0)
                if np.sum(wound_pts) > 0:
                    r = img_res[:, :, 0].astype(np.float32)
                    g = img_res[:, :, 1].astype(np.float32)
                    b = img_res[:, :, 2].astype(np.float32)
                    brightness = (r + g + b) / 3.0

                    # Color rule:
                    is_necrotic = wound_pts & (brightness < 45)
                    is_slough = wound_pts & (r > 130) & (g > 110) & (b < 100) & ~is_necrotic
                    is_granulation = wound_pts & ~is_necrotic & ~is_slough

                    tis_mask[is_granulation] = 3
                    tis_mask[is_slough] = 2
                    tis_mask[is_necrotic] = 4

                    wound_ratio = np.sum(wound_pts) / (self.target_size[0] * self.target_size[1])
                    risk = 2 if np.sum(is_necrotic) > 0 or wound_ratio > 0.20 else 1 if np.sum(is_slough) > 0 else 0
                else:
                    risk = 0

                img_tensor = torch.from_numpy(np.transpose(img_res.astype(np.float32) / 255.0, (2, 0, 1)))
                bin_tensor = torch.from_numpy(bin_mask).unsqueeze(0)
                tis_tensor = torch.from_numpy(tis_mask)

                self.images.append(img_tensor)
                self.binary_masks.append(bin_tensor)
                self.tissue_masks.append(tis_tensor)
                self.has_tissue.append(True)
                self.risk_labels.append(risk)
                count_f += 1

            print(f"  -> Loaded {count_f} FUSeg samples.")

        # 3. Load AZH (Wound Care Center Patches)
        azh_split = "train" if self.is_train else "test"
        azh_img_dir = PROJECT_ROOT / "data" / "raw" / "azh" / "extracted" / azh_split / "images"
        azh_lbl_dir = PROJECT_ROOT / "data" / "raw" / "azh" / "extracted" / azh_split / "labels"

        if azh_img_dir.exists():
            azh_imgs = sorted(list(azh_img_dir.glob("*.png")))
            limit = max_azh if self.is_train else min(30, len(azh_imgs))
            count_a = 0
            for img_p in azh_imgs[:limit]:
                lbl_p = azh_lbl_dir / img_p.name
                if not lbl_p.exists():
                    continue

                img_bgr = cv2.imread(str(img_p))
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                mask_raw = cv2.imread(str(lbl_p), 0)
                if mask_raw is None:
                    continue

                img_res = cv2.resize(img_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
                mask_res = cv2.resize(mask_raw, self.target_size, interpolation=cv2.INTER_NEAREST)

                bin_mask = (mask_res > 127).astype(np.float32)

                tis_mask = np.zeros(self.target_size, dtype=np.int64)
                wound_pts = (bin_mask > 0)
                if np.sum(wound_pts) > 0:
                    tis_mask[wound_pts] = 3  # Default granulation within confirmed wound bed
                    risk = 1
                else:
                    risk = 0

                img_tensor = torch.from_numpy(np.transpose(img_res.astype(np.float32) / 255.0, (2, 0, 1)))
                bin_tensor = torch.from_numpy(bin_mask).unsqueeze(0)
                tis_tensor = torch.from_numpy(tis_mask)

                self.images.append(img_tensor)
                self.binary_masks.append(bin_tensor)
                self.tissue_masks.append(tis_tensor)
                self.has_tissue.append(True)
                self.risk_labels.append(risk)
                count_a += 1

            print(f"  -> Loaded {count_a} AZH samples.")

        print(f"Total {len(self.images)} samples loaded in {time.time() - t0:.2f}s!")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        bin_mask = self.binary_masks[idx]
        tis_mask = self.tissue_masks[idx]
        has_tis = self.has_tissue[idx]
        risk = self.risk_labels[idx]

        # On-the-fly Data Augmentation
        if self.augment:
            # Horizontal Flip (50%)
            if random.random() > 0.5:
                img = torch.flip(img, dims=[2])
                bin_mask = torch.flip(bin_mask, dims=[2])
                tis_mask = torch.flip(tis_mask, dims=[1])
            # Vertical Flip (30%)
            if random.random() > 0.7:
                img = torch.flip(img, dims=[1])
                bin_mask = torch.flip(bin_mask, dims=[1])
                tis_mask = torch.flip(tis_mask, dims=[0])
            # Color Jitter: Brightness and Contrast
            if random.random() > 0.5:
                factor = 1.0 + (random.random() - 0.5) * 0.25
                img = torch.clamp(img * factor, 0.0, 1.0)

        return img, bin_mask, tis_mask, has_tis, risk


def train_high_accuracy_model(
    epochs: int = 6,
    batch_size: int = 16,
    lr: float = 1.2e-3,
    base_channels: int = 24
):
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    print("=" * 65)
    print("🚀 TRAINING HIGH-ACCURACY MULTI-DATASET WOUND MODEL")
    print(f"Datasets: FUSeg (250) + AZH (150) + LUTSeg (105)")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Base Channels: {base_channels}")
    print("=" * 65)

    # Datasets
    train_ds = MultiDatasetWoundDataset(
        target_size=(160, 160),
        is_train=True,
        max_fuseg=250,
        max_azh=150,
        augment=True
    )
    val_ds = MultiDatasetWoundDataset(
        target_size=(160, 160),
        is_train=False,
        max_fuseg=60,
        max_azh=30,
        augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Initialize Multi-Task Model
    model = CostEffectiveMultiTaskWoundUNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base_channels=base_channels,
        multi_task=True
    )
    model.to(device)

    # Losses
    tissue_criterion = CombinedWoundLoss(num_classes=NUM_CLASSES, alpha=0.4, beta=0.6)
    risk_criterion = nn.CrossEntropyLoss()
    pos_weight = torch.tensor([2.5]).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_boundary_dice": [],
        "val_tissue_dice": []
    }

    best_val_score = -1.0
    best_checkpoint_path = CHECKPOINTS_DIR / "best_cost_effective_unet.pt"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        running_loss = 0.0

        for images, bin_masks, tis_masks, has_tis, risks in train_loader:
            images = images.to(device)
            bin_masks = bin_masks.to(device)
            tis_masks = tis_masks.to(device)
            risks = risks.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)

            boundary_logits = outputs["boundary"]
            tissue_logits = outputs["tissue"]
            risk_logits = outputs["risk"]

            # 1. Boundary Loss (BCE + Dice)
            bce = F.binary_cross_entropy_with_logits(boundary_logits, bin_masks, pos_weight=pos_weight)
            prob_bound = torch.sigmoid(boundary_logits)
            inter = (prob_bound * bin_masks).sum(dim=(1, 2, 3))
            card = (prob_bound + bin_masks).sum(dim=(1, 2, 3))
            dice_loss = 1.0 - (2.0 * inter + 1e-5) / (card + 1e-5)
            loss_boundary = bce + dice_loss.mean()

            # 2. Tissue Loss
            loss_tissue = tissue_criterion(tissue_logits, tis_masks)

            # 3. Clinical Risk Loss
            loss_risk = risk_criterion(risk_logits, risks)

            total_loss = loss_boundary + 0.8 * loss_tissue + 0.2 * loss_risk
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += total_loss.item() * images.size(0)

        epoch_train_loss = running_loss / max(len(train_ds), 1)
        scheduler.step()

        # Validation
        model.eval()
        val_running_loss = 0.0
        boundary_dices = []
        tissue_dices = []

        with torch.no_grad():
            for images, bin_masks, tis_masks, has_tis, risks in val_loader:
                images = images.to(device)
                bin_masks = bin_masks.to(device)
                tis_masks = tis_masks.to(device)
                risks = risks.to(device).long()

                outputs = model(images)
                boundary_logits = outputs["boundary"]
                tissue_logits = outputs["tissue"]
                risk_logits = outputs["risk"]

                bce = F.binary_cross_entropy_with_logits(boundary_logits, bin_masks, pos_weight=pos_weight)
                prob_bound = torch.sigmoid(boundary_logits)
                inter = (prob_bound * bin_masks).sum(dim=(1, 2, 3))
                card = (prob_bound + bin_masks).sum(dim=(1, 2, 3))
                dice_loss = 1.0 - (2.0 * inter + 1e-5) / (card + 1e-5)
                loss_boundary = bce + dice_loss.mean()
                loss_tissue = tissue_criterion(tissue_logits, tis_masks)
                loss_risk = risk_criterion(risk_logits, risks)

                total_loss = loss_boundary + 0.8 * loss_tissue + 0.2 * loss_risk
                val_running_loss += total_loss.item() * images.size(0)

                # Metric computation
                pred_bounds = (prob_bound > 0.35).float()
                pred_tissues = torch.argmax(tissue_logits, dim=1)

                for b in range(images.size(0)):
                    b_dice = dice_coefficient(pred_bounds[b], bin_masks[b])
                    boundary_dices.append(b_dice)

                    # Tissue foreground dice
                    p_tis_fg = (pred_tissues[b] > 0).float()
                    t_tis_fg = (tis_masks[b] > 0).float()
                    t_dice = dice_coefficient(p_tis_fg, t_tis_fg)
                    tissue_dices.append(t_dice)

        epoch_val_loss = val_running_loss / max(len(val_ds), 1)
        mean_b_dice = float(np.mean(boundary_dices)) if boundary_dices else 0.0
        mean_t_dice = float(np.mean(tissue_dices)) if tissue_dices else 0.0
        combined_score = 0.6 * mean_b_dice + 0.4 * mean_t_dice

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_boundary_dice"].append(mean_b_dice)
        history["val_tissue_dice"].append(mean_t_dice)

        ep_duration = time.time() - ep_t0
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_duration:.1f}s) | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Wound Boundary Dice: {mean_b_dice*100:.2f}% | "
            f"Tissue Dice: {mean_t_dice*100:.2f}%"
        )

        # Save best model
        if combined_score > best_val_score:
            best_val_score = combined_score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_boundary_dice": mean_b_dice,
                "val_tissue_dice": mean_t_dice,
                "base_channels": base_channels,
                "num_classes": NUM_CLASSES,
                "multi_task": True,
                "target_size": (160, 160)
            }, best_checkpoint_path)

    total_time = time.time() - start_time
    print("=" * 65)
    print(f"🎉 Training Finished in {total_time:.1f}s ({total_time/60:.1f} mins)!")
    print(f"Best Combined Val Score: {best_val_score*100:.2f}%")
    print(f"Best Boundary Dice: {max(history['val_boundary_dice'])*100:.2f}%")
    print(f"Model saved to: {best_checkpoint_path}")

    # Plot & Save Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    epochs_range = range(1, epochs + 1)

    ax1.plot(epochs_range, history["train_loss"], label="Train Loss", color="#e74c3c", linewidth=2)
    ax1.plot(epochs_range, history["val_loss"], label="Val Loss", color="#3498db", linewidth=2, linestyle="--")
    ax1.set_title("Training and Validation Loss", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(epochs_range, [d * 100 for d in history["val_boundary_dice"]], label="Boundary Dice %", color="#2ecc71", linewidth=2)
    ax2.plot(epochs_range, [t * 100 for t in history["val_tissue_dice"]], label="Tissue Dice %", color="#9b59b6", linewidth=2, linestyle="-.")
    ax2.set_title("Wound Boundary & Tissue Dice Scores", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Dice Score (%)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    curve_path = RESULTS_DIR / "training_curves.png"
    plt.savefig(curve_path)
    plt.close()
    print(f"Learning curves saved to: {curve_path}")


if __name__ == "__main__":
    train_high_accuracy_model(epochs=6, batch_size=16, lr=1.2e-3, base_channels=24)
