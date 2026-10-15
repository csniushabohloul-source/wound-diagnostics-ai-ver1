import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiClassDiceLoss(nn.Module):
    """
    Robust Multi-Class Dice Loss for wound tissue segmentation.
    Combines binary wound bed localization (wound vs background) with
    class-wise Dice over tissues present in the ground truth.
    """

    def __init__(self, num_classes: int = 6, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets.clamp(0, self.num_classes - 1), num_classes=self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # 1. Binary wound bed Dice (wound vs background)
        wound_prob = 1.0 - probs[:, 0]
        wound_target = (targets > 0).float()
        wound_inter = (wound_prob * wound_target).sum(dim=(1, 2))
        wound_card = (wound_prob + wound_target).sum(dim=(1, 2))
        wound_dice = (2.0 * wound_inter + self.smooth) / (wound_card + self.smooth)

        # 2. Per-class tissue Dice for classes present in ground truth
        inter = (probs * one_hot).sum(dim=(0, 2, 3))
        card = (probs + one_hot).sum(dim=(0, 2, 3))
        present = (one_hot.sum(dim=(0, 2, 3)) > 0)
        
        tissue_dice = (2.0 * inter + self.smooth) / (card + self.smooth)
        tissue_dice_foreground = tissue_dice[1:][present[1:]]

        if tissue_dice_foreground.numel() > 0:
            loss = 1.0 - (0.5 * wound_dice.mean() + 0.5 * tissue_dice_foreground.mean())
        else:
            loss = 1.0 - wound_dice.mean()

        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss down-weights easy background pixels and focuses learning on hard
    and blurred clinical tissue boundaries (e.g. necrotic vs granulation).
    """

    def __init__(self, num_classes: int = 6, gamma: float = 2.0, ignore_index: int = 255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        # Give higher weight to clinical tissues (classes 1-5) and lower weight to background (class 0)
        weights = torch.ones(num_classes)
        weights[0] = 0.2  # Down-weight dominant background
        weights[1:] = 2.5 # Up-weight clinical tissues (granulation, slough, necrotic, epithelial)
        self.register_buffer("weights", weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.weights.to(logits.device),
            reduction='none',
            ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class CombinedWoundLoss(nn.Module):
    """
    Combined Loss function:
    L_total = alpha * L_Focal + beta * L_Dice
    """

    def __init__(self, num_classes: int = 6, alpha: float = 0.4, beta: float = 0.6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal = FocalLoss(num_classes=num_classes, gamma=2.0)
        self.dice = MultiClassDiceLoss(num_classes=num_classes)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_focal = self.focal(logits, targets)
        l_dice = self.dice(logits, targets)
        return self.alpha * l_focal + self.beta * l_dice
