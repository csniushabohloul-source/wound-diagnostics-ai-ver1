import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiClassDiceLoss(nn.Module):
    """
    Multi-Class Dice Loss designed for wound tissue segmentation with class imbalance.
    Computes smooth Dice coefficient across all classes (excluding or downweighting background).
    """

    def __init__(self, num_classes: int = 6, smooth: float = 1e-6, ignore_index: int = 255):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, H, W)
        targets: (B, H, W) with class indices [0, ..., C-1]
        """
        probs = F.softmax(logits, dim=1)
        
        # Mask out ignore index if present
        valid_mask = (targets != self.ignore_index)
        targets = targets * valid_mask.long()

        # One-hot encode targets
        one_hot = F.one_hot(targets.clamp(0, self.num_classes - 1), num_classes=self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float() # (B, C, H, W)

        # Apply valid mask
        valid_mask = valid_mask.unsqueeze(1).float()
        probs = probs * valid_mask
        one_hot = one_hot * valid_mask

        # Compute Dice per class (dim=(2, 3) over H, W)
        intersection = (probs * one_hot).sum(dim=(2, 3))
        cardinality = (probs.pow(2) + one_hot.pow(2)).sum(dim=(2, 3))
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Average across foreground classes (1 to C-1) to emphasize clinical tissues
        foreground_dice = dice_score[:, 1:].mean()
        return 1.0 - foreground_dice


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
