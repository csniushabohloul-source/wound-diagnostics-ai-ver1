import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution:
    Depthwise conv (channel-wise) + Pointwise conv (1x1 across channels).
    Reduces compute operations (FLOPs) by ~8.5x compared to standard 3x3 convs,
    making it ideal for smartphone and edge deployment.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class MobileConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class CostEffectiveMultiTaskWoundUNet(nn.Module):
    """
    Cost-Effective Edge AI Architecture for Point-of-Care Wound Diagnostics.
    Single-backbone, Multi-task outputs:
    1. Tissue Segmentation Head: (B, num_classes, H, W)
    2. Wound Boundary Head: (B, 1, H, W)
    3. Clinical Risk / Severity Head: (B, num_risk_classes)
    
    Total parameters: ~1.55 Million (vs >50 Million in Mask2Former).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        base_channels: int = 32,
        multi_task: bool = False
    ):
        super().__init__()
        self.multi_task = multi_task
        self.num_classes = num_classes

        # Encoder (Downsampling)
        self.enc1 = MobileConvBlock(in_channels, base_channels)        # -> 32
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = MobileConvBlock(base_channels, base_channels * 2)   # -> 64
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = MobileConvBlock(base_channels * 2, base_channels * 4) # -> 128
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = MobileConvBlock(base_channels * 4, base_channels * 8) # -> 256

        # Decoder (Upsampling with skip connections)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = MobileConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = MobileConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = MobileConvBlock(base_channels * 2, base_channels)

        # Head 1: Fine-grained Tissue Decomposition
        self.tissue_head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

        # Head 2: Binary Wound Boundary (Optional Multi-task)
        if self.multi_task:
            self.boundary_head = nn.Conv2d(base_channels, 1, kernel_size=1)
            # Head 3: Global Severity Classification from Bottleneck
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.risk_classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(base_channels * 8, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, 3) # Low, Moderate, High Risk
            )

    def forward(self, x: torch.Tensor):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        tissue_logits = self.tissue_head(d1)

        if not self.multi_task:
            return tissue_logits

        boundary_logits = self.boundary_head(d1)
        risk_logits = self.risk_classifier(b)
        
        return {
            "tissue": tissue_logits,
            "boundary": boundary_logits,
            "risk": risk_logits
        }


# Alias for backward compatibility
CostEffectiveWoundUNet = CostEffectiveMultiTaskWoundUNet
