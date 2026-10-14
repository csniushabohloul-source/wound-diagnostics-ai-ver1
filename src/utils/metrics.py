import time
import torch
import numpy as np


def dice_coefficient(pred_mask: torch.Tensor, true_mask: torch.Tensor, eps: float = 1e-6) -> float:
    """Computes the Dice similarity coefficient."""
    pred = (pred_mask > 0.5).float()
    intersection = (pred * true_mask).sum()
    union = pred.sum() + true_mask.sum()
    return float((2.0 * intersection + eps) / (union + eps))


def mean_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor, num_classes: int, eps: float = 1e-6) -> float:
    """Computes mean Intersection-over-Union across classes."""
    ious = []
    pred_classes = torch.argmax(pred_mask, dim=1) if pred_mask.ndim == 4 and pred_mask.shape[1] > 1 else pred_mask
    
    for cls in range(num_classes):
        pred_c = (pred_classes == cls)
        true_c = (true_mask == cls)
        intersection = (pred_c & true_c).sum().float()
        union = (pred_c | true_c).sum().float()
        if union == 0:
            ious.append(1.0)
        else:
            ious.append(float((intersection + eps) / (union + eps)))
    return float(np.mean(ious))


def profile_model_efficiency(model: torch.nn.Module, input_size=(1, 3, 256, 256), device="cpu"):
    """
    Profiles model parameters, inference latency, and memory.
    Essential for evaluating the 'Cost-Effective / Edge AI' requirement.
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(*input_size).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
            
    # Measure Latency
    num_runs = 50
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    latency_ms = ((time.perf_counter() - start) / num_runs) * 1000
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_size_mb": total_params * 4 / (1024 * 1024),
        "mean_latency_ms": latency_ms,
        "fps": 1000.0 / latency_ms if latency_ms > 0 else 0
    }
