import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import cv2
from typing import Dict, Tuple, Any
from src.config import CLASS_COLORS_RGB, TISSUE_CLASSES


class WoundMaskEncoder:
    """
    Encodes and decodes multi-tissue wound masks according to the canonical LUTSeg standard:
    0: Background
    1: Epithelial tissue (Green)
    2: Slough (Yellow)
    3: Granulation tissue (Magenta)
    4: Necrotic tissue (Dark Orange)
    5: Other (Cyan)
    """

    def __init__(self, color_map: Dict[int, Tuple[int, int, int]] = CLASS_COLORS_RGB):
        self.color_map = color_map

    def rgb_to_class_indices(self, mask_rgb: np.ndarray, tolerance: int = 35) -> np.ndarray:
        """
        Converts an RGB mask (H, W, 3) to a single-channel index mask (H, W).
        Uses Euclidean distance in RGB color space to match annotated pixels.
        """
        h, w = mask_rgb.shape[:2]
        indices = np.zeros((h, w), dtype=np.uint8)
        
        # Flatten image for distance computation
        pixels = mask_rgb.reshape(-1, 3).astype(np.float32)
        min_distances = np.full(pixels.shape[0], fill_value=1e9, dtype=np.float32)
        assigned_classes = np.zeros(pixels.shape[0], dtype=np.uint8)

        for class_id, target_rgb in self.color_map.items():
            if class_id == 255:
                continue
            color_vec = np.array(target_rgb, dtype=np.float32)
            dist = np.linalg.norm(pixels - color_vec, axis=1)
            closer = dist < min_distances
            min_distances[closer] = dist[closer]
            assigned_classes[closer] = class_id

        # Pixels with high distance from any known color remain background or ignore
        valid_mask = min_distances <= tolerance
        assigned_classes[~valid_mask] = 0
        
        return assigned_classes.reshape(h, w)

    def class_indices_to_rgb(self, mask_indices: np.ndarray) -> np.ndarray:
        """Converts an index mask (H, W) back to a vibrant RGB visualization (H, W, 3)."""
        h, w = mask_indices.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in self.color_map.items():
            rgb[mask_indices == class_id] = color
        return rgb

    def compute_tissue_composition(self, mask_indices: np.ndarray) -> Dict[str, Any]:
        """
        Computes the clinical area and percentage composition of each tissue type.
        Calculates:
        - Total wound area (sum of non-background pixels)
        - Percentage of Granulation, Slough, Necrotic, Epithelial
        - Healing Index (HI): (Granulation + Epithelial) / Total_Wound_Area
        """
        total_pixels = mask_indices.size
        wound_mask = (mask_indices > 0) & (mask_indices < 255)
        wound_pixels = int(np.sum(wound_mask))

        stats = {
            "total_pixels": total_pixels,
            "wound_pixels": wound_pixels,
            "wound_area_ratio": round(wound_pixels / total_pixels, 4) if total_pixels > 0 else 0.0,
            "tissues": {},
            "healing_index": 0.0
        }

        granulation_px = 0
        epithelial_px = 0
        slough_px = 0
        necrotic_px = 0

        for class_id, name in TISSUE_CLASSES.items():
            if class_id == 0:
                continue
            count = int(np.sum(mask_indices == class_id))
            pct_wound = round((count / wound_pixels * 100.0), 2) if wound_pixels > 0 else 0.0
            stats["tissues"][name] = {
                "pixels": count,
                "percentage_of_wound": pct_wound
            }
            if class_id == 1:
                epithelial_px = count
            elif class_id == 2:
                slough_px = count
            elif class_id == 3:
                granulation_px = count
            elif class_id == 4:
                necrotic_px = count

        if wound_pixels > 0:
            stats["healing_index"] = round((granulation_px + epithelial_px) / wound_pixels, 4)
            stats["slough_ratio"] = round(slough_px / wound_pixels, 4)
            stats["necrotic_ratio"] = round(necrotic_px / wound_pixels, 4)
        else:
            stats["healing_index"] = 0.0
            stats["slough_ratio"] = 0.0
            stats["necrotic_ratio"] = 0.0

        return stats
