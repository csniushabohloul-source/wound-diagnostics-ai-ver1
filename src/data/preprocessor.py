import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any


class WoundImagePreprocessor:
    """
    Cost-effective & Edge-friendly Preprocessor for smartphone wound images.
    Includes quality control (blur & lighting checks) and adaptive illumination correction (CLAHE).
    """

    def __init__(self, target_size: Tuple[int, int] = (256, 256), blur_threshold: float = 80.0):
        self.target_size = target_size
        self.blur_threshold = blur_threshold

    def assess_quality(self, img_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Assesses smartphone image quality:
        - Laplacian variance for blur detection.
        - Brightness and contrast statistics.
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # 1. Blur score (Laplacian variance)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = blur_score < self.blur_threshold
        
        # 2. Illumination statistics
        mean_brightness = float(np.mean(gray))
        std_contrast = float(np.std(gray))
        
        is_underexposed = mean_brightness < 40.0
        is_overexposed = mean_brightness > 220.0
        
        status = "ACCEPTABLE"
        warnings = []
        if is_blurry:
            status = "POOR"
            warnings.append(f"Image is blurry (Score: {blur_score:.1f} < {self.blur_threshold})")
        if is_underexposed:
            status = "POOR"
            warnings.append(f"Image is underexposed/too dark (Mean: {mean_brightness:.1f})")
        if is_overexposed:
            status = "POOR"
            warnings.append(f"Image is overexposed/glare (Mean: {mean_brightness:.1f})")
            
        return {
            "status": status,
            "blur_score": round(blur_score, 2),
            "brightness": round(mean_brightness, 2),
            "contrast": round(std_contrast, 2),
            "warnings": warnings,
            "is_valid": len(warnings) == 0
        }

    def apply_clahe(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization in LAB color space.
        Standardizes variable ambient lighting from commodity smartphones.
        """
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    def resize_with_aspect_ratio(self, img: np.ndarray, is_mask: bool = False) -> np.ndarray:
        """
        Resizes an image or mask to target_size with letterbox padding
        to prevent aspect ratio distortion of clinical lesions.
        """
        h, w = img.shape[:2]
        target_w, target_h = self.target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        
        # Pad to target dimensions
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        
        if is_mask and img.ndim == 2:
            padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        else:
            padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
        return padded

    def preprocess_pipeline(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Complete mobile preprocessing: quality assessment + CLAHE + aspect-ratio resize."""
        quality = self.assess_quality(img_rgb)
        enhanced = self.apply_clahe(img_rgb)
        resized = self.resize_with_aspect_ratio(enhanced, is_mask=False)
        return resized, quality
