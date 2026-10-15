import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from src.config import PROJECT_ROOT, NUM_CLASSES


class ProcessedWoundDataset:
    """
    Patient-level Wound Segmentation Dataset.
    Enforces strict patient-level separation to prevent data leakage across longitudinal visits.
    """

    def __init__(
        self,
        data_dir: Path = PROJECT_ROOT / "data" / "processed",
        patient_ids: Optional[List[str]] = None,
        transform=None,
        as_torch_tensor: bool = True
    ):
        self.data_dir = data_dir
        self.img_dir = data_dir / "images"
        self.mask_dir = data_dir / "masks"
        self.transform = transform
        self.as_torch_tensor = as_torch_tensor

        all_sample_files = sorted(list(self.img_dir.glob("*.npy")))
        
        # Filter by patient IDs if specified (e.g. for train/val split)
        if patient_ids:
            self.samples = [
                f for f in all_sample_files
                if any(f.stem.startswith(pid + "_") for pid in patient_ids)
            ]
        else:
            self.samples = all_sample_files

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, str]:
        img_path = self.samples[idx]
        mask_path = self.mask_dir / img_path.name

        img = np.load(img_path).astype(np.float32) / 255.0  # Normalize to [0, 1]
        mask = np.load(mask_path).astype(np.int64)          # Class indices (0 to 5)

        # Convert HWC to CHW for deep learning
        img = np.transpose(img, (2, 0, 1))

        if self.as_torch_tensor:
            try:
                import torch
                img = torch.from_numpy(img).float()
                mask = torch.from_numpy(mask).long()
            except ImportError:
                pass

        sample_id = img_path.stem
        return img, mask, sample_id

    @staticmethod
    def create_patient_split(
        data_dir: Path = PROJECT_ROOT / "data" / "processed",
        val_ratio: float = 0.25,
        seed: int = 42
    ) -> Tuple['ProcessedWoundDataset', 'ProcessedWoundDataset']:
        """
        Splits dataset by distinct patients (NOT by random images)
        to guarantee rigorous medical validation without leakage.
        """
        img_dir = data_dir / "images"
        files = list(img_dir.glob("*.npy"))
        
        # Extract unique patient names (e.g. 'Patient_10', 'Patient_20')
        patient_names = sorted(list(set(f.stem.split("_")[0] + "_" + f.stem.split("_")[1] for f in files)))
        
        np.random.seed(seed)
        shuffled = np.random.permutation(patient_names)
        split_idx = max(1, int(len(shuffled) * (1 - val_ratio)))
        
        train_patients = list(shuffled[:split_idx])
        val_patients = list(shuffled[split_idx:])

        train_ds = ProcessedWoundDataset(data_dir, patient_ids=train_patients)
        val_ds = ProcessedWoundDataset(data_dir, patient_ids=val_patients)

        return train_ds, val_ds
