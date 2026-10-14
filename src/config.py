import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset Specific Paths
LUTSEG_DIR = RAW_DATA_DIR / "lutseg"
WILLIE_DIR = RAW_DATA_DIR / "willie"
FUSEG_DIR = RAW_DATA_DIR / "fuseg"
DFUC_DIR = RAW_DATA_DIR / "dfuc"

# Image & Training Hyperparameters
IMAGE_SIZE = (256, 256)  # Cost-effective & fast for edge deployment
BATCH_SIZE = 8
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 40

# Canonical Tissue Classes for Multi-Class Wound Segmentation (LUTSeg Standard)
TISSUE_CLASSES = {
    0: "Background",
    1: "Epithelial tissue",
    2: "Slough",
    3: "Granulation tissue",
    4: "Necrotic tissue",
    5: "Other"
}
NUM_CLASSES = len(TISSUE_CLASSES)

CLASS_COLORS_RGB = {
    0: (0, 0, 0),         # Background
    1: (0, 255, 0),       # Epithelial (Green)
    2: (255, 255, 0),     # Slough (Yellow)
    3: (255, 0, 255),     # Granulation (Magenta)
    4: (250, 144, 49),    # Necrotic (Dark Orange)
    5: (0, 255, 255),     # Other (Cyan)
    255: (200, 200, 200)  # Ignore
}

# Hardware Config
DEVICE = "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu"
