"""Shared constants for FreshCheck."""

CLASS_NAMES = ["FRESH", "HALF_FRESH", "SPOILED"]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {idx: name for name, idx in LABEL_MAP.items()}
MODEL_NAMES = ["efficientnet_b0", "swin_t", "convnext_tiny"]
LEGACY_CHECKPOINT_FILENAMES = {
    "efficientnet_b0": ["efficientnet_b0_best.pt", "phase1_efficientnet_b0_best.pth"],
    "swin_t": ["swin_t_best.pt", "phase2_swin_t_best.pth"],
    "convnext_tiny": ["convnext_tiny_best.pt", "phase3_convnext_tiny_best.pth"],
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
