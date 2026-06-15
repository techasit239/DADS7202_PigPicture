"""Model factory for FreshCheck classifiers."""

from __future__ import annotations

from typing import Callable

import torch.nn as nn
from torchvision import models

from .config import MODEL_NAMES


def _safe_weights(fetcher: Callable[[], object | None]) -> object | None:
    try:
        return fetcher()
    except Exception:
        return None


def build_model(model_name: str, num_classes: int = 3, dropout: float = 0.3, pretrained: bool = True):
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model: {model_name}. Expected one of {MODEL_NAMES}")

    if model_name == "efficientnet_b0":
        weights = _safe_weights(lambda: models.EfficientNet_B0_Weights.IMAGENET1K_V1) if pretrained else None
        try:
            model = models.efficientnet_b0(weights=weights)
        except Exception:
            model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        return model

    weights = _safe_weights(lambda: models.Swin_T_Weights.IMAGENET1K_V1) if pretrained else None
    try:
        model = models.swin_t(weights=weights)
    except Exception:
        model = models.swin_t(weights=None)
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model
