"""Model factory for FreshCheck classifiers."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torchvision import models

from .config import MODEL_NAMES


def _safe_weights(fetcher: Callable[[], object | None]) -> object | None:
    try:
        return fetcher()
    except Exception:
        return None


class DINOv3LinearProbe(nn.Module):
    """Frozen DINOv3 backbone with a trainable linear classification head."""

    def __init__(self, backbone: nn.Module, hidden_size: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        return self.head(pooled)

def build_dinov3_model(num_classes: int = 3, dropout: float = 0.3):
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "DINOv3 support requires transformers. Install it with `pip install transformers`."
        ) from exc

    pretrained_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    backbone = AutoModel.from_pretrained(pretrained_model_name)
    hidden_size = int(backbone.config.hidden_size)
    return DINOv3LinearProbe(backbone=backbone, hidden_size=hidden_size, num_classes=num_classes, dropout=dropout)


def build_model(model_name: str, num_classes: int = 3, dropout: float = 0.3, pretrained: bool = True):
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model: {model_name}. Expected one of {MODEL_NAMES}")
    if model_name == "dinov3_vits16":
        return build_dinov3_model(num_classes=num_classes, dropout=dropout)

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

    if model_name == "swin_t":
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

    weights = _safe_weights(lambda: models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1) if pretrained else None
    try:
        model = models.convnext_tiny(weights=weights)
    except Exception:
        model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model
