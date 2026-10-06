"""Plain backbone classifier used as a matched architecture baseline."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from model.backbone.cnn_backbone import build_backbone


class BackboneClassifier(nn.Module):
    """Classify from pooled backbone features without SOE-FBANet modules."""

    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = "resnet50",
        in_channels: int = 3,
        pretrained: bool = False,
        dropout: float = 0.3,
        **kwargs: Any,
    ):
        super().__init__()
        self.backbone = build_backbone(
            backbone, in_channels=in_channels, pretrained=pretrained
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.backbone.out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, features = self.backbone(x)
        pooled = self.pool(features).flatten(1)
        logits = self.classifier(self.dropout(pooled))
        return {"logits": logits, "features": features}
