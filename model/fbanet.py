"""
FBANet-style Feature-enhanced Bi-Level Attention Network (clean stub).

Reference: Wang et al., Entropy 2023 — no official public code.
This is an original modular reimplementation for Viola ablation studies.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from model.backbone.cnn_backbone import build_backbone
from model.modules.bilevel_attention import BiLevelAttention
from model.modules.feature_alignment import align_feature_map
from model.modules.sofe import SOFE


class FBANet(nn.Module):
    """Baseline bi-level attention classifier (ablation: no SOA modules)."""

    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = "cnn_lite",
        in_channels: int = 3,
        pretrained: bool = False,
        use_feature_enhancement: bool = True,
        use_bilevel_attention: bool = True,
        dropout: float = 0.3,
        **kwargs: Any,
    ):
        super().__init__()
        self.backbone = build_backbone(
            backbone, in_channels=in_channels, pretrained=pretrained
        )
        c_stem = self.backbone.stem_channels
        c_out = self.backbone.out_channels

        self.use_feature_enhancement = use_feature_enhancement
        self.enhance = SOFE(
            c_stem, multi_scale=False) if use_feature_enhancement else nn.Identity()

        self.proj = nn.Sequential(
            nn.Conv2d(c_stem, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        self.attention = BiLevelAttention(c_out, enabled=use_bilevel_attention)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c_out, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        stem, deep = self.backbone(x)
        stem = self.enhance(stem)
        # Align stem to deep resolution/channels for attention path
        stem_up = align_feature_map(self.proj(stem), deep.shape[-2:])
        feat = self.attention(deep + stem_up)
        logits = self.head(self.pool(feat))
        return {"logits": logits, "features": feat}


def get_fbanet(num_classes: int = 2, **kwargs: Any) -> FBANet:
    return FBANet(num_classes=num_classes, **kwargs)
