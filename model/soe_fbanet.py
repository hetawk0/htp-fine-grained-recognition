"""
SOE-FBANet — Small Object-Enhanced Bi-Level Attention Network.

Combines SOFE (multi-scale) + SOSM + learnable WFF on an FBANet-style
bi-level attention backbone. Ablations mirror ideas2 Table 3.

Clean modular implementation for Viola; not official paper code.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from model.backbone.cnn_backbone import build_backbone
from model.modules.bilevel_attention import BiLevelAttention
from model.modules.feature_alignment import align_feature_map
from model.modules.sofe import SOFE
from model.modules.sosm import SOSM
from model.modules.wff import WFF


class SOEFBANet(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = "cnn_lite",
        in_channels: int = 3,
        pretrained: bool = False,
        use_sofe: bool = True,
        use_sosm: bool = True,
        use_wff: bool = True,
        use_bilevel_attention: bool = True,
        use_small_object_branch: bool = True,
        multi_scale_sofe: bool = True,
        so_weight: float = 1.5,
        learnable_fusion: bool = True,
        semantic_dim: int = 64,
        dropout: float = 0.3,
        **kwargs: Any,
    ):
        super().__init__()
        self.use_sofe = use_sofe
        self.use_sosm = use_sosm
        self.use_wff = use_wff
        self.use_small_object_branch = use_small_object_branch

        self.backbone = build_backbone(
            backbone, in_channels=in_channels, pretrained=pretrained
        )
        c_stem = self.backbone.stem_channels
        c_out = self.backbone.out_channels

        self.sofe = (
            SOFE(c_stem, multi_scale=multi_scale_sofe) if use_sofe else nn.Identity()
        )
        self.proj_stem = nn.Sequential(
            nn.Conv2d(c_stem, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

        if use_wff and use_small_object_branch:
            self.so_branch = nn.Sequential(
                nn.Conv2d(c_stem, c_out, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )
        else:
            self.so_branch = None

        self.attention = BiLevelAttention(c_out, enabled=use_bilevel_attention)
        self.wff = (
            WFF(c_out, so_weight=so_weight, learnable=learnable_fusion)
            if use_wff
            else None
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sosm = SOSM(
            c_out, semantic_dim=semantic_dim) if use_sosm else None
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c_out, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        stem, deep = self.backbone(x)
        stem = self.sofe(stem)

        global_feat = self.attention(
            deep + align_feature_map(self.proj_stem(stem), deep.shape[-2:])
        )

        if self.wff is not None:
            so_feat = (
                self.so_branch(stem)
                if self.so_branch is not None
                else self.proj_stem(stem)
            )
            if so_feat.shape[-2:] != global_feat.shape[-2:]:
                so_feat = align_feature_map(so_feat, global_feat.shape[-2:])
            fused = self.wff(global_feat, so_feat)
        else:
            fused = global_feat

        pooled = self.pool(fused).flatten(1)
        sims = None
        if self.sosm is not None:
            pooled, sims = self.sosm(pooled)

        logits = self.fc(self.dropout(pooled))
        out: Dict[str, torch.Tensor] = {"logits": logits, "features": fused}
        if sims is not None:
            out["semantic_sims"] = sims
        return out


def get_soe_fbanet(num_classes: int = 2, **kwargs: Any) -> SOEFBANet:
    return SOEFBANet(num_classes=num_classes, **kwargs)
