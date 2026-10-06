"""
Weighted Feature Fusion (WFF).

Emphasizes small-object features while preserving global context.
Default small-object weight 1.5 per Viola ideas drafts; learnable gate optional.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.modules.feature_alignment import align_feature_map


class WeightedFeatureFusion(nn.Module):
    def __init__(
        self,
        channels: int,
        so_weight: float = 1.5,
        learnable: bool = True,
    ):
        super().__init__()
        self.so_weight = so_weight
        self.learnable = learnable
        if learnable:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels * 2, channels),
                nn.ReLU(inplace=True),
                nn.Linear(channels, 2),
                nn.Softmax(dim=-1),
            )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        global_feat: torch.Tensor,
        so_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            global_feat, so_feat: [B, C, H, W] (spatially aligned)
        """
        if global_feat.shape[-2:] != so_feat.shape[-2:]:
            so_feat = align_feature_map(so_feat, global_feat.shape[-2:])
        if self.learnable:
            w = self.gate(torch.cat([global_feat, so_feat], dim=1))  # [B, 2]
            g = w[:, 0].view(-1, 1, 1, 1)
            s = w[:, 1].view(-1, 1, 1, 1) * self.so_weight
            fused_in = torch.cat([g * global_feat, s * so_feat], dim=1)
        else:
            fused_in = torch.cat(
                [global_feat, self.so_weight * so_feat], dim=1
            )
        return self.fuse(fused_in)


WFF = WeightedFeatureFusion
