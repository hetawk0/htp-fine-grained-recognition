"""
Small Object–Semantic Mapping (SOSM).

Binds visual features to depression-related HTP semantics
(size / position / detail) via learnable prototypes + cosine gating.
Draft-aligned clean stub — not copied from unpublished paper code.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticMappingLayer(nn.Module):
    """
    Maps pooled visual features to semantic prototype space.

    Semantics (from HTP theory used in Viola drafts):
      - size: very small objects → low self-esteem / security
      - position: lower-left placement → withdrawal
      - detail: missing doors/windows / facial detail → isolation
    """

    SEMANTIC_NAMES = ("size", "position", "detail")

    def __init__(self, in_dim: int, semantic_dim: int = 64, num_semantics: int = 3):
        super().__init__()
        self.num_semantics = num_semantics
        self.proj = nn.Sequential(
            nn.Linear(in_dim, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.ReLU(inplace=True),
            nn.Linear(semantic_dim, semantic_dim),
        )
        # Learnable semantic prototypes (initialized near unit sphere)
        self.prototypes = nn.Parameter(torch.randn(num_semantics, semantic_dim) * 0.02)
        self.out = nn.Sequential(
            nn.Linear(semantic_dim + num_semantics, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat: [B, C] pooled features
        Returns:
            enriched: [B, C]
            similarities: [B, num_semantics] cosine sims to prototypes
        """
        z = F.normalize(self.proj(feat), dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        sims = z @ proto.t()  # [B, S]
        enriched = self.out(torch.cat([z, sims], dim=-1))
        return feat + enriched, sims


SOSM = SemanticMappingLayer
