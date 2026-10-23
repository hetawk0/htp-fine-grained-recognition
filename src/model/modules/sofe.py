"""
Small Object Feature Enhancement (SOFE).

Inspired by Viola ideas1/ideas2 (SOANet / SOE-FBANet drafts):
multi-scale small kernels (1x1, 2x2, 3x3) after early backbone layers.
Clean reimplementation — no official public code for FBANet/SOANet.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SmallObjectEnhancement(nn.Module):
    """Single- or multi-scale small-object feature enhancement."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int | None = None,
        multi_scale: bool = True,
    ):
        super().__init__()
        mid = mid_channels or max(in_channels // 2, 32)
        self.multi_scale = multi_scale
        self.compress = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)

        if multi_scale:
            # 2x2 via padded 3x3 approx + exact 1x1 / 3x3 (PyTorch has no native 2x2 pad-even)
            self.branch_1x1 = nn.Conv2d(mid, mid, 1, bias=False)
            self.branch_2x2 = nn.Conv2d(mid, mid, 2, padding=0, bias=False)
            self.branch_3x3 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
            fuse_in = mid * 3
        else:
            self.branch_3x3 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
            fuse_in = mid

        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.compress(x)
        if self.multi_scale:
            b1 = self.branch_1x1(z)
            # Asymmetric bottom/right padding preserves size without CUDA resize backward.
            b2 = self.branch_2x2(nn.functional.pad(z, (0, 1, 0, 1)))
            b3 = self.branch_3x3(z)
            z = torch.cat([b1, b2, b3], dim=1)
        else:
            z = self.branch_3x3(z)
        out = self.fuse(z)
        return x + self.residual_scale * out


# Alias matching paper acronym
SOFE = SmallObjectEnhancement
