"""Triplet interaction simplified fix — channel/spatial coupling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Use explicit spatial reductions instead of adaptive pooling.
        # adaptive_max_pool2d backward is not deterministic under
        # torch.use_deterministic_algorithms(True) on CUDA in this environment.
        avg = x.mean(dim=(2, 3), keepdim=True).view(b, c)
        mx = x.amax(dim=(2, 3), keepdim=True).view(b, c)
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        w = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class TripletInteraction(nn.Module):
    """Lightweight local–global coupling (FBANet triplet-attention stand-in)."""

    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1,
                            groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw(self.dw(x))
        return self.bn(F.relu(x + y))


class BiLevelAttention(nn.Module):
    """Self-attention (channel+spatial) + triplet-style interaction."""

    def __init__(self, channels: int, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()
        self.triplet = TripletInteraction(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        x = self.channel(x)
        x = self.spatial(x)
        x = self.triplet(x)
        return x
