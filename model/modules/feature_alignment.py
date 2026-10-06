"""Deterministic spatial alignment for feature-map branches."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def align_feature_map(features: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    target_size = (int(size[0]), int(size[1]))
    if features.shape[-2:] == target_size:
        return features
    return F.interpolate(features, size=target_size, mode="nearest")
