"""Synthetic HTP-like sketches for smoke tests (no external data)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class SyntheticHTPDataset(Dataset):
    """
    Procedural house / tree / person stick drawings on white canvas.

    Labels: randomly assigned binary depression labels for pipeline testing.
    Small-object flag: when the drawn subject occupies <1/9 of the canvas.
    """

    def __init__(
        self,
        num_samples: int = 64,
        image_size: int = 224,
        seed: int = 42,
        transform=None,
        small_object_ratio: float = 0.5,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform
        self.rng = np.random.default_rng(seed)
        self.small_object_ratio = small_object_ratio
        # Pre-draw for reproducibility across epochs
        self._cache = [self._make_one(i) for i in range(num_samples)]

    def _make_one(self, i: int) -> Tuple[Image.Image, int, int]:
        rng = np.random.default_rng(self.rng.integers(0, 1_000_000) + i)
        size = self.image_size
        img = Image.new("RGB", (size, size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        is_small = rng.random() < self.small_object_ratio
        scale = float(rng.uniform(0.12, 0.28) if is_small else rng.uniform(0.45, 0.75))
        cx = int(rng.uniform(size * 0.25, size * 0.75))
        cy = int(rng.uniform(size * 0.3, size * 0.75))
        kind = int(rng.integers(0, 3))  # 0 house, 1 tree, 2 person

        if kind == 0:
            self._draw_house(draw, cx, cy, scale, size)
        elif kind == 1:
            self._draw_tree(draw, cx, cy, scale, size)
        else:
            self._draw_person(draw, cx, cy, scale, size)

        # Weak label heuristic for smoke: small → more likely depressed
        p_dep = 0.7 if is_small else 0.3
        label = int(rng.random() < p_dep)
        return img, label, int(is_small)

    @staticmethod
    def _draw_house(draw: ImageDraw.ImageDraw, cx: int, cy: int, scale: float, size: int) -> None:
        w = int(size * scale * 0.6)
        h = int(size * scale * 0.45)
        x0, y0 = cx - w // 2, cy - h // 2
        draw.rectangle([x0, y0, x0 + w, y0 + h], outline=(0, 0, 0), width=2)
        draw.polygon(
            [(x0, y0), (x0 + w // 2, y0 - h // 2), (x0 + w, y0)],
            outline=(0, 0, 0),
        )
        # door
        dw, dh = w // 5, h // 2
        draw.rectangle([cx - dw // 2, y0 + h - dh, cx + dw // 2, y0 + h], outline=(0, 0, 0), width=2)

    @staticmethod
    def _draw_tree(draw: ImageDraw.ImageDraw, cx: int, cy: int, scale: float, size: int) -> None:
        trunk_h = int(size * scale * 0.35)
        trunk_w = max(2, int(size * scale * 0.06))
        draw.rectangle(
            [cx - trunk_w, cy, cx + trunk_w, cy + trunk_h],
            outline=(0, 0, 0),
            width=2,
        )
        r = int(size * scale * 0.22)
        draw.ellipse([cx - r, cy - r * 2, cx + r, cy], outline=(0, 0, 0), width=2)

    @staticmethod
    def _draw_person(draw: ImageDraw.ImageDraw, cx: int, cy: int, scale: float, size: int) -> None:
        r = int(size * scale * 0.08)
        draw.ellipse([cx - r, cy - 3 * r, cx + r, cy - r], outline=(0, 0, 0), width=2)
        body = int(size * scale * 0.25)
        draw.line([cx, cy - r, cx, cy + body], fill=(0, 0, 0), width=2)
        draw.line([cx, cy, cx - body // 2, cy + body // 2], fill=(0, 0, 0), width=2)
        draw.line([cx, cy, cx + body // 2, cy + body // 2], fill=(0, 0, 0), width=2)
        draw.line([cx, cy + body, cx - body // 3, cy + body + body // 2], fill=(0, 0, 0), width=2)
        draw.line([cx, cy + body, cx + body // 3, cy + body + body // 2], fill=(0, 0, 0), width=2)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img, label, so = self._cache[index]
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return {"image": img, "label": label, "so_mask": so, "path": f"synthetic/{index}"}
