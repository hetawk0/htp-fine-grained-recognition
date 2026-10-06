"""Google QuickDraw bootstrap loader (house / tree / face)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


QUICKDRAW_CATEGORIES = ("house", "tree", "face")
QUICKDRAW_URL = (
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy"
)


class QuickDrawHTPDataset(Dataset):
    """
    Load QuickDraw 28x28 bitmaps and map categories to pseudo HTP labels.

    Default task: multi-class house/tree/face (num_classes=3) for bootstrap.
    For binary depression smoke with QuickDraw, set binary=True (random or
    heuristic labels — not clinically meaningful).

    QuickDraw strokes are often white-on-black; set invert=True in transforms
    (or invert_strokes=True here) to match HTP black-on-white.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        categories: Sequence[str] = QUICKDRAW_CATEGORIES,
        samples_per_class: int = 200,
        image_size: int = 224,
        transform=None,
        invert_strokes: bool = True,
        binary: bool = False,
        seed: int = 42,
        split_ratios: Sequence[float] = (0.7, 0.15, 0.15),
    ):
        self.root = Path(root)
        self.transform = transform
        self.invert_strokes = invert_strokes
        self.binary = binary
        self.image_size = image_size
        self.categories = list(categories)

        rng = np.random.default_rng(seed)
        records: List[Tuple[np.ndarray, int]] = []

        for ci, cat in enumerate(self.categories):
            path = self.root / f"{cat}.npy"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}. Run: python scripts/download_quickdraw.py"
                )
            arr = np.load(path)  # [N, 784] uint8
            n = min(samples_per_class, len(arr))
            idx = rng.choice(len(arr), size=n, replace=False)
            for j in idx:
                label = int(rng.integers(0, 2)) if binary else ci
                records.append((arr[j], label))

        rng.shuffle(records)
        n = len(records)
        n_train = int(n * split_ratios[0])
        n_val = int(n * split_ratios[1])
        if split == "train":
            records = records[:n_train]
        elif split in {"val", "valid", "validation"}:
            records = records[n_train : n_train + n_val]
        else:
            records = records[n_train + n_val :]

        self.images = [r[0] for r in records]
        self.labels = [r[1] for r in records]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        flat = self.images[index]
        img = flat.reshape(28, 28).astype(np.uint8)
        if self.invert_strokes:
            img = 255 - img
        pil = Image.fromarray(img, mode="L").convert("RGB")
        if self.transform is not None:
            tensor = self.transform(pil)
        else:
            pil = pil.resize((self.image_size, self.image_size))
            tensor = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float() / 255.0
        # Heuristic so_mask: low ink density ≈ "small" drawing
        ink = (flat > 0).mean()
        so = int(ink < 0.12)
        return {
            "image": tensor,
            "label": int(self.labels[index]),
            "so_mask": so,
            "path": f"quickdraw/{index}",
        }
