"""Shared transforms for HTP / QuickDraw sketches."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ContrastStretch:
    """Amplify faint ink on near-white HTP canvases (DD mean≈254)."""

    def __init__(self, cutoff: float = 0.5):
        self.cutoff = cutoff

    def __call__(self, img):
        from PIL import ImageOps

        return ImageOps.autocontrast(img, cutoff=self.cutoff)


def build_transforms(
    split: str = "train",
    image_size: int = 224,
    invert: bool = False,
    augment: bool = True,
    contrast_stretch: bool = False,
    strong_aug: bool = False,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    """
    Resize protocol is YAML-driven via ``image_size``.

    - 224: classification baseline (Hatipoglu-style); tiny objects mostly lost on DD.
    - 512: default Topic A SOE-FBANet train (see docs/doc.md).
    - 768–1024: optional higher-res / future detection path.

    ``strong_aug`` adds mild ColorJitter + Affine (on-the-fly minority diversity;
    do **not** dump hospital ``1_augmented/`` into the split).

    Pass ``augment=False`` (even for split=\"train\") to rebuild an eval-like
    pipeline — used by Ultralytics-style ``close_train_aug`` late in training.
    """
    ops = []
    if invert:
        ops.append(transforms.Lambda(lambda img: TF.invert(img)))
    if contrast_stretch:
        ops.append(ContrastStretch())

    if split == "train" and augment:
        pad = 48 if strong_aug else 32
        ops.extend(
            [
                transforms.Resize((image_size + pad, image_size + pad)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20 if strong_aug else 15),
            ]
        )
        if strong_aug:
            ops.extend(
                [
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(0.05, 0.05),
                        scale=(0.9, 1.1),
                        shear=5,
                        fill=255,
                    ),
                    transforms.ColorJitter(
                        brightness=0.15, contrast=0.2, saturation=0.05, hue=0.02
                    ),
                ]
            )
    else:
        ops.append(transforms.Resize((image_size, image_size)))

    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )
    return transforms.Compose(ops)


def cfg_transforms(
    cfg: Dict[str, Any],
    split: str,
    strong_aug: Optional[bool] = None,
    augment: Optional[bool] = None,
) -> transforms.Compose:
    data = cfg.get("data", {})
    if strong_aug is None:
        strong_aug = bool(data.get("strong_aug", False))
    if augment is None:
        augment = bool(data.get("augment", True))
    return build_transforms(
        split=split,
        image_size=int(data.get("image_size", 224)),
        invert=bool(data.get("invert", False)),
        augment=bool(augment),
        contrast_stretch=bool(data.get("contrast_stretch", False)),
        strong_aug=bool(strong_aug),
        mean=tuple(data.get("mean", IMAGENET_MEAN)),
        std=tuple(data.get("std", IMAGENET_STD)),
    )
