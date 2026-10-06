"""Lightweight and torchvision backbones for HTP sketch classification."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class StemBlock(nn.Module):
    """Early conv stem used by SOANet / FBANet-style models."""

    def __init__(self, in_ch: int = 3, out_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 2, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNLite(nn.Module):
    """
    Small CNN backbone for smoke tests and QuickDraw bootstrap.
    Returns (stem_feat, deep_feat) for small-object / global paths.
    """

    def __init__(self, in_channels: int = 3, width: int = 64):
        super().__init__()
        self.stem = StemBlock(in_channels, width)
        self.stage2 = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 4, width * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
        )
        self.out_channels = width * 4
        self.stem_channels = width

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stem = self.stem(x)
        x = self.stage2(stem)
        deep = self.stage3(x)
        return stem, deep


class TorchvisionBackbone(nn.Module):
    """Wrap torchvision VGG16 / ResNet18/50 to expose stem + deep features."""

    def __init__(self, name: str = "resnet18", in_channels: int = 3, pretrained: bool = False):
        super().__init__()
        try:
            from torchvision import models
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for backbone=%s" % name
            ) from exc

        self.name = name
        if name in ("resnet18", "resnet50"):
            # Lab torchvision 0.11 uses pretrained=; newer uses weights=.
            if name == "resnet18":
                ctor = models.resnet18
                weights_enum = getattr(models, "ResNet18_Weights", None)
                out_ch = 512
            else:
                ctor = models.resnet50
                weights_enum = getattr(models, "ResNet50_Weights", None)
                out_ch = 2048
            if weights_enum is not None:
                weights = weights_enum.DEFAULT if pretrained else None
                net = ctor(weights=weights)
            else:
                net = ctor(pretrained=pretrained)
            if in_channels != 3:
                net.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
            self.deep = nn.Sequential(net.layer2, net.layer3, net.layer4)
            self.stem_channels = 64 if name == "resnet18" else 256
            self.out_channels = out_ch
        elif name == "vgg16":
            if hasattr(models, "VGG16_Weights"):
                weights = models.VGG16_Weights.DEFAULT if pretrained else None
                net = models.vgg16(weights=weights)
            else:
                net = models.vgg16(pretrained=pretrained)
            feats = list(net.features.children())
            if in_channels != 3:
                feats[0] = nn.Conv2d(in_channels, 64, 3, padding=1)
            # First 3 conv blocks ≈ early layers (ideas: insert SOFE after first 3 convs)
            self.stem = nn.Sequential(*feats[:10])  # through pool2
            self.deep = nn.Sequential(*feats[10:])
            self.stem_channels = 128
            self.out_channels = 512
        else:
            raise ValueError(f"Unknown torchvision backbone: {name}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stem = self.stem(x)
        deep = self.deep(stem)
        return stem, deep


BACKBONE_REGISTRY: Dict[str, str] = {
    "cnn_lite": "lite",
    "resnet18": "tv",
    "resnet50": "tv",
    "vgg16": "tv",
}


def build_backbone(
    name: str = "cnn_lite",
    in_channels: int = 3,
    pretrained: bool = False,
    width: int = 64,
) -> nn.Module:
    name = name.lower()
    if name == "cnn_lite":
        return CNNLite(in_channels=in_channels, width=width)
    if name in ("resnet18", "resnet50", "vgg16"):
        return TorchvisionBackbone(name=name, in_channels=in_channels, pretrained=pretrained)
    raise ValueError(
        f"Unknown backbone '{name}'. Choose from: {list(BACKBONE_REGISTRY)}"
    )
