"""Meddef-style model factory driven by config / ablation flags."""

from __future__ import annotations

from typing import Any, Dict, List

import torch.nn as nn

from model.fbanet import FBANet
from model.backbone_classifier import BackboneClassifier
from model.soanet import SOANet
from model.soe_fbanet import SOEFBANet


MODEL_REGISTRY = {
    "soanet": SOANet,
    "soe_fbanet": SOEFBANet,
    "soe-fbanet": SOEFBANet,
    "fbanet": FBANet,
    "backbone_classifier": BackboneClassifier,
}


def list_models() -> List[str]:
    return sorted(set(MODEL_REGISTRY.keys()) - {"soe-fbanet"})


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """
    Build model from a config dict.

    Expected keys under cfg['model'] (or flat cfg):
    name (backbone_classifier|fbanet|soanet|soe_fbanet), num_classes,
    backbone, pretrained,
      use_sofe, use_sosm, use_wff, use_bilevel_attention,
      use_small_object_branch, multi_scale_sofe, so_weight, ...
    """
    model_cfg = cfg.get("model", cfg)
    name = str(model_cfg.get("name", "soe_fbanet")).lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list_models()}")

    kwargs = {
        "num_classes": int(model_cfg.get("num_classes", 2)),
        "backbone": model_cfg.get("backbone", "cnn_lite"),
        "in_channels": int(model_cfg.get("in_channels", 3)),
        "pretrained": bool(model_cfg.get("pretrained", False)),
        "use_sofe": bool(model_cfg.get("use_sofe", True)),
        "use_sosm": bool(model_cfg.get("use_sosm", True)),
        "use_wff": bool(model_cfg.get("use_wff", True)),
        "use_bilevel_attention": bool(model_cfg.get("use_bilevel_attention", True)),
        "use_small_object_branch": bool(model_cfg.get("use_small_object_branch", True)),
        "use_feature_enhancement": bool(model_cfg.get("use_feature_enhancement", True)),
        "multi_scale_sofe": bool(model_cfg.get("multi_scale_sofe", name.startswith("soe"))),
        "so_weight": float(model_cfg.get("so_weight", 1.5)),
        "learnable_fusion": bool(model_cfg.get("learnable_fusion", True)),
        "semantic_dim": int(model_cfg.get("semantic_dim", 64)),
        "dropout": float(model_cfg.get("dropout", 0.3)),
    }
    return MODEL_REGISTRY[name](**kwargs)
