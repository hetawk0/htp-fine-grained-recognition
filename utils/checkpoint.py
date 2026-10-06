"""Checkpoint save / load."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


def torch_load(
    path: Union[str, Path],
    map_location: str | torch.device = "cpu",
    weights_only: bool = False,
):
    """``torch.load`` compatible with older PyTorch (e.g. 1.10).

    On Torch ≥2.0, ``weights_only`` is a real parameter (default True in 2.6+);
    we pass ``False`` for full checkpoints. On 1.10, the kwarg is *not* in the
    signature but is swallowed by ``**pickle_load_args`` and then crashes
    ``Unpickler`` — so we must omit it entirely, not rely on TypeError alone.
    """
    path = str(path)
    params = inspect.signature(torch.load).parameters
    if "weights_only" in params:
        try:
            return torch.load(path, map_location=map_location, weights_only=weights_only)
        except TypeError:
            # Extremely defensive: some builds advertise the kwarg but still fail.
            return torch.load(path, map_location=map_location)
    return torch.load(path, map_location=map_location)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "metrics": metrics or {},
        "cfg": cfg or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load weights. ``strict=False`` skips mismatched heads (e.g. QD 3-cls → DD 2-cls)."""
    path = Path(path)
    ckpt = torch_load(path, map_location=map_location, weights_only=False)
    state = ckpt["model"]
    if strict:
        model.load_state_dict(state)
    else:
        model_sd = model.state_dict()
        filtered = {
            k: v
            for k, v in state.items()
            if k in model_sd and model_sd[k].shape == v.shape
        }
        missing = [k for k in model_sd if k not in filtered]
        unexpected = [k for k in state if k not in filtered]
        model.load_state_dict(filtered, strict=False)
        ckpt["_partial_load"] = {
            "loaded": len(filtered),
            "skipped_missing": missing,
            "skipped_shape_or_extra": unexpected,
        }
    if optimizer is not None and "optimizer" in ckpt and strict:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
