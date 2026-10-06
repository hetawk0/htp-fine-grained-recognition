"""Exponential moving average of model weights (eval / checkpointing)."""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn


class ModelEMA:
    """Shadow weights with optional Ultralytics-style decay ramp.

    Fixed decay (default)::
        ema = decay * ema + (1 - decay) * model

    With ``tau > 0`` (Ultralytics ``ModelEMA``)::
        d(t) = decay * (1 - exp(-t / tau))
        ema = d(t) * ema + (1 - d(t)) * model

    Early updates use a smaller effective decay so the shadow tracks the
    student; it approaches ``decay`` after roughly ``tau`` steps.
    D15 used fixed ``decay=0.999`` (tied D1). Prefer ``tau=2000`` only on a
    dedicated follow-up — do not stack onto a falsified EMA relaunch.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        tau: Optional[float] = None,
    ):
        self.decay = float(decay)
        self.tau = float(tau) if tau is not None and float(tau) > 0 else None
        self.updates = 0
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _decay_at(self) -> float:
        if self.tau is None:
            return self.decay
        return self.decay * (1.0 - math.exp(-self.updates / self.tau))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self._decay_at()
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd and v.dtype.is_floating_point:
                v.copy_(v * d + msd[k].detach() * (1.0 - d))
            elif k in msd:
                v.copy_(msd[k])

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.ema.load_state_dict(state_dict, strict=strict)
