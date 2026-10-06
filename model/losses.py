"""Losses for depression recognition + optional small-object auxiliary term."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepressionLoss(nn.Module):
    """
    Cross-entropy or focal CE + optional small-object misclassification penalty.

    ideas2: Loss = CE + so_penalty * SmallObjectLoss
    When so_mask is absent (bootstrap), falls back to CE only.
    Supports inverse-frequency ``class_weight`` for imbalance (DD ~1239:352).
    Focal mode (γ>0) down-weights easy examples — helpful when class weights
    + sampler push over-confident P(De) (τ→0.95).

    Soft LDAM (``ldam_max_m`` > 0): subtract per-class margins from true-class
    logits before CE (Cao et al. NeurIPS'19). Larger margin on minority De.
    Use alone — do not stack with focal / strong_aug (falsified here).
    """

    def __init__(
        self,
        so_penalty: float = 2.0,
        label_smoothing: float = 0.0,
        class_weight: Optional[Union[Sequence[float], torch.Tensor]] = None,
        focal_gamma: float = 0.0,
        dice_weight: float = 0.0,
        ldam_max_m: float = 0.0,
        ldam_s: float = 1.0,
        cls_num_list: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.so_penalty = so_penalty
        self.focal_gamma = float(focal_gamma)
        self.dice_weight = float(dice_weight)
        self.ldam_max_m = float(ldam_max_m)
        self.ldam_s = float(ldam_s)
        weight = None
        if class_weight is not None:
            weight = torch.as_tensor(class_weight, dtype=torch.float32)
        self.register_buffer(
            "class_weight", weight if weight is not None else None, persistent=False
        )
        self.label_smoothing = label_smoothing
        margins = None
        if self.ldam_max_m > 0.0:
            # Default DD train counts (Nor:De ≈ 1239:352) if not provided.
            counts = list(cls_num_list) if cls_num_list is not None else [1239.0, 352.0]
            c = torch.as_tensor(counts, dtype=torch.float32)
            m_list = 1.0 / torch.sqrt(torch.sqrt(c.clamp_min(1.0)))
            m_list = m_list * (self.ldam_max_m / m_list.max().clamp_min(1e-8))
            margins = m_list
        self.register_buffer("ldam_margins", margins, persistent=False)

    def _apply_ldam(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.ldam_margins is None or self.ldam_max_m <= 0.0:
            return logits
        margins = self.ldam_margins
        if margins.device != logits.device:
            margins = margins.to(logits.device)
        batch_m = margins[labels]
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        return self.ldam_s * (logits - one_hot * batch_m.unsqueeze(1))

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        so_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Softmax / CE are numerically unsafe in fp16 under AMP — always float32.
        logits = logits.float()
        weight = self.class_weight
        if weight is not None and weight.device != logits.device:
            weight = weight.to(logits.device)

        logits = self._apply_ldam(logits, labels)

        if self.focal_gamma > 0.0:
            # Focal CE: (1−p_t)^γ * CE; keep class_weight as α_t.
            log_probs = F.log_softmax(logits, dim=1)
            # Manual CE with optional label smoothing (Torch 1.10-safe).
            n_cls = logits.size(1)
            if self.label_smoothing > 0.0:
                with torch.no_grad():
                    true_dist = torch.zeros_like(log_probs)
                    true_dist.fill_(self.label_smoothing / (n_cls - 1))
                    true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)
                per = -(true_dist * log_probs).sum(dim=1)
                pt = torch.exp(-per).clamp(min=1e-8)  # approx under smoothing
            else:
                per = F.nll_loss(log_probs, labels, reduction="none")
                pt = log_probs.gather(1, labels.unsqueeze(1)).exp().squeeze(1).clamp(min=1e-8)
            focal = (1.0 - pt).pow(self.focal_gamma)
            if weight is not None:
                per = per * weight[labels]
            per = focal * per
        else:
            per = F.cross_entropy(
                logits,
                labels,
                weight=weight,
                label_smoothing=self.label_smoothing,
                reduction="none",
            )

        ce_mean = per.mean()
        dice_loss = torch.zeros((), device=logits.device)
        if self.dice_weight > 0.0 and logits.size(1) == 2:
            # Soft Dice on P(De) — mild F1 surrogate; keep weight << CE (v1 kitchen-sink hurt).
            probs = F.softmax(logits, dim=1)[:, 1]
            y = labels.float()
            inter = (probs * y).sum()
            denom = probs.sum() + y.sum()
            dice_loss = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)

        if so_mask is not None:
            so_mask = so_mask.float()
            if so_mask.sum() > 0:
                so_loss = (per * so_mask).sum() / so_mask.sum().clamp_min(1.0)
            else:
                so_loss = torch.zeros((), device=logits.device)
            total = ce_mean + self.so_penalty * so_loss
        else:
            so_loss = torch.zeros((), device=logits.device)
            total = ce_mean
        if self.dice_weight > 0.0:
            total = total + self.dice_weight * dice_loss
        return {
            "loss": total,
            "ce": ce_mean.detach(),
            "so_loss": so_loss.detach(),
            "dice_loss": dice_loss.detach(),
        }
