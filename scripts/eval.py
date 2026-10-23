#!/usr/bin/env python3
"""Evaluate a checkpoint on val/test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from loader.config import load_config
from loader.dataset_loader import build_dataloaders
from model.model_loader import build_model
from utils.checkpoint import load_checkpoint
from utils.metrics import compute_metrics, positive_class_probs, sweep_threshold
from utils.seed import set_seed


def _tta_logits(model, images: torch.Tensor, tta_mode: str) -> torch.Tensor:
    """Match ``scripts/train.py`` geometric TTA (hflip / hvflip)."""
    out = model(images)
    logits = out["logits"] if isinstance(out, dict) else out
    probs = torch.softmax(logits.float(), dim=1)
    n_views = 1
    mode = (tta_mode or "none").lower()
    if mode in ("hflip", "hvflip", "true", "1"):
        out_f = model(torch.flip(images, dims=[-1]))
        logits_f = out_f["logits"] if isinstance(out_f, dict) else out_f
        probs = probs + torch.softmax(logits_f.float(), dim=1)
        n_views += 1
    if mode == "hvflip":
        out_v = model(torch.flip(images, dims=[-2]))
        logits_v = out_v["logits"] if isinstance(out_v, dict) else out_v
        probs = probs + torch.softmax(logits_v.float(), dim=1)
        n_views += 1
        out_hv = model(torch.flip(images, dims=[-1, -2]))
        logits_hv = out_hv["logits"] if isinstance(out_hv, dict) else out_hv
        probs = probs + torch.softmax(logits_hv.float(), dim=1)
        n_views += 1
    if n_views > 1:
        probs = probs / float(n_views)
        return torch.log(probs.clamp_min(1e-8))
    return logits


@torch.no_grad()
def run_eval(
    model,
    loader,
    device,
    num_classes: int,
    return_logits: bool = False,
    tta_hflip: bool = False,
    tta_mode: str | None = None,
):
    model.eval()
    if tta_mode is None:
        tta_mode = "hflip" if tta_hflip else "none"
    all_logits, all_labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        mode = (tta_mode or "none").lower()
        if mode not in ("none", "false", "0"):
            logits = _tta_logits(model, images, mode)
        else:
            out = model(images)
            logits = out["logits"] if isinstance(out, dict) else out
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    metrics = compute_metrics(logits, labels, num_classes=num_classes)
    if return_logits:
        return metrics, logits, labels
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--sweep-threshold",
        action="store_true",
        help="Sweep τ on val (threshold_criterion from config) and report test@τ",
    )
    parser.add_argument(
        "--tta-hflip",
        action="store_true",
        help="Average softmax with horizontal flip (match train.tta_hflip)",
    )
    parser.add_argument(
        "--tta-mode",
        type=str,
        default=None,
        choices=["none", "hflip", "hvflip"],
        help="Geometric TTA mode (overrides --tta-hflip / config when set)",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default=None,
        help="τ sweep criterion: f1|f1_de|youden (default: config/ckpt)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(args.device)
    _, val_loader, test_loader = build_dataloaders(cfg)
    loader = val_loader if args.split == "val" else test_loader
    num_classes = int(cfg.get("model", {}).get("num_classes", 2))
    if args.tta_mode is not None:
        tta_mode = args.tta_mode
    elif args.tta_hflip or cfg.get("train", {}).get("tta_hflip", False):
        tta_mode = str(cfg.get("train", {}).get("tta_mode") or "hflip")
    else:
        tta_mode = "none"

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(args.checkpoint, model, map_location=device)
    metrics = run_eval(
        model, loader, device, num_classes, tta_mode=tta_mode
    )

    out = {
        "split": args.split,
        "epoch": ckpt.get("epoch"),
        "ckpt_metrics": ckpt.get("metrics"),
        "ckpt_threshold": ckpt.get("threshold"),
        "tta_mode": tta_mode,
        "tta_hflip": tta_mode in ("hflip", "hvflip", "true", "1"),
        "argmax": metrics,
    }

    # Prefer τ stored on checkpoint; --sweep-threshold forces a fresh val sweep.
    tau = None if args.sweep_threshold else ckpt.get("threshold")
    threshold_criterion = (
        args.criterion
        or (ckpt.get("metrics") or {}).get("threshold_criterion")
        or cfg.get("train", {}).get("threshold_criterion", "f1")
    )
    if tau is None and (args.sweep_threshold or num_classes == 2):
        val_m, val_logits, val_labels = run_eval(
            model,
            val_loader,
            device,
            num_classes,
            return_logits=True,
            tta_mode=tta_mode,
        )
        test_m, test_logits, test_labels = run_eval(
            model,
            test_loader,
            device,
            num_classes,
            return_logits=True,
            tta_mode=tta_mode,
        )
        out["val_argmax"] = val_m
        out["test_argmax"] = test_m
        val_probs = positive_class_probs(val_logits, num_classes=num_classes)
        if val_probs is not None:
            tau, val_at_tau = sweep_threshold(
                val_probs,
                val_labels.numpy(),
                num_classes=num_classes,
                criterion=threshold_criterion,
            )
            test_at_tau = compute_metrics(
                test_logits, test_labels, num_classes=num_classes, threshold=tau
            )
            test_at_tau["auc"] = test_m["auc"]
            test_at_tau["pr_auc"] = test_m["pr_auc"]
            out["threshold"] = {
                "tau": tau,
                "criterion": threshold_criterion,
                "val": {**val_at_tau, "auc": val_m["auc"], "pr_auc": val_m["pr_auc"]},
                "test": test_at_tau,
            }
            # Surface primary test metrics at τ for easy parsing
            out["test"] = test_at_tau
    elif tau is not None:
        test_m, test_logits, test_labels = run_eval(
            model,
            test_loader,
            device,
            num_classes,
            return_logits=True,
            tta_mode=tta_mode,
        )
        test_at_tau = compute_metrics(
            test_logits, test_labels, num_classes=num_classes, threshold=float(tau)
        )
        test_at_tau["auc"] = test_m["auc"]
        test_at_tau["pr_auc"] = test_m["pr_auc"]
        out["test_argmax"] = test_m
        out["threshold"] = {
            "tau": float(tau),
            "criterion": threshold_criterion,
            "test": test_at_tau,
        }
        out["test"] = test_at_tau

    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
