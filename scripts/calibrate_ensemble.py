#!/usr/bin/env python3
"""D5/D6 — temperature scaling + multi-ckpt probability ensemble (val-only τ).

Fits temperature T on val NLL (optionally after averaging several checkpoints),
then sweeps τ on calibrated P(De). Reports val/test F1@τ without touching test
for selection.

Example:
  python scripts/calibrate_ensemble.py \\
    --config configs/experiments/d1_head_micro.yaml \\
    --checkpoints outputs/htp_depression_dd_exp_d1_head_micro/best.pt \\
                 outputs/htp_depression_dd_init/best_long_ep64.pt \\
    --tta-hflip --criterion f1 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loader.config import load_config
from loader.dataset_loader import build_dataloaders
from model.model_loader import build_model
from utils.checkpoint import load_checkpoint
from utils.metrics import compute_metrics, positive_class_probs, sweep_threshold
from utils.seed import set_seed


@torch.no_grad()
def collect_logits(model, loader, device, tta_hflip: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        out = model(images)
        logits = out["logits"] if isinstance(out, dict) else out
        if tta_hflip:
            out_f = model(torch.flip(images, dims=[-1]))
            logits_f = out_f["logits"] if isinstance(out_f, dict) else out_f
            probs = 0.5 * (torch.softmax(logits, dim=1) + torch.softmax(logits_f, dim=1))
            logits = torch.log(probs.clamp_min(1e-8))
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 80) -> float:
    """Minimize NLL w.r.t. scalar temperature T>0 (logits / T)."""
    device = torch.device("cpu")
    logits = logits.to(device).float()
    labels = labels.to(device).long()
    # Optimize log_T for positivity
    log_t = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        t = log_t.exp().clamp(min=1e-3, max=100.0)
        loss = F.cross_entropy(logits / t, labels)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        t = float(log_t.exp().clamp(min=1e-3, max=100.0).item())
    return t


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / float(temperature)


def average_logits(logit_list: Sequence[torch.Tensor]) -> torch.Tensor:
    """Average in probability space then convert back to log-probs."""
    probs = torch.stack([torch.softmax(x, dim=1) for x in logit_list], dim=0).mean(dim=0)
    return torch.log(probs.clamp_min(1e-8))


def metrics_bundle(logits: torch.Tensor, labels: torch.Tensor, tau: float, num_classes: int):
    m = compute_metrics(logits, labels, num_classes=num_classes, threshold=tau)
    argmax = compute_metrics(logits, labels, num_classes=num_classes)
    m["auc"] = argmax["auc"]
    m["pr_auc"] = argmax["pr_auc"]
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="One or more ckpts; >1 → probability ensemble (D6)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument(
        "--criterion",
        type=str,
        default="f1",
        choices=["f1", "f1_de", "youden"],
        help="τ selection on val after calibration",
    )
    parser.add_argument(
        "--no-temperature",
        action="store_true",
        help="Skip temperature fit (ensemble + τ only)",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="*",
        default=None,
        help="Optional fixed T grid to also report (in addition to fitted T)",
    )
    parser.add_argument("--out", type=str, default=None, help="Write JSON report path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _, val_loader, test_loader = build_dataloaders(cfg)
    num_classes = int(cfg.get("model", {}).get("num_classes", 2))
    tta = bool(args.tta_hflip or cfg.get("train", {}).get("tta_hflip", False))

    val_logit_sets: List[torch.Tensor] = []
    test_logit_sets: List[torch.Tensor] = []
    labels_val = labels_test = None
    for ckpt_path in args.checkpoints:
        model = build_model(cfg).to(device)
        load_checkpoint(ckpt_path, model, map_location=device)
        v_logits, v_labels = collect_logits(model, val_loader, device, tta_hflip=tta)
        t_logits, t_labels = collect_logits(model, test_loader, device, tta_hflip=tta)
        val_logit_sets.append(v_logits)
        test_logit_sets.append(t_logits)
        labels_val, labels_test = v_labels, t_labels
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    val_logits = average_logits(val_logit_sets) if len(val_logit_sets) > 1 else val_logit_sets[0]
    test_logits = average_logits(test_logit_sets) if len(test_logit_sets) > 1 else test_logit_sets[0]

    report = {
        "checkpoints": args.checkpoints,
        "n_models": len(args.checkpoints),
        "tta_hflip": tta,
        "criterion": args.criterion,
        "runs": [],
    }

    def run_one(name: str, T: Optional[float]):
        if T is None:
            v_l, t_l = val_logits, test_logits
            T_used = 1.0
        else:
            v_l = apply_temperature(val_logits, T)
            t_l = apply_temperature(test_logits, T)
            T_used = float(T)
        v_probs = positive_class_probs(v_l, num_classes=num_classes)
        assert v_probs is not None
        tau, val_at = sweep_threshold(
            v_probs, labels_val.numpy(), num_classes=num_classes, criterion=args.criterion
        )
        val_m = metrics_bundle(v_l, labels_val, tau, num_classes)
        test_m = metrics_bundle(t_l, labels_test, tau, num_classes)
        entry = {
            "name": name,
            "temperature": T_used,
            "tau": tau,
            "val": val_m,
            "test": test_m,
            "val_sweep": val_at,
        }
        report["runs"].append(entry)
        print(
            f"[{name}] T={T_used:.4f} τ={tau:.3f}  "
            f"val F1@τ={val_m['f1']:.4f} De={val_m.get('f1_c1', float('nan')):.4f}  "
            f"test F1@τ={test_m['f1']:.4f} De={test_m.get('f1_c1', float('nan')):.4f}  "
            f"test AUC={test_m['auc']:.4f}"
        )
        return entry

    # Baseline: no temperature
    run_one("raw", None)

    if not args.no_temperature:
        T_fit = fit_temperature(val_logits, labels_val)
        report["fitted_temperature"] = T_fit
        run_one("temp_fit", T_fit)

    for T in args.temperatures or []:
        run_one(f"temp_{T}", float(T))

    # Pick best by val F1@τ among runs
    best = max(report["runs"], key=lambda r: r["val"]["f1"])
    report["selected"] = {
        "rule": "max val F1@τ",
        "name": best["name"],
        "temperature": best["temperature"],
        "tau": best["tau"],
        "val_f1_tau": best["val"]["f1"],
        "test_f1_tau": best["test"]["f1"],
        "test_f1_de": best["test"].get("f1_c1"),
        "test_auc": best["test"]["auc"],
    }
    print(
        f"SELECTED {best['name']}: val F1@τ={best['val']['f1']:.4f} → "
        f"test F1@τ={best['test']['f1']:.4f} De={best['test'].get('f1_c1')} "
        f"AUC={best['test']['auc']:.4f}"
    )

    out_path = args.out
    if out_path is None:
        tag = "ens" if len(args.checkpoints) > 1 else "cal"
        out_dir = Path("outputs") / f"htp_depression_dd_{tag}_d5d6"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / "report.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
