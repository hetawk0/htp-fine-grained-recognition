#!/usr/bin/env python3
"""
Smoke test: config load → dataset → forward → one train step → metrics.

Uses synthetic data by default (no downloads). Optional QuickDraw if present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loader.config import load_config
from loader.dataset_loader import build_dataloaders
from model.losses import DepressionLoss
from model.model_loader import build_model, list_models
from utils.metrics import compute_metrics
from utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Viola smoke test")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs/smoke.yaml"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(args.device)

    print("models:", list_models())
    train_loader, val_loader, _ = build_dataloaders(cfg)
    batch = next(iter(train_loader))
    images, labels = batch["image"].to(device), batch["label"].to(device)
    print(f"batch image={tuple(images.shape)} labels={tuple(labels.shape)}")

    for name in ("fbanet", "soanet", "soe_fbanet"):
        mcfg = dict(cfg)
        mcfg["model"] = dict(cfg.get("model", {}))
        mcfg["model"]["name"] = name
        if name == "soanet":
            mcfg["model"]["multi_scale_sofe"] = False
        if name == "soe_fbanet":
            mcfg["model"]["multi_scale_sofe"] = True
        model = build_model(mcfg).to(device)
        out = model(images)
        logits = out["logits"]
        assert logits.shape[0] == images.shape[0]
        nparam = sum(p.numel() for p in model.parameters())
        print(f"  {name:12s} logits={tuple(logits.shape)} params={nparam:,}")

    # One optimizer step on SOE-FBANet
    model = build_model(cfg).to(device)
    criterion = DepressionLoss(so_penalty=2.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    opt.zero_grad(set_to_none=True)
    out = model(images)
    loss = criterion(out["logits"], labels, batch["so_mask"].to(device))["loss"]
    loss.backward()
    opt.step()
    print(f"train step ok  loss={float(loss.detach()):.4f}")

    model.eval()
    with torch.no_grad():
        vbatch = next(iter(val_loader))
        vout = model(vbatch["image"].to(device))
        metrics = compute_metrics(vout["logits"].cpu(), vbatch["label"], num_classes=2)
    print(f"val metrics (1 batch): {metrics}")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
