#!/usr/bin/env python3
"""Transfer backbone (+ optional SOFE) weights from a checkpoint into a DD model.

Used by D4: QuickDraw 3-class pretrain → HTP-Depression-DD 2-class fine-tune.
Does not copy the final classifier ``fc`` when shapes differ.
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
from model.model_loader import build_model
from utils.checkpoint import save_checkpoint, torch_load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Source checkpoint (e.g. QD best.pt)")
    p.add_argument("--dst-config", required=True, help="Target DD YAML")
    p.add_argument("--out", required=True, help="Output warm-start .pt path")
    p.add_argument(
        "--include-prefixes",
        type=str,
        default="backbone.,sofe.,sosm.,wff.,attention.",
        help="Comma-separated state_dict key prefixes to copy when shapes match",
    )
    args = p.parse_args()

    cfg = load_config(args.dst_config)
    model = build_model(cfg)
    src = torch_load(args.src, map_location="cpu", weights_only=False)
    src_sd = src["model"] if isinstance(src, dict) and "model" in src else src
    dst_sd = model.state_dict()
    prefixes = [x.strip() for x in args.include_prefixes.split(",") if x.strip()]

    copied, skipped = [], []
    for k, v in src_sd.items():
        if not any(k.startswith(pref) for pref in prefixes):
            continue
        if k not in dst_sd or dst_sd[k].shape != v.shape:
            skipped.append(k)
            continue
        dst_sd[k] = v
        copied.append(k)
    model.load_state_dict(dst_sd)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out, model, optimizer=None, epoch=0, metrics={"transferred_from": str(args.src)}, cfg=cfg)
    print(f"wrote {out}  copied={len(copied)}  skipped={len(skipped)}")
    if skipped:
        print("skipped (shape/key):", skipped[:20], ("..." if len(skipped) > 20 else ""))


if __name__ == "__main__":
    main()
