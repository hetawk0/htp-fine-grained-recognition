#!/usr/bin/env python3
"""Export validation/test probabilities from saved Viola checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from utils.metrics import positive_class_probs, sweep_threshold
from utils.seed import set_seed

try:
    from eval import _tta_logits
except ImportError:
    from scripts.eval import _tta_logits


MATRIX_RUNS = [
    "baseline_resnet50_ce",
    "attention_control_ce",
    "ablation_sofe_ce",
    "ablation_sosm_ce",
    "ablation_wff_ce",
    "ablation_sofe_sosm_ce",
    "ablation_sofe_wff_ce",
    "ablation_sosm_wff_ce",
    "soefbanet_full_ce",
    "soefbanet_full_focal_gamma_0_5",
]

HISTORICAL_RUNS = {
    "historical_focal_gamma_0_5": "src/configs/archive/experiments/vabl_focal.yaml",
    "historical_focal_gamma_1_0": "src/configs/archive/experiments/vabl_focal_g1.yaml",
    "historical_class_weighted": "src/configs/archive/experiments/vabl_cw.yaml",
    "historical_focal_gamma_0_5_repeat": "src/configs/archive/experiments/vabl_focal_retry.yaml",
}


def _resolve(run_id: str, source: str) -> tuple[Path, Path]:
    output_root = ROOT / "outputs" / "research_phase1_phase2_seed42"
    if source == "matrix":
        run_dir = output_root / run_id
        if run_id == "attention_control_ce":
            run_dir = output_root / "a0_attention_control_ce"
        config = run_dir / "resolved_config.yaml"
        return config, run_dir / "best.pt"
    if source == "historical":
        config = ROOT / HISTORICAL_RUNS[run_id]
        output_dir = load_config(config).get("train", {}).get("out_dir")
        if not output_dir:
            raise ValueError(f"No train.out_dir in {config}")
        run_dir = Path(output_dir)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        return config, run_dir / "best.pt"
    raise ValueError(f"Unknown run source: {source}")


@torch.no_grad()
def _collect(model, loader, device, tta_mode: str) -> tuple[list[str], list[int], list[float]]:
    model.eval()
    sample_ids, labels, scores = [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        logits = _tta_logits(model, images, tta_mode)
        probabilities = positive_class_probs(logits.cpu(), num_classes=2)
        if probabilities is None:
            raise ValueError("Prediction export currently supports binary tasks only")
        sample_ids.extend(Path(path).name for path in batch["path"])
        labels.extend(int(value) for value in batch["label"].tolist())
        scores.extend(float(value) for value in probabilities)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate sample basenames found; refusing ambiguous sample IDs")
    return sample_ids, labels, scores


def _write_predictions(
    path: Path,
    sample_ids: list[str],
    labels: list[int],
    scores: list[float],
    threshold: float,
    run_id: str,
    split: str,
    epoch: int | None,
    tta_mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "run_id", "split", "sample_id", "true_class", "true_label",
            "probability_depressed", "predicted_at_0_5",
            "predicted_at_validation_threshold", "validation_threshold",
            "checkpoint_epoch", "tta_mode",
        ])
        for sample_id, label, score in zip(sample_ids, labels, scores):
            writer.writerow([
                run_id, split, sample_id, label,
                "Depressed" if label == 1 else "Normal",
                f"{score:.10f}", int(score >= 0.5), int(score >= threshold),
                f"{threshold:.6f}", epoch, tta_mode,
            ])


def _run_one(run_id: str, source: str, output_dir: Path, device) -> None:
    config_path, checkpoint_path = _resolve(run_id, source)
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config for {run_id}: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint for {run_id}: {checkpoint_path}")

    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 42)))
    _, val_loader, test_loader = build_dataloaders(cfg)
    if val_loader is None or test_loader is None:
        raise ValueError(f"Validation and test loaders are required for {run_id}")

    model = build_model(cfg).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    train_cfg = cfg.get("train", {})
    tta_mode = str(train_cfg.get("tta_mode") or (
        "hflip" if train_cfg.get("tta_hflip", False) else "none"
    )).lower()

    val_ids, val_labels, val_scores = _collect(model, val_loader, device, tta_mode)
    test_ids, test_labels, test_scores = _collect(model, test_loader, device, tta_mode)
    threshold = checkpoint.get("threshold")
    if threshold is None:
        threshold, _ = sweep_threshold(
            val_scores,
            val_labels,
            num_classes=2,
            criterion=str(train_cfg.get("threshold_criterion", "f1")),
        )
    threshold = float(threshold)
    epoch = checkpoint.get("epoch")

    prefix = f"{source}_{run_id}"
    exported_run_id = prefix
    for split, values in (
        ("val", (val_ids, val_labels, val_scores)),
        ("test", (test_ids, test_labels, test_scores)),
    ):
        _write_predictions(
            output_dir / f"{prefix}_{split}_predictions.csv",
            *values, threshold, exported_run_id, split, epoch, tta_mode,
        )
    metadata = {
        "run_id": run_id,
        "source": source,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": epoch,
        "threshold": threshold,
        "tta_mode": tta_mode,
        "val_samples": len(val_labels),
        "test_samples": len(test_labels),
        "sample_id_policy": "image basename; local research use only",
        "checkpoint_sha256": _sha256_file(checkpoint_path),
    }
    (output_dir / f"{prefix}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"EXPORTED {prefix}: epoch={epoch}, threshold={threshold:.3f}, "
        f"val={len(val_labels)}, test={len(test_labels)}",
        flush=True,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-set", choices=["matrix", "historical", "all"], default="all"
    )
    parser.add_argument(
        "--run-ids",
        help="Comma-separated run IDs to export; defaults to every run in --run-set",
    )
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument(
        "--output-dir", default="outputs/research_phase1_phase2_seed42/predictions"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    requested_ids = (
        {run_id.strip() for run_id in args.run_ids.split(",") if run_id.strip()}
        if args.run_ids
        else None
    )
    available_ids = set(MATRIX_RUNS) | set(HISTORICAL_RUNS)
    if requested_ids is not None:
        unknown_ids = requested_ids - available_ids
        if unknown_ids:
            raise ValueError(f"Unknown run IDs: {', '.join(sorted(unknown_ids))}")
    if args.run_set in ("matrix", "all"):
        for run_id in MATRIX_RUNS:
            if requested_ids is None or run_id in requested_ids:
                _run_one(run_id, "matrix", output_dir, device)
    if args.run_set in ("historical", "all"):
        for run_id in HISTORICAL_RUNS:
            if requested_ids is None or run_id in requested_ids:
                _run_one(run_id, "historical", output_dir, device)


if __name__ == "__main__":
    main()
