#!/usr/bin/env python3
"""Export qualitative Grad-CAM overlays for held-out classification examples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from loader.config import load_config
from loader.dataset_loader import build_dataloaders
from model.model_loader import build_model
from utils.checkpoint import load_checkpoint
from utils.seed import set_seed


RUN_ID = "ablation_sosm_wff_ce"
RUN_DIR = ROOT / "outputs/research_phase1_phase2_seed42" / RUN_ID
PREDICTIONS = ROOT / "docs/results/img/generated_csv/matrix_ablation_sosm_wff_ce_test_predictions.csv"


def _read_examples(path: Path) -> tuple[dict[str, dict], float]:
    chosen: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: row["sample_id"])
    for row in rows:
        truth = int(row["true_class"])
        predicted = int(row["predicted_at_validation_threshold"])
        key = None
        if truth == 0 and predicted == 0:
            key = "normal_correct"
        elif truth == 1 and predicted == 1:
            key = "depressed_correct"
        elif truth == 1 and predicted == 0:
            key = "depressed_missed"
        if key and key not in chosen:
            chosen[key] = row
    required = {"normal_correct", "depressed_correct", "depressed_missed"}
    missing = required - chosen.keys()
    if missing:
        raise ValueError(f"Prediction export lacks representative cases: {sorted(missing)}")
    thresholds = {float(row["validation_threshold"]) for row in rows}
    if len(thresholds) != 1:
        raise ValueError("Prediction export contains inconsistent validation thresholds")
    return chosen, thresholds.pop()


def _views(model, image: torch.Tensor, mode: str) -> tuple[list[torch.Tensor], torch.Tensor]:
    mode = (mode or "none").lower()
    views = [image]
    if mode in ("hflip", "hvflip", "true", "1"):
        views.append(torch.flip(image, dims=[-1]))
    if mode == "hvflip":
        views.extend((torch.flip(image, dims=[-2]), torch.flip(image, dims=[-1, -2])))
    logits = []
    for view in views:
        output = model(view)
        logits.append(output["logits"] if isinstance(output, dict) else output)
    return views, torch.stack([torch.softmax(item.float(), dim=1) for item in logits]).mean(dim=0)


def _gradcam(model, image: torch.Tensor, mode: str, target_class: int) -> torch.Tensor:
    try:
        target_layer = model.backbone.deep[-1][-1]
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError("Grad-CAM target requires the final residual backbone block") from exc

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor | None] = []

    def capture(_module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("Grad-CAM target layer must return a tensor")
        view_index = len(activations)
        activations.append(output.detach())
        gradients.append(None)

        def save_gradient(gradient: torch.Tensor) -> torch.Tensor:
            gradients[view_index] = gradient.detach()
            return gradient

        output.register_hook(save_gradient)

    handle = target_layer.register_forward_hook(capture)
    try:
        views, probabilities = _views(model, image, mode)
        if len(views) != len(activations):
            raise RuntimeError("Grad-CAM did not capture every evaluation view")
        probabilities[0, target_class].backward()
    finally:
        handle.remove()

    maps = []
    for index, (activation, gradient) in enumerate(zip(activations, gradients)):
        if gradient is None:
            raise RuntimeError(f"Missing Grad-CAM gradient for view {index}")
        weights = gradient.mean(dim=(-2, -1), keepdim=True)
        heatmap = F.relu((weights * activation).sum(dim=1, keepdim=True))
        heatmap = F.interpolate(
            heatmap, size=image.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / heatmap.max().clamp_min(1e-8)
        mode_name = (mode or "none").lower()
        if index == 1 and mode_name in ("hflip", "hvflip", "true", "1"):
            heatmap = torch.flip(heatmap, dims=[-1])
        elif index == 2 and mode_name == "hvflip":
            heatmap = torch.flip(heatmap, dims=[-2])
        elif index == 3 and mode_name == "hvflip":
            heatmap = torch.flip(heatmap, dims=[-1, -2])
        maps.append(heatmap)
    combined = torch.stack(maps).mean(dim=0)
    return (combined - combined.min()) / (combined.max() - combined.min()).clamp_min(1e-8)


def _overlay(
    image: torch.Tensor, heatmap: torch.Tensor, cfg: dict
) -> Image.Image:
    mean = image.new_tensor(cfg["data"]["mean"]).view(3, 1, 1)
    std = image.new_tensor(cfg["data"]["std"]).view(3, 1, 1)
    gray = ((image[0].detach().cpu() * std + mean).clamp(0, 1).mean(dim=0).numpy() * 255).astype(np.uint8)
    base = Image.fromarray(gray, mode="L").convert("RGB")
    values = heatmap.detach().cpu().numpy().clip(0, 1)
    red = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    colors = (np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8)
    return Image.blend(base, Image.fromarray(colors, mode="RGB"), 0.38)


def _write_figure(panels: list[tuple[str, Image.Image]], output: Path, epoch: int) -> None:
    width, image_h, header_h, footer_h, gap = 400, 400, 82, 74, 18
    canvas = Image.new("RGB", (gap + len(panels) * (width + gap), header_h + image_h + footer_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((gap, 14), "Grad-CAM — SOSM + WFF, held-out test examples", fill="#17212b", font=font)
    draw.text((gap, 38), f"Best checkpoint epoch {epoch}; TTA-matched classifier class score", fill="#495461", font=font)
    for index, (caption, image) in enumerate(panels):
        x = gap + index * (width + gap)
        draw.multiline_text(
            (x, 59),
            "\n".join(textwrap.wrap(caption, width=60)),
            fill="#17212b",
            font=font,
            spacing=2,
        )
        canvas.paste(image.resize((width, image_h), Image.Resampling.BILINEAR), (x, header_h))
    draw.text(
        (gap, header_h + image_h + 20),
        "Qualitative classifier saliency only; not object boxes, detection, or a validated clinical explanation.",
        fill="#7a3524",
        font=font,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/results/img")
    args = parser.parse_args()

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback is used")
    device = torch.device(args.device)
    config_path = args.run_dir / "resolved_config.yaml"
    checkpoint_path = args.run_dir / "best.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing resolved config or best checkpoint in {args.run_dir}")

    examples, threshold = _read_examples(args.predictions)
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 42)))
    _, _, test_loader = build_dataloaders(cfg)
    if test_loader is None:
        raise ValueError("The configured test split is unavailable")

    model = build_model(cfg).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    checkpoint_threshold = checkpoint.get("threshold")
    if checkpoint_threshold is not None and abs(float(checkpoint_threshold) - threshold) > 1e-6:
        raise ValueError("Prediction threshold does not match the selected checkpoint")
    tta_mode = str(cfg.get("train", {}).get("tta_mode") or (
        "hflip" if cfg.get("train", {}).get("tta_hflip", False) else "none"
    )).lower()

    wanted = {row["sample_id"]: key for key, row in examples.items()}
    found: dict[str, tuple[str, Image.Image]] = {}
    for batch in test_loader:
        for index, path in enumerate(batch["path"]):
            sample_id = Path(path).name
            if sample_id not in wanted:
                continue
            key = wanted[sample_id]
            image = batch["image"][index : index + 1].to(device)
            model.zero_grad(set_to_none=True)
            views, probabilities = _views(model, image, tta_mode)
            probability_depressed = float(probabilities[0, 1].detach().cpu())
            row = examples[key]
            if abs(probability_depressed - float(row["probability_depressed"])) > 1e-4:
                raise ValueError(f"Recomputed score mismatch for selected {key} example")
            prediction = int(probability_depressed >= threshold)
            if prediction != int(row["predicted_at_validation_threshold"]):
                raise ValueError(f"Recomputed prediction mismatch for selected {key} example")
            del views, probabilities
            model.zero_grad(set_to_none=True)
            heatmap = _gradcam(model, image, tta_mode, prediction)
            true_label = "Normal" if int(row["true_class"]) == 0 else "Depressed"
            pred_label = "Normal" if prediction == 0 else "Depressed"
            caption = f"{key.replace('_', ' ').title()}: true {true_label} / predicted {pred_label} / P(De) {probability_depressed:.2f}"
            found[key] = (caption, _overlay(image, heatmap, cfg))
            if len(found) == len(examples):
                break
        if len(found) == len(examples):
            break
    if len(found) != len(examples):
        raise FileNotFoundError(f"Could not load all selected held-out examples: {sorted(examples.keys() - found.keys())}")

    order = ("normal_correct", "depressed_correct", "depressed_missed")
    epoch = int(checkpoint.get("epoch") or -1)
    output_path = args.output_dir / "gradcam_sosm_wff_test_examples.png"
    _write_figure([found[key] for key in order], output_path, epoch)
    metadata = {
        "run_id": RUN_ID,
        "checkpoint_epoch": epoch,
        "threshold": threshold,
        "tta_mode": tta_mode,
        "examples": list(order),
        "interpretation": "Qualitative classifier saliency only; not object detection or a validated clinical explanation.",
        "output": output_path.name,
    }
    (args.output_dir / "gradcam_sosm_wff_test_examples.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
