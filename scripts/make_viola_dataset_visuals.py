#!/usr/bin/env python3
"""Create local, ID-free dataset visuals from the frozen full DD split."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import random
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


SPLIT_PATH = ROOT / "datasets/splits/htp_depression_dd_seed42.json"
IMAGE_ROOT = ROOT / "datasets/raw/htp_depression_dd"
OUT_DIR = ROOT / "docs/results/img"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _label(name: str) -> str | None:
    stem = Path(name).stem
    if "-Nor-" in stem:
        return "Nor"
    if "-De-" in stem or "depression" in stem.lower() or "hospital" in stem.lower():
        return "De"
    return None


def _write_counts(payload: dict, out_dir: Path) -> None:
    path = out_dir / "htp_dataset_split_counts.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "class", "count"])
        for split in ("train", "val", "test"):
            for label, display in (("Nor", "Normal"), ("De", "Depressed")):
                writer.writerow([split, display, payload["counts"][split][label]])


def _write_balance_svg(payload: dict, out_dir: Path) -> None:
    width, height = 900, 450
    left, top, plot_w, plot_h = 100, 95, 740, 250
    counts = payload["counts"]
    maximum = max(counts[s][label] for s in ("train", "val", "test") for label in ("Nor", "De"))
    colors = {"Nor": "#3178a8", "De": "#d47050"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="450" y="38" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold">Dataset class counts by frozen split</text>',
        '<text x="450" y="64" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#555">Full canonical dataset • seed 42 • 70/15/15 stratified split</text>',
    ]
    for tick in range(5):
        value = maximum * tick / 4
        y = top + plot_h - plot_h * tick / 4
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#444">{value:.0f}</text>')
    group_w = plot_w / 3
    for i, split in enumerate(("train", "val", "test")):
        center = left + group_w * (i + 0.5)
        for j, label in enumerate(("Nor", "De")):
            value = counts[split][label]
            bar_w = 64
            x = center + (j - 0.5) * 76 - bar_w / 2
            bar_h = plot_h * value / maximum
            y = top + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{colors[label]}"/>')
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#222">{value}</text>')
        parts.append(f'<text x="{center:.1f}" y="{top + plot_h + 28}" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(split.title())}</text>')
    parts.extend([
        '<rect x="320" y="405" width="14" height="14" fill="#3178a8"/><text x="341" y="417" font-family="sans-serif" font-size="12">Normal</text>',
        '<rect x="435" y="405" width="14" height="14" fill="#d47050"/><text x="456" y="417" font-family="sans-serif" font-size="12">Depressed</text>',
        '</svg>',
    ])
    (out_dir / "htp_dataset_class_balance.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


def _make_montage(payload: dict, out_dir: Path, image_root: Path) -> None:
    rng = random.Random(42)
    groups: dict[str, list[Path]] = {"Nor": [], "De": []}
    for relpath in payload["train"]:
        label = _label(Path(relpath).name)
        path = image_root / relpath
        if label in groups and path.is_file():
            groups[label].append(path)
    if any(len(paths) < 4 for paths in groups.values()):
        raise ValueError("Not enough verified train-split images for a balanced montage")
    sips = shutil.which("sips")
    if sips is None:
        raise FileNotFoundError("sips is required to create self-contained montage thumbnails")

    width, height, margin, cell_w, gap = 1340, 860, 24, 300, 18
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f6f8fa"/>',
        '<text x="24" y="30" font-family="Arial,sans-serif" font-size="22" font-weight="bold" fill="#17212b">Training-set drawing examples</text>',
        '<text x="24" y="55" font-family="Arial,sans-serif" font-size="13" fill="#495461">8 randomly selected, ID-free training images; row order follows filename-derived image labels.</text>',
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        for row, label in enumerate(("Nor", "De")):
            selected = rng.sample(groups[label], 4)
            label_name = "NORMAL" if label == "Nor" else "DEPRESSED"
            safe_class = label.lower()
            total = payload["counts"]["train"][label]
            heading_y = 83 + row * 380
            parts.append(
                f'<text x="24" y="{heading_y}" font-family="Arial,sans-serif" '
                f'font-size="15" font-weight="700" letter-spacing=".7" fill="#334155">'
                f'{label_name}  •  4 DISPLAYED OF {total} TRAINING IMAGES</text>'
            )
            for col, path in enumerate(selected):
                x = margin + col * (cell_w + gap)
                y = 96 + row * 380
                sample_name = f"{safe_class}_{col + 1:02d}.png"
                sample_path = out_dir / "montage_samples" / sample_name
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, sample_path)
                thumbnail_path = Path(temp_dir) / sample_name
                subprocess.run(
                    [sips, "-Z", "550", str(path), "--out", str(thumbnail_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                image_data = base64.b64encode(thumbnail_path.read_bytes()).decode("ascii")
                xlink = f"data:image/png;base64,{image_data}"
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="330" rx="9" fill="#ffffff" stroke="#d9e0e8"/>')
                parts.append(
                    f'<image x="{x + 8}" y="{y + 8}" width="{cell_w - 16}" height="275" '
                    f'preserveAspectRatio="xMidYMid meet" style="filter:contrast(1.12)" '
                    f'href="{xlink}"/>'
                )
                parts.append(f'<text x="{x + 12}" y="{y + 310}" font-family="Arial,sans-serif" font-size="13" font-weight="600" fill="#17212b">Filename-derived class: {label_name.title()}</text>')
    parts.append("</svg>")
    svg_path = out_dir / "htp_dataset_sample_montage.svg"
    svg_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    renderer = shutil.which("rsvg-convert")
    if renderer is None:
        raise FileNotFoundError(
            "rsvg-convert is required to keep the PNG montage in sync with the SVG"
        )
    subprocess.run(
        [
            renderer,
            "-w",
            str(width),
            "-o",
            str(out_dir / "htp_dataset_sample_montage.png"),
            str(svg_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _write_scope_svg(out_dir: Path) -> None:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="360" viewBox="0 0 1050 360">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="525" y="38" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold">Classification and detection scope</text>',
        '<rect x="45" y="100" width="250" height="150" rx="14" fill="#eef6fb" stroke="#3178a8" stroke-width="2"/>',
        '<text x="170" y="145" text-anchor="middle" font-family="sans-serif" font-size="17" font-weight="bold">Input drawing</text>',
        '<text x="170" y="175" text-anchor="middle" font-family="sans-serif" font-size="13">Image-level Normal /</text>',
        '<text x="170" y="197" text-anchor="middle" font-family="sans-serif" font-size="13">Depressed label</text>',
        '<line x1="295" y1="175" x2="390" y2="175" stroke="#667085" stroke-width="3" marker-end="url(#arrow)"/>',
        '<rect x="400" y="100" width="260" height="150" rx="14" fill="#eff8f1" stroke="#4b9560" stroke-width="2"/>',
        '<text x="530" y="145" text-anchor="middle" font-family="sans-serif" font-size="17" font-weight="bold">Classifier</text>',
        '<text x="530" y="175" text-anchor="middle" font-family="sans-serif" font-size="13">Binary logits and</text>',
        '<text x="530" y="197" text-anchor="middle" font-family="sans-serif" font-size="13">class probabilities</text>',
        '<line x1="660" y1="175" x2="755" y2="175" stroke="#667085" stroke-width="3" marker-end="url(#arrow)"/>',
        '<rect x="765" y="100" width="240" height="150" rx="14" fill="#f3effa" stroke="#8064a2" stroke-width="2"/>',
        '<text x="885" y="145" text-anchor="middle" font-family="sans-serif" font-size="17" font-weight="bold">Grad-CAM</text>',
        '<text x="885" y="175" text-anchor="middle" font-family="sans-serif" font-size="13">Qualitative classifier</text>',
        '<text x="885" y="197" text-anchor="middle" font-family="sans-serif" font-size="13">activation map only</text>',
        '<rect x="250" y="285" width="550" height="48" rx="10" fill="#fff5f4" stroke="#d47050" stroke-width="2"/>',
        '<text x="525" y="315" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#8a3420">No verified HTP box labels, box-prediction head, or detection evaluator</text>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#667085"/></marker></defs>',
        '</svg>',
    ]
    (out_dir / "htp_classification_vs_detection.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=SPLIT_PATH)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    payload = json.loads(args.split.read_text(encoding="utf-8"))
    expected_counts = payload["counts"]
    physical = Counter()
    for path in args.image_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if "1_augmented" in path.parts or "_duplicates_removed" in path.parts:
            continue
        label = _label(path.name)
        if label is None:
            raise ValueError(f"Unrecognized label in canonical dataset: {path.name}")
        physical[label] += 1
    if (
        physical["Nor"] != sum(expected_counts[split]["Nor"] for split in ("train", "val", "test"))
        or physical["De"] != sum(expected_counts[split]["De"] for split in ("train", "val", "test"))
    ):
        raise ValueError("Frozen split counts do not match the canonical dataset inventory")

    observed = Counter()
    missing: list[str] = []
    for split in ("train", "val", "test"):
        for relpath in payload[split]:
            label = _label(Path(relpath).name)
            if label is None:
                raise ValueError(f"Unrecognized label in frozen split path: {relpath}")
            path = args.image_root / relpath
            if not path.is_file():
                missing.append(relpath)
            observed[(split, label)] += 1
    if missing:
        raise FileNotFoundError(f"{len(missing)} frozen split images are missing; first: {missing[0]}")
    for split in ("train", "val", "test"):
        for label in ("Nor", "De"):
            if observed[(split, label)] != payload["counts"][split][label]:
                raise ValueError(f"Manifest count mismatch for {split}/{label}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_counts(payload, args.output_dir)
    _write_balance_svg(payload, args.output_dir)
    _make_montage(payload, args.output_dir, args.image_root)
    _write_scope_svg(args.output_dir)
    print(f"Verified {sum(observed.values())} frozen-split images; wrote ID-free visuals to {args.output_dir}")


if __name__ == "__main__":
    main()
