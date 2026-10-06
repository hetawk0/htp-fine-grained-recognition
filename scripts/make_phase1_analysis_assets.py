#!/usr/bin/env python3
"""Build local tables and plots from completed Phase 1/2 run histories."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "results" / "img"
SOURCE_DIR = ASSET_DIR / "source_artifacts"

RUNS = [
    ("baseline_resnet50_ce", "Plain ResNet-50 baseline"),
    ("attention_control_ce", "Attention-only control"),
    ("ablation_sofe_ce", "SOFE only"),
    ("ablation_sosm_ce", "SOSM only"),
    ("ablation_wff_ce", "WFF only"),
    ("ablation_sofe_sosm_ce", "SOFE + SOSM"),
    ("ablation_sofe_wff_ce", "SOFE + WFF"),
    ("ablation_sosm_wff_ce", "SOSM + WFF"),
]

RUN_DIRS = {
    "attention_control_ce": "a0_attention_control_ce",
    **{run_id: run_id for run_id, _ in RUNS if run_id != "attention_control_ce"},
}

METRICS = [
    ("accuracy", "accuracy_pct"),
    ("macro_precision", "macro_precision_pct"),
    ("macro_recall", "macro_recall_pct"),
    ("macro_f1", "macro_f1_pct"),
    ("balanced_accuracy", "balanced_accuracy_pct"),
    ("weighted_f1", "weighted_f1_pct"),
    ("auc", "roc_auc_pct"),
    ("pr_auc", "pr_auc_pct"),
    ("precision_c0", "normal_precision_pct"),
    ("recall_c0", "normal_recall_pct"),
    ("f1_c0", "normal_f1_pct"),
    ("precision_c1", "depressed_precision_pct"),
    ("recall_c1", "depressed_recall_pct"),
    ("f1_c1", "depressed_f1_pct"),
    ("sensitivity", "sensitivity_pct"),
    ("specificity", "specificity_pct"),
    ("tn", "tn"),
    ("fp", "fp"),
    ("fn", "fn"),
    ("tp", "tp"),
    ("threshold", "threshold"),
    ("loss", "test_loss"),
]


def load_runs(source_dir: Path) -> list[dict]:
    output = []
    for run_id, label in RUNS:
        history_path = source_dir / RUN_DIRS[run_id] / "history.json"
        if not history_path.is_file():
            raise FileNotFoundError(f"Missing completed run history: {history_path}")
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        test = payload.get("test")
        if not isinstance(test, dict):
            raise ValueError(f"Missing held-out test metrics in {history_path}")
        row = {
            "run_id": run_id,
            "condition": label,
            "best_epoch": payload.get("best_epoch"),
            "evidence": "completed seed-42 run; held-out test",
        }
        for source, target in METRICS:
            value = test.get(source)
            row[target] = None if value is None else (
                value if source in {"tn", "fp", "fn", "tp", "threshold", "loss"}
                else 100.0 * value
            )
        output.append(row)
    return output


def write_metrics_csv(rows: list[dict], output_dir: Path) -> None:
    path = output_dir / "phase1_phase2_test_metrics.csv"
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_history_csv(source_dir: Path, output_dir: Path) -> dict[str, list[dict]]:
    histories = {}
    path = output_dir / "phase1_phase2_training_history.csv"
    fields = [
        "run_id", "condition", "epoch", "train_loss", "val_loss",
        "train_accuracy_pct", "val_accuracy_pct", "train_f1_pct",
        "val_macro_f1_argmax_pct", "val_macro_f1_at_selected_threshold_pct",
        "val_threshold",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run_id, label in RUNS:
            payload = json.loads(
                (source_dir / RUN_DIRS[run_id] / "history.json").read_text(
                    encoding="utf-8"
                )
            )
            epochs = payload["history"]
            histories[run_id] = epochs
            for item in epochs:
                writer.writerow({
                    "run_id": run_id,
                    "condition": label,
                    "epoch": item["epoch"],
                    "train_loss": item.get("train_loss"),
                    "val_loss": item.get("val_loss"),
                    "train_accuracy_pct": _pct(item.get("train_acc")),
                    "val_accuracy_pct": _pct(item.get("val_accuracy")),
                    "train_f1_pct": _pct(item.get("train_f1")),
                    "val_macro_f1_argmax_pct": _pct(item.get("val_macro_f1")),
                    "val_macro_f1_at_selected_threshold_pct": _pct(
                        item.get("val_f1_tau")
                    ),
                    "val_threshold": item.get("val_tau"),
                })
    return histories


def _pct(value):
    return None if value is None else float(value) * 100.0


PALETTE = [
    "#2563EB", "#DC2626", "#7C3AED", "#EA580C",
    "#0891B2", "#DB2777", "#6366F1", "#14B8A6",
]
INK = "#1E293B"
MUTED = "#64748B"
GRID = "#E2E8F0"
SHORT_LABELS = {
    "baseline_resnet50_ce": "Baseline",
    "attention_control_ce": "Attention",
    "ablation_sofe_ce": "SOFE",
    "ablation_sosm_ce": "SOSM",
    "ablation_wff_ce": "WFF",
    "ablation_sofe_sosm_ce": "SOFE + SOSM",
    "ablation_sofe_wff_ce": "SOFE + WFF",
    "ablation_sosm_wff_ce": "SOSM + WFF",
}
HISTORICAL_LABELS = {
    "historical_historical_focal_gamma_0_5": "Focal gamma 0.5",
    "historical_historical_focal_gamma_1_0": "Focal gamma 1.0",
    "historical_historical_class_weighted": "Class weighted",
    "historical_historical_focal_gamma_0_5_repeat": "Focal gamma 0.5 repeat",
}


def _svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f"<title>{html.escape(title)}</title>",
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="32" y="35" text-anchor="start" '
        f'font-family="Arial, sans-serif" font-size="21" font-weight="700" '
        f'fill="{INK}">{html.escape(title)}</text>',
    ]


def _svg_text(
    elements: list[str], x: float, y: float, value: str, size: int = 12,
    anchor: str = "middle",
) -> None:
    elements.append(
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{size}" fill="{INK}">'
        f'{html.escape(str(value))}</text>'
    )


def _line_panel(
    elements: list[str], x0: float, y0: float, width: float, height: float,
    title: str, x_label: str, y_label: str,
    series: list[tuple[str, list[float] | np.ndarray, list[float] | np.ndarray, str]],
    xlim: tuple[float, float], ylim: tuple[float, float],
) -> None:
    elements.append(
        f'<rect x="{x0 + 5}" y="{y0}" width="{width - 10}" height="{height + 27}" '
        f'rx="12" fill="#ffffff" stroke="#E5EAF1"/>'
    )
    left, top, right, bottom = x0 + 62, y0 + 42, x0 + width - 22, y0 + height - 130
    plot_width, plot_height = right - left, bottom - top
    _svg_text(elements, x0 + 24, y0 + 25, title, 14, "start")
    xspan, yspan = max(xlim[1] - xlim[0], 1e-12), max(ylim[1] - ylim[0], 1e-12)
    for tick in range(6):
        fraction = tick / 5
        gx, gy = left + fraction * plot_width, top + (1 - fraction) * plot_height
        _svg_text(elements, gx, bottom + 17, f"{xlim[0] + fraction * xspan:.2f}".rstrip("0").rstrip("."), 9)
        _svg_text(elements, left - 10, gy + 3, f"{ylim[0] + fraction * yspan:.2f}".rstrip("0").rstrip("."), 9, "end")
        elements.append(f'<line x1="{gx:.1f}" y1="{top:.1f}" x2="{gx:.1f}" y2="{bottom:.1f}" stroke="#F1F4F8"/>')
        elements.append(f'<line x1="{left:.1f}" y1="{gy:.1f}" x2="{right:.1f}" y2="{gy:.1f}" stroke="{GRID}"/>')
    elements.append(f'<path d="M {left} {top} V {bottom} H {right}" fill="none" stroke="#94A3B8" stroke-width="1.2"/>')
    _svg_text(elements, (left + right) / 2, bottom + 43, x_label, 10)
    elements.append(
        f'<text x="{x0 + 14}" y="{(top + bottom) / 2}" text-anchor="middle" '
        f'transform="rotate(-90 {x0 + 14} {(top + bottom) / 2})" '
        f'font-family="Arial, sans-serif" font-size="10" fill="{INK}">{html.escape(y_label)}</text>'
    )
    for label, xs, ys, color in series:
        points = []
        for x, y in zip(xs, ys):
            px = left + (float(x) - xlim[0]) / xspan * plot_width
            py = top + plot_height - (float(y) - ylim[0]) / yspan * plot_height
            points.append(f"{px:.1f},{py:.1f}")
        if points:
            elements.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
                f'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>'
            )
    legend_y = bottom + 76
    for index, (label, _, _, color) in enumerate(series):
        col, line = index % 2, index // 2
        lx, ly = left + col * (plot_width / 2), legend_y + line * 15
        elements.append(f'<line x1="{lx:.1f}" y1="{ly - 4:.1f}" x2="{lx + 20:.1f}" y2="{ly - 4:.1f}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        _svg_text(elements, lx + 25, ly, label, 9, "start")


def _format_count(value) -> str:
    return str(int(value))


def _blend_color(color: str, fraction: float) -> str:
    base = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
    mix = max(0.0, min(0.72, fraction))
    rgb = tuple(round(255 * (1 - mix) + channel * mix) for channel in base)
    return "rgb(" + ",".join(map(str, rgb)) + ")"


def plot_confusion_matrices(rows: list[dict], output_dir: Path) -> None:
    elements = _svg_start(1440, 760, "Held-out test confusion matrices")
    for index, row in enumerate(rows):
        col, panel_row = index % 4, index // 4
        x0, y0 = col * 360, panel_row * 345 + 52
        tn, fp, fn, tp = (row[k] for k in ("tn", "fp", "fn", "tp"))
        matrix = ((tn, fp), (fn, tp))
        supports = (tn + fp, fn + tp)
        _svg_text(
            elements, x0 + 180, y0 + 24,
            f"{SHORT_LABELS.get(row['run_id'], row['condition'])}  |  {row['accuracy_pct']:.1f}% accuracy",
            12,
        )
        cell, mx, my = 82, x0 + 130, y0 + 55
        for y in range(2):
            for x in range(2):
                value = matrix[y][x]
                rate = value / supports[y] if supports[y] else 0.0
                correct = x == y
                fill = _blend_color("#0891B2" if correct else "#DC2626", rate)
                elements.append(
                    f'<rect x="{mx + x * cell}" y="{my + y * cell}" width="{cell}" '
                    f'height="{cell}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>'
                )
                _svg_text(elements, mx + x * cell + cell / 2, my + y * cell + 36, str(int(value)), 16)
                _svg_text(elements, mx + x * cell + cell / 2, my + y * cell + 56, f"{100 * rate:.1f}%", 9)
        _svg_text(elements, mx + cell / 2, my + 2 * cell + 18, "Normal", 9)
        _svg_text(elements, mx + 1.5 * cell, my + 2 * cell + 18, "Depressed", 9)
        _svg_text(elements, mx - 34, my + cell / 2 + 4, "Normal", 9)
        _svg_text(elements, mx - 34, my + 1.5 * cell + 4, "Depressed", 9)
        _svg_text(elements, mx + cell, my + 2 * cell + 35, "Predicted", 9)
        elements.append(
            f'<text x="{x0 + 16}" y="{my + cell}" text-anchor="middle" '
            f'transform="rotate(-90 {x0 + 16} {my + cell})" '
            f'font-family="Arial, sans-serif" font-size="9" fill="{INK}">Actual</text>'
        )
    _svg_text(
        elements, 720, 750,
        "Cell labels show count and row-normalized percentage; decision thresholds were selected on validation.",
        11,
    )
    elements.append("</svg>")
    (output_dir / "confusion_matrices.svg").write_text("\n".join(elements), encoding="utf-8")


def plot_training_curves(
    histories: dict[str, list[dict]], rows: list[dict], output_dir: Path
) -> None:
    elements = _svg_start(1400, 1000, "Training and validation learning curves")
    panels = [
        ("train_loss", "Training loss", "Loss"),
        ("val_loss", "Validation loss", "Loss"),
        ("train_f1", "Training macro-F1", "F1 (%)"),
        ("val_f1_tau", "Validation macro-F1 at selected threshold", "F1 (%)"),
    ]
    for panel_index, (key, title, ylabel) in enumerate(panels):
        col, panel_row = panel_index % 2, panel_index // 2
        x0, y0 = col * 700, panel_row * 475 + 45
        series = []
        for index, (run_id, epochs) in enumerate(histories.items()):
            points = [(e["epoch"], e.get(key)) for e in epochs if e.get(key) is not None]
            if not points:
                continue
            x, y = zip(*points)
            if "f1" in key:
                y = [value * 100.0 for value in y]
            best_epoch = next(item["best_epoch"] for item in rows if item["run_id"] == run_id)
            series.append((f"{SHORT_LABELS[run_id]} (best e{best_epoch})", list(x), list(y), PALETTE[index % len(PALETTE)]))
        xmax = max(x for _, xs, _, _ in series for x in xs)
        ymin = min(y for _, _, ys, _ in series for y in ys)
        ymax = max(y for _, _, ys, _ in series for y in ys)
        if ymin == ymax:
            ymax += 1
        pad = (ymax - ymin) * 0.08
        _line_panel(
            elements, x0, y0, 690, 440, title, "Epoch", ylabel, series,
            (0, xmax), (max(0, ymin - pad), ymax + pad),
        )
    elements.append("</svg>")
    (output_dir / "training_validation_curves.svg").write_text("\n".join(elements), encoding="utf-8")


def plot_factorial_heatmap(rows: list[dict], output_dir: Path) -> None:
    by_id = {row["run_id"]: row for row in rows}
    cells = {
        (0, 0, 0): ("attention_control_ce", "Attention-only"),
        (1, 0, 0): ("ablation_sofe_ce", "SOFE"),
        (0, 1, 0): ("ablation_sosm_ce", "SOSM"),
        (0, 0, 1): ("ablation_wff_ce", "WFF"),
        (1, 1, 0): ("ablation_sofe_sosm_ce", "SOFE+SOSM"),
        (1, 0, 1): ("ablation_sofe_wff_ce", "SOFE+WFF"),
        (0, 1, 1): ("ablation_sosm_wff_ce", "SOSM+WFF"),
        (1, 1, 1): (None, "Full model (pending)"),
    }
    elements = _svg_start(900, 540, "SOFE / SOSM / WFF factorial ablation")
    for panel, wff in enumerate((0, 1)):
        x0 = 20 + panel * 430
        elements.append(
            f'<rect x="{x0 + 5}" y="60" width="410" height="420" rx="12" '
            f'fill="#ffffff" stroke="#E5EAF1"/>'
        )
        _svg_text(elements, x0 + 210, 88, "WFF " + ("on" if wff else "off"), 15)
        values = np.full((2, 2), np.nan)
        labels = [["" for _ in range(2)] for _ in range(2)]
        for sofe in (0, 1):
            for sosm in (0, 1):
                run_id, title = cells[(sofe, sosm, wff)]
                labels[sofe][sosm] = title
                if run_id is not None:
                    values[sofe, sosm] = by_id[run_id]["macro_f1_pct"]
        for sofe in (0, 1):
            for sosm in (0, 1):
                value = values[sofe, sosm]
                cell_x, cell_y = x0 + 125 + sosm * 128, 122 + sofe * 128
                if np.isfinite(value):
                    level = max(0.0, min(0.72, (value - 80.0) / 12.0))
                    color = _blend_color("#2563EB", level)
                    display = f"{value:.2f}%"
                else:
                    color, display = "#F1F5F9", "TBD"
                dash = ' stroke-dasharray="5 4"' if not np.isfinite(value) else ""
                elements.append(
                    f'<rect x="{cell_x}" y="{cell_y}" width="118" height="116" rx="8" '
                    f'fill="{color}" stroke="#CBD5E1" stroke-width="1.2"'
                    f'{dash}/>'
                )
                _svg_text(elements, cell_x + 59, cell_y + 48, display, 18)
                _svg_text(elements, cell_x + 59, cell_y + 75, labels[sofe][sosm], 9)
        _svg_text(elements, x0 + 184, 114, "SOSM off", 9)
        _svg_text(elements, x0 + 312, 114, "SOSM on", 9)
        _svg_text(elements, x0 + 98, 188, "SOFE off", 9, "end")
        _svg_text(elements, x0 + 98, 316, "SOFE on", 9, "end")
    _svg_text(elements, 450, 515, "Held-out test macro-F1 (%) • pale dashed cell = result not yet verified", 11)
    elements.append("</svg>")
    (output_dir / "factorial_macro_f1_heatmap.svg").write_text("\n".join(elements), encoding="utf-8")


def plot_class_recall(rows: list[dict], output_dir: Path) -> None:
    labels = [SHORT_LABELS[row["run_id"]] for row in rows]
    normal = [row["specificity_pct"] for row in rows]
    depressed = [row["sensitivity_pct"] for row in rows]
    elements = _svg_start(1200, 610, "Class-specific recall on the held-out test set")
    left, top, plot_width, plot_height = 90, 100, 1080, 330
    group_width = plot_width / len(labels)
    for tick in range(0, 101, 20):
        y = top + plot_height * (1 - tick / 100)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="{GRID}"/>')
        _svg_text(elements, left - 12, y + 4, f"{tick}%", 10, "end")
    for index, label in enumerate(labels):
        for series_index, (value, color) in enumerate(((normal[index], PALETTE[0]), (depressed[index], PALETTE[1]))):
            bar_width = 28
            center = left + group_width * (index + 0.5)
            x = center + (series_index - 0.5) * (bar_width + 6) - bar_width / 2
            bar_height = plot_height * value / 100
            y = top + plot_height - bar_height
            elements.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="4" fill="{color}"/>')
            _svg_text(elements, x + bar_width / 2, y - 7, f"{value:.1f}", 8)
        x_label = left + group_width * (index + 0.5)
        elements.append(
            f'<text x="{x_label:.1f}" y="{top + plot_height + 48}" text-anchor="end" '
            f'transform="rotate(-28 {x_label:.1f} {top + plot_height + 48})" '
            f'font-family="Arial, sans-serif" font-size="9" fill="{INK}">{html.escape(label)}</text>'
        )
    for index, (label, color) in enumerate((
        ("Normal recall (specificity)", PALETTE[0]),
        ("Depressed recall (sensitivity)", PALETTE[1]),
    )):
        x = 410 + index * 250
        elements.append(f'<rect x="{x}" y="66" width="13" height="13" rx="3" fill="{color}"/>')
        _svg_text(elements, x + 21, 77, label, 10, "start")
    _svg_text(elements, 600, 590, "Thresholds selected using validation data; bars report held-out test recall.", 10)
    elements.append("</svg>")
    (output_dir / "class_recall_comparison.svg").write_text("\n".join(elements), encoding="utf-8")


def read_prediction_csv(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty predictions file: {path}")
    labels = np.array([int(row["true_class"]) for row in rows], dtype=int)
    scores = np.array(
        [float(row["probability_depressed"]) for row in rows], dtype=float
    )
    threshold = float(rows[0]["validation_threshold"])
    if any(float(row["validation_threshold"]) != threshold for row in rows):
        raise ValueError(f"Inconsistent validation threshold in {path}")
    return {
        "run_id": rows[0]["run_id"],
        "source": rows[0]["split"],
        "labels": labels,
        "scores": scores,
        "threshold": threshold,
        "sample_ids": [row["sample_id"] for row in rows],
    }


def _ece(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    error = 0.0
    for index in range(bins):
        selected = (scores >= edges[index]) & (
            scores <= edges[index + 1] if index == bins - 1
            else scores < edges[index + 1]
        )
        if np.any(selected):
            confidence = float(np.mean(scores[selected]))
            observed = float(np.mean(labels[selected]))
            error += np.sum(selected) / total * abs(confidence - observed)
    return error


def _binary_curves(labels: np.ndarray, scores: np.ndarray) -> tuple:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if not positives or not negatives:
        raise ValueError("ROC and precision-recall curves require both classes")
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    cumulative_tp = np.cumsum(sorted_labels == 1)
    cumulative_fp = np.cumsum(sorted_labels == 0)
    changes = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    roc_tp = cumulative_tp[changes]
    roc_fp = cumulative_fp[changes]
    roc_fpr = np.r_[0.0, roc_fp / negatives]
    roc_tpr = np.r_[0.0, roc_tp / positives]
    recall = cumulative_tp / positives
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    pr_recall = np.r_[0.0, recall]
    pr_precision = np.r_[1.0, precision]
    roc_auc = float(np.trapezoid(roc_tpr, roc_fpr))
    average_precision = float(np.sum(np.diff(pr_recall) * pr_precision[1:]))
    return roc_fpr, roc_tpr, pr_recall, pr_precision, roc_auc, average_precision


def _thresholded_metrics(labels: np.ndarray, predicted: np.ndarray) -> tuple:
    tn = int(np.sum((labels == 0) & (predicted == 0)))
    fp = int(np.sum((labels == 0) & (predicted == 1)))
    fn = int(np.sum((labels == 1) & (predicted == 0)))
    tp = int(np.sum((labels == 1) & (predicted == 1)))
    class_precision = np.array([
        tn / (tn + fn) if tn + fn else 0.0,
        tp / (tp + fp) if tp + fp else 0.0,
    ])
    class_recall = np.array([
        tn / (tn + fp) if tn + fp else 0.0,
        tp / (tp + fn) if tp + fn else 0.0,
    ])
    return tn, fp, fn, tp, class_precision, class_recall


def _calibration_points(
    labels: np.ndarray, scores: np.ndarray, bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    means, fractions = [], []
    for index in range(bins):
        selected = (scores >= edges[index]) & (
            scores <= edges[index + 1] if index == bins - 1
            else scores < edges[index + 1]
        )
        if np.any(selected):
            means.append(float(np.mean(scores[selected])))
            fractions.append(float(np.mean(labels[selected])))
    return np.asarray(means), np.asarray(fractions)


def _prediction_metrics(item: dict) -> dict:
    labels = item["labels"]
    scores = item["scores"]
    threshold = item["threshold"]
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp, class_precision, class_recall = _thresholded_metrics(
        labels, predicted
    )
    _, _, _, _, roc_auc, average_precision = _binary_curves(labels, scores)
    return {
        "run_id": item["run_id"],
        "split": item["source"],
        "sample_count": len(labels),
        "accuracy_pct": 100 * float(np.mean(labels == predicted)),
        "macro_f1_pct": 100 * np.mean([
            2 * p * r / (p + r) if p + r else 0
            for p, r in zip(class_precision, class_recall)
        ]),
        "normal_precision_pct": 100 * class_precision[0],
        "normal_recall_pct": 100 * class_recall[0],
        "depressed_precision_pct": 100 * class_precision[1],
        "depressed_recall_pct": 100 * class_recall[1],
        "roc_auc_pct": 100 * roc_auc,
        "pr_auc_pct": 100 * average_precision,
        "brier_score": float(np.mean((scores - labels) ** 2)),
        "ece_10_bins": _ece(labels, scores),
        "threshold": threshold,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def export_prediction_metrics(output_dir: Path) -> list[dict]:
    prediction_dir = output_dir / "generated_csv"
    test_predictions = {
        item["run_id"]: item
        for item in (
            read_prediction_csv(path)
            for path in sorted(prediction_dir.glob("*_test_predictions.csv"))
        )
    }
    val_predictions = {
        item["run_id"]: item
        for item in (
            read_prediction_csv(path)
            for path in sorted(prediction_dir.glob("*_val_predictions.csv"))
        )
    }
    if not test_predictions:
        return []
    rows = [_prediction_metrics(item) for item in test_predictions.values()]
    with (output_dir / "prediction_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_prediction_curves(list(test_predictions.values()), output_dir)
    plot_threshold_curves(list(val_predictions.values()), output_dir)
    plot_calibration(list(test_predictions.values()), output_dir)
    return rows


def _prediction_label(run_id: str) -> str:
    if run_id in HISTORICAL_LABELS:
        return HISTORICAL_LABELS[run_id]
    for candidate, label in RUNS:
        if run_id.endswith(candidate):
            return SHORT_LABELS[candidate]
    return run_id.replace("_", " ")


def plot_prediction_curves(predictions: list[dict], output_dir: Path) -> None:
    matrix_items = [item for item in predictions if item["run_id"].startswith("matrix_")]
    historical_items = [
        item for item in predictions if item["run_id"].startswith("historical_")
    ]
    for items, suffix, title_prefix in (
        (matrix_items, "matrix", "Component ablation matrix"),
        (historical_items, "historical", "Historical model recipes"),
    ):
        if not items:
            continue
        elements = _svg_start(1500, 730, f"{title_prefix}: held-out test curves")
        roc_series, pr_series = [], []
        for item in items:
            label = _prediction_label(item["run_id"])
            fpr, tpr, recall, precision, roc_auc, pr_auc = _binary_curves(
                item["labels"], item["scores"]
            )
            color = PALETTE[len(roc_series) % len(PALETTE)]
            roc_series.append((f"{label} (AUC {roc_auc:.3f})", fpr, tpr, color))
            pr_series.append((f"{label} (AP {pr_auc:.3f})", recall, precision, color))
        roc_series.append(("chance", [0, 1], [0, 1], "#888888"))
        _line_panel(elements, 0, 60, 740, 630, "ROC", "False positive rate",
                    "True positive rate", roc_series, (0, 1), (0, 1))
        _line_panel(elements, 750, 60, 740, 630, "Precision-recall", "Recall",
                    "Precision", pr_series, (0, 1), (0, 1))
        elements.append("</svg>")
        (output_dir / f"roc_pr_curves_{suffix}.svg").write_text(
            "\n".join(elements), encoding="utf-8"
        )


def plot_threshold_curves(predictions: list[dict], output_dir: Path) -> None:
    matrix_items = [item for item in predictions if item["run_id"].startswith("matrix_")]
    if not matrix_items:
        return
    thresholds = np.linspace(0.01, 0.99, 99)
    accuracy_series, f1_series = [], []
    for item in matrix_items:
        labels = item["labels"]
        scores = item["scores"]
        accuracy, macro_f1 = [], []
        for threshold in thresholds:
            predicted = (scores >= threshold).astype(int)
            accuracy.append(float(np.mean(labels == predicted)) * 100)
            _, _, _, _, p, r = _thresholded_metrics(labels, predicted)
            f1s = [2 * a * b / (a + b) if a + b else 0 for a, b in zip(p, r)]
            macro_f1.append(float(np.mean(f1s)) * 100)
        label = _prediction_label(item["run_id"])
        color = PALETTE[len(accuracy_series) % len(PALETTE)]
        accuracy_series.append((label, thresholds, accuracy, color))
        f1_series.append((label, thresholds, macro_f1, color))
    elements = _svg_start(1500, 730, "Threshold sensitivity on validation set; test labels not used")
    _line_panel(elements, 0, 60, 740, 630, "Accuracy vs threshold",
                "Threshold", "Accuracy (%)", accuracy_series, (0, 1), (0, 100))
    _line_panel(elements, 750, 60, 740, 630, "Macro-F1 vs threshold",
                "Threshold", "Macro-F1 (%)", f1_series, (0, 1), (0, 100))
    elements.append("</svg>")
    (output_dir / "threshold_sensitivity_validation.svg").write_text(
        "\n".join(elements), encoding="utf-8"
    )


def plot_calibration(predictions: list[dict], output_dir: Path) -> None:
    matrix_items = [item for item in predictions if item["run_id"].startswith("matrix_")]
    historical_items = [
        item for item in predictions if item["run_id"].startswith("historical_")
    ]
    for items, suffix, title in (
        (matrix_items, "matrix", "Seed-42 matrix: held-out test calibration"),
        (historical_items, "historical", "Historical recipes: held-out test calibration"),
    ):
        if not items:
            continue
        cols = 4
        rows = int(np.ceil(len(items) / cols))
        elements = _svg_start(1400, rows * 300 + 80, title)
        for index, item in enumerate(items):
            col, panel_row = index % cols, index // cols
            x0, y0 = col * 350, panel_row * 300 + 55
            confidence, observed = _calibration_points(item["labels"], item["scores"])
            metrics = _prediction_metrics(item)
            _svg_text(
                elements, x0 + 175, y0 + 18,
                f"{_prediction_label(item['run_id'])}  |  Brier {metrics['brier_score']:.3f}  |  ECE {metrics['ece_10_bins']:.3f}",
                9,
            )
            _line_panel(
                elements, x0, y0 + 25, 340, 250, "Reliability", "Mean predicted probability",
                "Observed positive fraction",
                [("Ideal", [0, 1], [0, 1], "#94A3B8"),
                 ("Observed", confidence, observed, PALETTE[index % len(PALETTE)])],
                (0, 1), (0, 1),
            )
        elements.append("</svg>")
        (output_dir / f"calibration_{suffix}.svg").write_text(
            "\n".join(elements), encoding="utf-8"
        )


def write_status_csv(output_dir: Path) -> None:
    rows = [
        ("baseline_resnet50_ce", "Plain ResNet-50 baseline", "complete"),
        ("attention_control_ce", "Attention-only control", "complete; reused verified a0 run"),
        ("ablation_sofe_ce", "SOFE only", "complete"),
        ("ablation_sosm_ce", "SOSM only", "complete"),
        ("ablation_wff_ce", "WFF only", "complete"),
        ("ablation_sofe_sosm_ce", "SOFE + SOSM", "complete"),
        ("ablation_sofe_wff_ce", "SOFE + WFF", "complete"),
        ("ablation_sosm_wff_ce", "SOSM + WFF", "complete"),
        ("soefbanet_full_ce", "Full SOE-FBANet; cross-entropy", "last confirmed running at 2026-10-06 18:18 HKT; current server refresh timed out, state unknown"),
        ("soefbanet_full_focal_gamma_0_5", "Full SOE-FBANet; focal gamma 0.5", "last confirmed queued at 2026-10-06 18:18 HKT; current server refresh timed out, state unknown"),
    ]
    with (output_dir / "phase1_phase2_run_status.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["run_id", "condition", "status"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        default="docs/results/img/source_artifacts",
        help="Run folders containing history.json",
    )
    parser.add_argument("--output-dir", default="docs/results/img")
    args = parser.parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    if not source_dir.is_absolute():
        source_dir = ROOT / source_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_runs(source_dir)
    histories = write_history_csv(source_dir, output_dir)
    write_metrics_csv(rows, output_dir)
    write_status_csv(output_dir)
    plot_confusion_matrices(rows, output_dir)
    plot_training_curves(histories, rows, output_dir)
    plot_factorial_heatmap(rows, output_dir)
    plot_class_recall(rows, output_dir)
    export_prediction_metrics(output_dir)
    for path in sorted(output_dir.glob("*")):
        if path.is_file():
            print(f"{path.name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
