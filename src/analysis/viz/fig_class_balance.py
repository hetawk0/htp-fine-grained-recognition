"""Class-balance and overview bar charts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .data import DDStats
from .style import COLORS, apply_pub_style, savefig


def plot_dd_class_balance(stats: DDStats, output_path: Path) -> Path:
    apply_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)

    # Overall Nor vs De
    ax = axes[0]
    labels = ["Nor", "De"]
    values = [stats.nor, stats.de]
    colors = [COLORS["nor"], COLORS["de"]]
    bars = ax.bar(labels, values, color=colors, width=0.62, edgecolor="white", linewidth=0.6)
    ax.set_ylabel("Images")
    ax.set_title("DD class balance (filename tags)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ymax = max(values + [1]) * 1.22
    ax.set_ylim(0, ymax)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.02,
            f"{val}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ratio = (stats.de / stats.nor) if stats.nor else 0.0
    ax.text(
        0.98, 0.95,
        f"De/Nor = {ratio:.3f}\n⚠ severe imbalance" if stats.nor and ratio < 0.2 else f"De/Nor = {ratio:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#9a3412" if ratio < 0.2 else "#334155",
    )

    # Per-folder breakdown
    ax = axes[1]
    folders = sorted(stats.by_folder.keys())
    nor_v = [stats.by_folder[f].get("Nor", 0) for f in folders]
    de_v = [stats.by_folder[f].get("De", 0) for f in folders]
    x = np.arange(len(folders))
    w = 0.38
    ax.bar(x - w / 2, nor_v, width=w, color=COLORS["nor"], label="Nor")
    ax.bar(x + w / 2, de_v, width=w, color=COLORS["de"], label="De")
    ax.set_xticks(x)
    ax.set_xticklabels(folders, fontsize=8)
    ax.set_ylabel("Images")
    ax.set_title("DD by folder (do not trust folder id)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")

    savefig(fig, output_path)
    return output_path


def plot_dataset_overview(
    output_path: Path,
    *,
    dd: Optional[DDStats] = None,
    quickdraw: Optional[Dict[str, int]] = None,
    sketchyscene: Optional[Dict[str, int]] = None,
    tu_berlin: Optional[Dict[str, int]] = None,
) -> Path:
    apply_pub_style()
    labels, values, colors = [], [], []
    if dd is not None and dd.total:
        labels += ["DD\nNor", "DD\nDe"]
        values += [dd.nor, dd.de]
        colors += [COLORS["nor"], COLORS["de"]]
    if quickdraw:
        for k, c in (("house", COLORS["house"]), ("tree", COLORS["tree"]), ("face", COLORS["face"])):
            if quickdraw.get(k, 0):
                labels.append(f"QD\n{k}")
                values.append(quickdraw[k])
                colors.append(c)
    if sketchyscene:
        total = sum(sketchyscene.values())
        if total:
            labels.append("Sketchy\nScene")
            values.append(total)
            colors.append(COLORS["accent"])
    if tu_berlin:
        total = sum(tu_berlin.values())
        if total:
            labels.append("TU-B\nHTP cats")
            values.append(total)
            colors.append(COLORS["person"])

    fig, ax = plt.subplots(figsize=(max(6.0, 0.85 * len(labels) + 1.5), 3.8), constrained_layout=True)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.72, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Samples")
    ax.set_title("Viola dataset overview")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    if values:
        ymax = max(values) * 1.18
        ax.set_ylim(0, ymax)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.012,
                f"{val:,}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=0 if val < 10000 else 90,
            )
    savefig(fig, output_path)
    return output_path


def plot_dd_size_intensity(stats: DDStats, output_path: Path) -> Optional[Path]:
    if not stats.widths or not stats.mean_intensities:
        return None
    apply_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), constrained_layout=True)

    ax = axes[0]
    ax.scatter(stats.widths, stats.heights, s=18, c=COLORS["muted"], alpha=0.75, edgecolors="none")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("DD image sizes (subsample)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.hist(stats.mean_intensities, bins=20, color=COLORS["nor"], edgecolor="white")
    ax.set_xlabel("Mean intensity (0–255, 64² downsample)")
    ax.set_ylabel("Count")
    ax.set_title("DD intensity distribution")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    savefig(fig, output_path)
    return output_path
