"""Sample montage grids for DD / QuickDraw / SketchyScene / TU-Berlin."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .data import pick_even, rasterize_tu_svg, sketchyscene_sample_paths
from .style import apply_pub_style, savefig


def _show(ax, arr: np.ndarray, cmap=None) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.35)
        spine.set_color("#94a3b8")
    if arr.ndim == 2:
        ax.imshow(arr, cmap=cmap or "gray", vmin=0, vmax=255)
    else:
        ax.imshow(arr)


def _load_rgb(path: Path, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img.resize(size, Image.Resampling.BILINEAR))


def _load_gray(path: Path, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img.resize(size, Image.Resampling.BILINEAR))


def plot_dd_montage(
    output_path: Path,
    nor_paths: Sequence[Path],
    de_paths: Sequence[Path],
    *,
    n_cols: int = 6,
) -> Optional[Path]:
    nor = pick_even(nor_paths, n_cols)
    de = pick_even(de_paths, n_cols)
    if not nor and not de:
        return None
    apply_pub_style()
    rows: List[Tuple[str, List[Path]]] = []
    if nor:
        rows.append(("DD Nor", nor))
    if de:
        rows.append(("DD De", de))
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.35 * n_cols, 1.55 * n_rows + 0.3))
    if n_rows == 1:
        axes = np.asarray([axes])
    for r, (title, paths) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            if c >= len(paths):
                ax.axis("off")
                continue
            try:
                _show(ax, _load_rgb(paths[c], size=(180, 128)))
            except OSError:
                ax.axis("off")
        axes[r][0].set_ylabel(title, fontsize=8, fontweight="bold", rotation=90, labelpad=4)
    fig.suptitle("DD HTP sketches — Nor vs De samples", fontsize=11, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.02, 0.01, 1.0, 0.96))
    savefig(fig, output_path)
    return output_path


def plot_quickdraw_montage(
    output_path: Path,
    npy_root: Path,
    *,
    categories: Sequence[str] = ("house", "tree", "face"),
    n_cols: int = 8,
) -> Optional[Path]:
    apply_pub_style()
    rows: List[Tuple[str, List[np.ndarray]]] = []
    rng = np.random.default_rng(42)
    for cat in categories:
        path = npy_root / f"{cat}.npy"
        if not path.exists() or path.stat().st_size < 128:
            continue
        try:
            arr = np.load(path, mmap_mode="r")
            n = min(n_cols, int(arr.shape[0]))
            if n <= 0:
                continue
            idxs = rng.choice(arr.shape[0], size=n, replace=False)
            # Invert to black-on-white for HTP-like display
            imgs = [255 - arr[i].reshape(28, 28) for i in idxs]
            rows.append((f"QuickDraw {cat}", imgs))
        except (ValueError, OSError, EOFError) as exc:
            print(f"warn: skip QuickDraw montage {cat}: {exc}")
            continue
    if not rows:
        return None
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.05 * n_cols, 1.2 * n_rows + 0.25))
    if n_rows == 1:
        axes = np.asarray([axes])
    for r, (title, imgs) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            if c >= len(imgs):
                ax.axis("off")
                continue
            _show(ax, imgs[c])
        axes[r][0].set_ylabel(title, fontsize=7.5, fontweight="bold", rotation=90, labelpad=3)
    fig.suptitle("QuickDraw house / tree / face (inverted strokes)", fontsize=11, fontweight="bold", y=0.998)
    fig.tight_layout(rect=(0.03, 0.01, 1.0, 0.96))
    savefig(fig, output_path)
    return output_path


def plot_sketchyscene_montage(
    output_path: Path,
    root: Path,
    *,
    n_cols: int = 6,
) -> Optional[Path]:
    paths = sketchyscene_sample_paths(root, n=n_cols, split="train")
    if not paths:
        return None
    apply_pub_style()
    fig, axes = plt.subplots(1, n_cols, figsize=(1.35 * n_cols, 1.7))
    axes = np.asarray(axes).reshape(-1)
    for c, ax in enumerate(axes):
        if c >= len(paths):
            ax.axis("off")
            continue
        try:
            _show(ax, _load_gray(paths[c], size=(180, 180)))
        except OSError:
            ax.axis("off")
    fig.suptitle("SketchyScene-7k train drawings", fontsize=11, fontweight="bold")
    fig.tight_layout()
    savefig(fig, output_path)
    return output_path


def plot_tu_berlin_montage(
    output_path: Path,
    svg_root: Path,
    *,
    categories: Sequence[str] = ("house", "tree", "person sitting", "person walking", "face"),
    n_cols: int = 5,
) -> Optional[Path]:
    apply_pub_style()
    rows: List[Tuple[str, List[Image.Image]]] = []
    for cat in categories:
        d = svg_root / cat
        if not d.is_dir():
            continue
        svgs = sorted(d.glob("*.svg"))
        picked = pick_even(svgs, n_cols)
        imgs = []
        for p in picked:
            im = rasterize_tu_svg(p, size=224)
            if im is not None:
                imgs.append(im)
        if imgs:
            rows.append((f"TU-B {cat}", imgs))
    if not rows:
        return None
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.15 * n_cols, 1.25 * n_rows + 0.3))
    if n_rows == 1:
        axes = np.asarray([axes])
    for r, (title, imgs) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            if c >= len(imgs):
                ax.axis("off")
                continue
            _show(ax, np.asarray(imgs[c]))
        axes[r][0].set_ylabel(title, fontsize=7, fontweight="bold", rotation=90, labelpad=3)
    fig.suptitle("TU-Berlin HTP-related categories (SVG→raster)", fontsize=11, fontweight="bold", y=0.998)
    fig.tight_layout(rect=(0.04, 0.01, 1.0, 0.96))
    savefig(fig, output_path)
    return output_path
