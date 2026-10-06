"""Lightweight dataset inventory helpers (no torch required for viz)."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class DDStats:
    total: int = 0
    nor: int = 0
    de: int = 0
    unknown: int = 0
    by_folder: Dict[str, Counter] = field(default_factory=dict)
    severity: Counter = field(default_factory=Counter)
    widths: List[int] = field(default_factory=list)
    heights: List[int] = field(default_factory=list)
    mean_intensities: List[float] = field(default_factory=list)
    nor_paths: List[Path] = field(default_factory=list)
    de_paths: List[Path] = field(default_factory=list)
    all_paths: List[Path] = field(default_factory=list)


def parse_dd_label(path: Path) -> Tuple[Optional[int], Optional[str]]:
    """Return (0=Nor, 1=De, None) and optional severity letter."""
    stem = path.stem
    severity: Optional[str] = None
    if "-Nor-" in stem:
        label = 0
        tail = stem.rsplit("-", 1)[-1]
        if len(tail) <= 2:
            severity = tail
        return label, severity
    if "-De-" in stem:
        label = 1
        tail = stem.rsplit("-", 1)[-1]
        if len(tail) <= 2:
            severity = tail
        return label, severity
    low = stem.lower()
    if "depression" in low or "hospital" in low:
        return 1, None
    return None, None


def inventory_dd(
    root: Path,
    *,
    include_augmented: bool = False,
    sample_intensity: int = 80,
) -> DDStats:
    """Walk DD folders and collect class / size / intensity stats."""
    stats = DDStats()
    if not root.is_dir():
        return stats

    paths: List[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        if not include_augmented and "1_augmented" in p.parts:
            continue
        paths.append(p)

    stats.all_paths = paths
    stats.total = len(paths)

    for i, path in enumerate(paths):
        folder = path.parent.name
        label, severity = parse_dd_label(path)
        if folder not in stats.by_folder:
            stats.by_folder[folder] = Counter()
        key = {0: "Nor", 1: "De"}.get(label, "unknown")
        stats.by_folder[folder][key] += 1
        if label == 0:
            stats.nor += 1
            stats.nor_paths.append(path)
        elif label == 1:
            stats.de += 1
            stats.de_paths.append(path)
        else:
            stats.unknown += 1
        if severity:
            stats.severity[severity] += 1

        # Size / intensity on a subsample (DD PNGs are large)
        if i < sample_intensity or label == 1:
            try:
                with Image.open(path) as im:
                    w, h = im.size
                    stats.widths.append(w)
                    stats.heights.append(h)
                    gray = im.convert("L")
                    # downsample for mean intensity
                    small = gray.resize((64, 64), Image.Resampling.BILINEAR)
                    import numpy as np

                    stats.mean_intensities.append(float(np.asarray(small).mean()))
            except OSError:
                pass

    return stats


def pick_even(paths: Sequence[Path], n: int) -> List[Path]:
    if not paths:
        return []
    if len(paths) <= n:
        return list(paths)
    import numpy as np

    idxs = np.linspace(0, len(paths) - 1, num=n, dtype=int)
    return [paths[int(i)] for i in idxs]


def quickdraw_counts(root: Path, categories: Sequence[str] = ("house", "tree", "face")) -> Dict[str, int]:
    import numpy as np

    out: Dict[str, int] = {}
    for cat in categories:
        path = root / f"{cat}.npy"
        if not path.exists() or path.stat().st_size < 128:
            out[cat] = 0
            continue
        try:
            # mmap to avoid loading full arrays; skip incomplete transfers
            arr = np.load(path, mmap_mode="r")
            out[cat] = int(arr.shape[0])
        except (ValueError, OSError, EOFError) as exc:
            print(f"warn: skip incomplete QuickDraw {path.name}: {exc}")
            out[cat] = 0
    return out


def sketchyscene_counts(root: Path) -> Dict[str, int]:
    """Count DRAWING_GT PNGs per split."""
    out: Dict[str, int] = {}
    for split in ("train", "val", "test"):
        d = root / split / "DRAWING_GT"
        if d.is_dir():
            out[split] = sum(1 for p in d.iterdir() if p.suffix.lower() == ".png")
        else:
            out[split] = 0
    return out


def sketchyscene_sample_paths(root: Path, n: int = 8, split: str = "train") -> List[Path]:
    d = root / split / "DRAWING_GT"
    if not d.is_dir():
        return []
    paths = sorted(p for p in d.iterdir() if p.suffix.lower() == ".png")
    return pick_even(paths, n)


def tu_berlin_categories(svg_root: Path, wanted: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for cat in wanted:
        d = svg_root / cat
        if d.is_dir():
            out[cat] = sum(1 for p in d.iterdir() if p.suffix.lower() == ".svg")
        else:
            out[cat] = 0
    return out


_NUM = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def rasterize_tu_svg(path: Path, size: int = 224) -> Optional["Image.Image"]:
    """Minimal SVG path rasterizer for TU-Berlin sketches (M/C commands)."""
    import numpy as np
    from PIL import ImageDraw

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    ds = re.findall(r'\bd="([^"]+)"', text)
    if not ds:
        return None

    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)

    def sample_cubic(p0, p1, p2, p3, steps=12):
        pts = []
        for t in np.linspace(0, 1, steps):
            u = 1 - t
            pt = u**3 * p0 + 3 * u**2 * t * p1 + 3 * u * t**2 * p2 + t**3 * p3
            pts.append(pt)
        return pts

    def parse_path(d: str) -> List[List[np.ndarray]]:
        """Return list of polylines (pen-up between subpaths / new M)."""
        tokens = re.findall(r"[MMCLmlZz]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", d)
        i = 0
        cur = np.array([0.0, 0.0])
        start = cur.copy()
        cmd = "M"
        strokes: List[List[np.ndarray]] = []
        stroke: List[np.ndarray] = []

        def flush():
            nonlocal stroke
            if len(stroke) >= 2:
                strokes.append(stroke)
            stroke = []

        while i < len(tokens):
            tok = tokens[i]
            if tok.isalpha():
                cmd = tok
                i += 1
                if cmd in ("M", "m"):
                    flush()
                continue
            if cmd in ("M", "m"):
                x, y = float(tokens[i]), float(tokens[i + 1])
                i += 2
                cur = cur + np.array([x, y]) if cmd == "m" else np.array([x, y])
                start = cur.copy()
                stroke = [cur.copy()]
                cmd = "l" if cmd == "m" else "L"
            elif cmd in ("L", "l"):
                x, y = float(tokens[i]), float(tokens[i + 1])
                i += 2
                cur = cur + np.array([x, y]) if cmd == "l" else np.array([x, y])
                stroke.append(cur.copy())
            elif cmd in ("C", "c"):
                nums = [float(tokens[i + k]) for k in range(6)]
                i += 6
                if cmd == "c":
                    p1 = cur + np.array(nums[0:2])
                    p2 = cur + np.array(nums[2:4])
                    p3 = cur + np.array(nums[4:6])
                else:
                    p1 = np.array(nums[0:2])
                    p2 = np.array(nums[2:4])
                    p3 = np.array(nums[4:6])
                for pt in sample_cubic(cur, p1, p2, p3):
                    stroke.append(pt)
                cur = p3
            elif cmd in ("Z", "z"):
                stroke.append(start.copy())
                cur = start.copy()
                flush()
            else:
                i += 1
        flush()
        return strokes

    all_strokes: List[np.ndarray] = []
    for d in ds:
        for stroke in parse_path(d):
            all_strokes.append(np.stack(stroke, axis=0))
    if not all_strokes:
        return None

    pts_all = np.concatenate(all_strokes, axis=0)
    mn = pts_all.min(axis=0)
    mx = pts_all.max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    scale = (size - 16) / span.max()

    for stroke in all_strokes:
        scaled = (stroke - mn) * scale + 8
        for a, b in zip(scaled[:-1], scaled[1:]):
            draw.line([tuple(a), tuple(b)], fill=0, width=2)
    return img
