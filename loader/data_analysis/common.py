"""Shared audit helpers (path integrity, hashes, image stats) — no medical ImageFolder assumptions."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ImageStatSample:
    path: Path
    width: int
    height: int
    mode: str
    mean: float
    std: float
    ink_frac: float  # fraction of pixels darker than ink_threshold
    lap_var: float  # Laplacian-ish sharpness via numpy (no cv2)


@dataclass
class DupReport:
    exact_sets: int = 0
    exact_extra_files: int = 0
    near_sets: int = 0
    examples: List[str] = field(default_factory=list)


def iter_images(root: Path, *, skip_dirs: Optional[set] = None) -> List[Path]:
    skip_dirs = skip_dirs or set()
    out: List[Path] = []
    if not root.is_dir():
        return out
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        if any(part in skip_dirs for part in p.parts):
            continue
        out.append(p)
    return out


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def average_hash(path: Path, hash_size: int = 8) -> Optional[str]:
    """Simple aHash (no imagehash dep): mean-thresholded 8x8 grayscale."""
    try:
        with Image.open(path) as im:
            gray = im.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
            arr = np.asarray(gray, dtype=np.float32)
        mean = float(arr.mean())
        bits = (arr >= mean).astype(np.uint8).flatten()
        # pack to hex
        val = 0
        for b in bits:
            val = (val << 1) | int(b)
        width = (hash_size * hash_size + 3) // 4
        return f"{val:0{width}x}"
    except OSError:
        return None


def check_exact_and_near_duplicates(
    paths: Sequence[Path],
    *,
    near: bool = True,
    example_limit: int = 5,
) -> DupReport:
    exact: Dict[str, List[Path]] = defaultdict(list)
    near_map: Dict[str, List[Path]] = defaultdict(list)
    for p in paths:
        try:
            exact[md5_file(p)].append(p)
        except OSError:
            continue
        if near:
            ah = average_hash(p)
            if ah is not None:
                near_map[ah].append(p)

    report = DupReport()
    for h, group in exact.items():
        if len(group) > 1:
            report.exact_sets += 1
            report.exact_extra_files += len(group) - 1
            if len(report.examples) < example_limit:
                rels = ", ".join(str(g.name) for g in group[:4])
                report.examples.append(f"exact md5={h[:8]}… → {rels}")

    # Near-dup sets that are not already exact duplicates
    exact_path_groups = {frozenset(g) for g in exact.values() if len(g) > 1}
    for h, group in near_map.items():
        if len(group) <= 1:
            continue
        key = frozenset(group)
        if key in exact_path_groups:
            continue
        # also skip if all pairs share exact hashes already counted
        hashes = {md5_file(p) for p in group}
        if len(hashes) == 1:
            continue
        report.near_sets += 1
        if len(report.examples) < example_limit:
            rels = ", ".join(str(g.name) for g in group[:4])
            report.examples.append(f"near aHash={h} → {rels}")
    return report


def _laplacian_var(gray: np.ndarray) -> float:
    """3x3 Laplacian variance (blur proxy) without OpenCV."""
    g = gray.astype(np.float64)
    # kernel [[0,1,0],[1,-4,1],[0,1,0]]
    if g.ndim != 2 or g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    center = g[1:-1, 1:-1]
    lap = (
        g[:-2, 1:-1]
        + g[2:, 1:-1]
        + g[1:-1, :-2]
        + g[1:-1, 2:]
        - 4.0 * center
    )
    return float(lap.var())


def sample_image_stats(
    paths: Sequence[Path],
    *,
    max_n: int = 120,
    prefer: Optional[Sequence[Path]] = None,
    seed: int = 42,
    thumb: int = 128,
    ink_threshold: int = 240,
) -> List[ImageStatSample]:
    """Subsample large canvases; always include `prefer` paths (e.g. minority class)."""
    prefer = list(prefer or [])
    chosen: List[Path] = []
    seen = set()
    for p in prefer:
        if p not in seen:
            chosen.append(p)
            seen.add(p)
    remaining = [p for p in paths if p not in seen]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    for p in remaining:
        if len(chosen) >= max_n:
            break
        chosen.append(p)

    out: List[ImageStatSample] = []
    for p in chosen:
        try:
            with Image.open(p) as im:
                w, h = im.size
                mode = im.mode
                gray = im.convert("L")
                small = gray.resize((thumb, thumb), Image.Resampling.BILINEAR)
                arr = np.asarray(small, dtype=np.float32)
                ink = float((arr < ink_threshold).mean())
                lap = _laplacian_var(arr)
                out.append(
                    ImageStatSample(
                        path=p,
                        width=w,
                        height=h,
                        mode=mode,
                        mean=float(arr.mean()),
                        std=float(arr.std()),
                        ink_frac=ink,
                        lap_var=lap,
                    )
                )
        except OSError:
            continue
    return out


def summarize_stats(samples: Sequence[ImageStatSample]) -> Dict[str, object]:
    if not samples:
        return {"n": 0}
    sizes = Counter((s.width, s.height) for s in samples)
    modes = Counter(s.mode for s in samples)
    means = [s.mean for s in samples]
    stds = [s.std for s in samples]
    inks = [s.ink_frac for s in samples]
    laps = [s.lap_var for s in samples]
    aspects = [s.width / s.height if s.height else 0.0 for s in samples]
    return {
        "n": len(samples),
        "unique_sizes": len(sizes),
        "size_top": sizes.most_common(5),
        "modes": dict(modes),
        "mean_intensity": float(np.mean(means)),
        "std_intensity_of_means": float(np.std(means)),
        "mean_contrast": float(np.mean(stds)),
        "mean_ink_frac": float(np.mean(inks)),
        "mean_laplacian_var": float(np.mean(laps)),
        "aspect_mean": float(np.mean(aspects)),
        "aspect_min": float(np.min(aspects)),
        "aspect_max": float(np.max(aspects)),
        "brightness_warn": (
            "low" if np.mean(means) < 50 else ("high/white-bg" if np.mean(means) > 200 else "ok")
        ),
        "sharpness_warn": "blurry?" if np.mean(laps) < 5 else "ok",
    }


def check_corrupted(paths: Sequence[Path]) -> List[str]:
    bad: List[str] = []
    for p in paths:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:  # verify() leaves file unusable
                im.load()
        except Exception as exc:  # noqa: BLE001 — audit wants any failure
            bad.append(f"{p}: {exc}")
    return bad


def stratified_split_indices(
    labels: Sequence[int],
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Per-class shuffle split (preferred for imbalanced DD)."""
    by_class: Dict[int, List[int]] = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[int(y)].append(i)
    rng = random.Random(seed)
    out: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for idxs in by_class.values():
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        out["train"].extend(idxs[:n_train])
        out["val"].extend(idxs[n_train : n_train + n_val])
        out["test"].extend(idxs[n_train + n_val :])
    return out


def random_split_indices(
    n: int,
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Match current HTPFolderDataset._split (global shuffle — NOT stratified)."""
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": idx[:n_train],
        "val": idx[n_train : n_train + n_val],
        "test": idx[n_train + n_val :],
    }


def split_leakage(
    path_groups: Dict[str, Iterable[Path]],
) -> Dict[str, object]:
    """Detect path overlap across named splits."""
    sets = {k: {Path(p).resolve() for p in v} for k, v in path_groups.items()}
    overlaps = {}
    keys = list(sets)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            inter = sets[a] & sets[b]
            if inter:
                overlaps[f"{a}∩{b}"] = sorted(str(p) for p in list(inter)[:10])
    return {
        "counts": {k: len(v) for k, v in sets.items()},
        "overlaps": overlaps,
        "leakage": bool(overlaps),
    }


def class_balance_msg(counts: Dict[str, int], *, severe_ratio: float = 5.0) -> str:
    if not counts:
        return "No classes."
    vals = [c for c in counts.values() if c > 0]
    if len(vals) < 2:
        return f"Only one non-empty class: {counts}"
    imb = max(vals) / max(min(vals), 1)
    if imb > severe_ratio:
        return (
            f"SEVERE imbalance ratio={imb:.2f} (max/min). counts={counts}. "
            "Use class weights, oversampling, or focal loss — do not train with plain CE alone."
        )
    return f"Relatively balanced ratio={imb:.2f}. counts={counts}"
