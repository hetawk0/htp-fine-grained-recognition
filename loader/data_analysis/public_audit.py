"""Public sketch-domain audits: QuickDraw, SketchyScene-7k, TU-Berlin."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from loader.data_analysis.common import (
    iter_images,
    md5_file,
    sample_image_stats,
    split_leakage,
    summarize_stats,
)


@dataclass
class PublicAuditResult:
    name: str
    root: str
    available: bool = False
    summary: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    must_fix: List[str] = field(default_factory=list)
    nice_to_have: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def audit_quickdraw(
    root: Path,
    categories: Sequence[str] = ("house", "tree", "face"),
    *,
    sample_rows: int = 2048,
) -> PublicAuditResult:
    root = root.resolve()
    res = PublicAuditResult(name="quickdraw", root=str(root))
    if not root.is_dir():
        res.issues.append("QuickDraw root missing")
        res.must_fix.append("Download house/tree/face .npy under datasets/raw/quickdraw/numpy_bitmap")
        return res

    res.available = True
    per: Dict[str, Any] = {}
    for cat in categories:
        path = root / f"{cat}.npy"
        info: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if not path.exists():
            res.must_fix.append(f"Missing QuickDraw {cat}.npy")
            per[cat] = info
            continue
        try:
            arr = np.load(path, mmap_mode="r")
            info["shape"] = list(arr.shape)
            info["dtype"] = str(arr.dtype)
            n = int(arr.shape[0])
            info["n"] = n
            # sample value stats
            idx = np.linspace(0, n - 1, num=min(sample_rows, n), dtype=int)
            sample = np.asarray(arr[idx], dtype=np.float32)
            info["min"] = float(sample.min())
            info["max"] = float(sample.max())
            info["mean"] = float(sample.mean())
            if arr.ndim != 2 or arr.shape[1] != 784:
                res.must_fix.append(
                    f"QuickDraw {cat}: unexpected shape {arr.shape}, expected [N,784]"
                )
            if sample.max() > 1.5:
                info["value_range_note"] = "[0,255]-like bitmap; invert when matching HTP white-bg"
            per[cat] = info
        except (ValueError, OSError, EOFError) as exc:
            res.must_fix.append(f"QuickDraw {cat}.npy unreadable: {exc}")
            per[cat] = info

    res.summary = {
        "categories": per,
        "role": "Sketch-domain pretrain / bootstrap ONLY — no clinical Nor/De labels",
        "person_category": "Official person.npy does not exist; use face or TU-Berlin person sitting/walking",
    }
    if "binary" not in res.summary:
        res.nice_to_have.append(
            "QuickDrawHTPDataset binary=True invents random depression labels — never use for clinical metrics."
        )
    return res


def audit_sketchyscene(
    root: Path,
    *,
    hash_sample_per_split: int = 200,
) -> PublicAuditResult:
    root = root.resolve()
    res = PublicAuditResult(name="sketchyscene", root=str(root))
    if not root.is_dir():
        res.issues.append("SketchyScene root missing")
        res.nice_to_have.append("Optional: extract SketchyScene-7k for composition priors")
        return res

    res.available = True
    splits: Dict[str, List[Path]] = {}
    for split in ("train", "val", "test"):
        d = root / split / "DRAWING_GT"
        splits[split] = iter_images(d) if d.is_dir() else []

    counts = {k: len(v) for k, v in splits.items()}
    leak = split_leakage(splits)

    # Basename collisions are expected (each split renumbers L0_sample*.png).
    # Only flag leakage when identical content hashes appear in multiple splits.
    basenames = {k: {p.name for p in v} for k, v in splits.items()}
    name_overlaps = {}
    keys = list(basenames)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            inter = basenames[a] & basenames[b]
            if inter:
                name_overlaps[f"{a}∩{b}"] = {
                    "n": len(inter),
                    "examples": sorted(inter)[:10],
                    "note": "Same basenames across splits are normal for SketchyScene; check content hashes.",
                }

    hash_to_splits: Dict[str, set] = defaultdict(set)
    for split, paths in splits.items():
        step = max(1, len(paths) // max(hash_sample_per_split, 1))
        for p in paths[::step][:hash_sample_per_split]:
            try:
                hash_to_splits[md5_file(p)].add(split)
            except OSError:
                continue
    hash_cross = {h: sorted(s) for h, s in hash_to_splits.items() if len(s) > 1}

    # Spot-check a few shared basenames: content should differ
    shared_name_same_content = 0
    shared_checked = 0
    shared = basenames.get("train", set()) & basenames.get("test", set())
    for name in sorted(shared)[:20]:
        pt, pq = root / "train" / "DRAWING_GT" / name, root / "test" / "DRAWING_GT" / name
        if pt.exists() and pq.exists():
            shared_checked += 1
            try:
                if md5_file(pt) == md5_file(pq):
                    shared_name_same_content += 1
            except OSError:
                pass

    # MAT sidecar presence
    mat_ok = {}
    for split in ("train", "val", "test"):
        drawing = root / split / "DRAWING_GT"
        class_gt = root / split / "CLASS_GT"
        inst = root / split / "INSTANCE_GT"
        mat_ok[split] = {
            "DRAWING_GT_png": counts.get(split, 0),
            "CLASS_GT_exists": class_gt.is_dir(),
            "INSTANCE_GT_exists": inst.is_dir(),
            "CLASS_GT_mat": sum(1 for _ in class_gt.glob("*.mat")) if class_gt.is_dir() else 0,
        }

    stats = sample_image_stats(
        [p for ps in splits.values() for p in ps[:40]],
        max_n=60,
        prefer=[],
    )

    res.summary = {
        "counts": counts,
        "sum": sum(counts.values()),
        "path_leakage": leak,
        "basename_overlaps": name_overlaps,
        "content_hash_cross_split_sampled": {
            "n_cross": len(hash_cross),
            "examples": dict(list(hash_cross.items())[:5]),
        },
        "shared_basename_content_check": {
            "checked": shared_checked,
            "same_content": shared_name_same_content,
        },
        "sidecars": mat_ok,
        "image_stats": summarize_stats(stats),
        "role": "Scene composition / mask priors — NOT depression labels",
    }

    if leak.get("leakage") or hash_cross or shared_name_same_content:
        res.must_fix.append(
            "SketchyScene: real split leakage "
            f"(path_overlap={leak.get('leakage')}, content_hash_cross={len(hash_cross)}, "
            f"same_basename_same_md5={shared_name_same_content})"
        )
    elif name_overlaps:
        res.nice_to_have.append(
            "SketchyScene: basenames repeat across train/val/test (per-split indexing); "
            "content hashes differ — not leakage. Always key samples by split+path, not basename alone."
        )
    if sum(counts.values()) == 0:
        res.must_fix.append("SketchyScene: no DRAWING_GT PNGs found")
    expected = {"train": 5617, "val": 535, "test": 1113}
    for k, exp in expected.items():
        if counts.get(k, 0) and counts[k] != exp:
            res.nice_to_have.append(f"SketchyScene {k} count={counts[k]} (paper/readme often cites {exp})")
    return res


def audit_tu_berlin(
    svg_root: Path,
    wanted: Sequence[str] = (
        "house",
        "tree",
        "face",
        "person sitting",
        "person walking",
        "palm tree",
    ),
) -> PublicAuditResult:
    svg_root = svg_root.resolve()
    res = PublicAuditResult(name="tu_berlin", root=str(svg_root))
    if not svg_root.is_dir():
        res.issues.append("TU-Berlin svg root missing")
        res.nice_to_have.append("Optional: unzip TU-Berlin sketches_svg for object-level sketch aux")
        return res

    res.available = True
    counts = {}
    empty = []
    for cat in wanted:
        d = svg_root / cat
        n = sum(1 for p in d.glob("*.svg")) if d.is_dir() else 0
        counts[cat] = n
        if n == 0:
            empty.append(cat)

    # total cats on disk
    all_cats = [p.name for p in svg_root.iterdir() if p.is_dir()] if svg_root.is_dir() else []
    res.summary = {
        "htp_adjacent_counts": counts,
        "n_category_dirs": len(all_cats),
        "role": "Object-level sketch classification pretrain — rasterize SVG before CNN",
    }
    if empty:
        res.nice_to_have.append(f"TU-Berlin missing HTP-adjacent cats: {empty}")
    if len(all_cats) and len(all_cats) < 200:
        res.nice_to_have.append(
            f"TU-Berlin only {len(all_cats)} category dirs (full set ≈252) — check unzip"
        )
    return res


def audit_public_sketches(raw: Path) -> Dict[str, PublicAuditResult]:
    return {
        "quickdraw": audit_quickdraw(raw / "quickdraw" / "numpy_bitmap"),
        "sketchyscene": audit_sketchyscene(raw / "sketchyscene" / "SketchyScene-7k"),
        "tu_berlin": audit_tu_berlin(raw / "tu_berlin" / "svg"),
    }
