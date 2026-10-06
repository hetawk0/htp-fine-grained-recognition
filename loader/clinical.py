"""Abstract HTP sample protocol and clinical folder conventions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torch.utils.data import Dataset


@dataclass
class HTPSample:
    """Unified sample record for Viola loaders."""

    path: Optional[Path] = None
    label: int = 0  # 0=normal / negative, 1=depressed / positive
    has_small_object: bool = False
    bbox: Optional[Tuple[float, float, float, float]] = None  # x,y,w,h normalized
    meta: Dict[str, Any] = field(default_factory=dict)


# Expected clinical layouts under datasets/raw/ (see docs/DATASETS.md)
CLINICAL_LAYOUTS: Dict[str, Dict[str, Any]] = {
    "zhang": {
        "root": "datasets/raw/zhang_htp_1615",
        "description": (
            "Zhang / FBANet HTP (~1615 sketches). Prefer class folders "
            "normal/ and depressed/ OR a labels.csv with columns "
            "filename,label[,has_small_object,x,y,w,h]."
        ),
        "class_dirs": {"normal": 0, "depressed": 1},
        "label_aliases": {
            "0": 0,
            "normal": 0,
            "neg": 0,
            "negative": 0,
            "1": 1,
            "depressed": 1,
            "depression": 1,
            "pos": 1,
            "positive": 1,
        },
    },
    "pan": {
        "root": "datasets/raw/pan_htp_college",
        "description": (
            "Pan et al. college HTP. Class folders positive/negative "
            "(or pos/neg) or labels.csv."
        ),
        "class_dirs": {"negative": 0, "positive": 1, "neg": 0, "pos": 1},
        "label_aliases": {
            "0": 0,
            "negative": 0,
            "neg": 0,
            "1": 1,
            "positive": 1,
            "pos": 1,
        },
    },
    "bilgi": {
        "root": "datasets/raw/biligi_htp",
        "description": (
            "Istanbul Bilgi child HTP. Often PDF/Excel from Hatipoglu pipeline; "
            "after conversion, use images/ + scores.csv with "
            "filename,depression_label (or continuous score + threshold)."
        ),
        "class_dirs": {"normal": 0, "depressed": 1},
        "label_aliases": {
            "0": 0,
            "normal": 0,
            "1": 1,
            "depressed": 1,
        },
    },
    "dd": {
        "root": "datasets/raw/htp_depression_dd",
        "display_name": "HTP-Depression-DD",
        "canonical_name": "htp_depression_dd",
        "description": (
            "Primary Topic A clinical HTP dump (paper name: HTP-Depression-DD / "
            "htp_depression_dd). Filesystem dir: datasets/raw/htp_depression_dd "
            "(archives deleted after extract). Full share ≈1591 unique PNG after "
            "dedup (1239 Nor / 352 De); folder layout 0/≈1296 + 1/≈295 matches "
            "Zhang-style counts but labels MUST come from filename Nor/De "
            "(0/ mixes both). Optional 1_augmented/ (exclude by default). "
            "Severity L/M/H after -TH- in meta['severity']."
        ),
        "class_dirs": {"0": 0, "1": 1},  # fallback only; loader prefers filename parse
        "label_aliases": {
            "0": 0,
            "nor": 0,
            "normal": 0,
            "1": 1,
            "de": 1,
            "depressed": 1,
            "depression": 1,
        },
        "parse_filename_labels": True,
        "include_augmented_default": False,
    },
}


def parse_dd_filename(path: Path) -> Optional[HTPSample]:
    """
    Parse DD naming: ``...-Nor-TH-L.png`` / ``...-De-TH-M.png`` / hospital names.

    Returns None if the path should be skipped (unknown pattern).
    """
    stem = path.stem
    severity: Optional[str] = None
    label: Optional[int] = None
    if "-Nor-" in stem:
        label = 0
        tail = stem.rsplit("-", 1)[-1]
        if len(tail) <= 2:
            severity = tail
    elif "-De-" in stem:
        label = 1
        tail = stem.rsplit("-", 1)[-1]
        if len(tail) <= 2:
            severity = tail
    elif "depression" in stem.lower() or "hospital" in stem.lower():
        label = 1
    else:
        return None
    return HTPSample(
        path=path,
        label=int(label),
        meta={"severity": severity, "source_folder": path.parent.name},
    )


class HTPFolderDataset(Dataset):
    """
    Clinical HTP folder reader.

    Accepts either:
      root/
        normal/*.png|jpg
        depressed/*.png|jpg
    or:
      root/images/* + root/labels.csv
    or (DD):
      root/{0,1,1_augmented}/*.png with Nor/De tags in filenames
    """

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform=None,
        class_dirs: Optional[Dict[str, int]] = None,
        split_ratios: Sequence[float] = (0.7, 0.15, 0.15),
        seed: int = 42,
        require_exists: bool = True,
        parse_filename_labels: bool = False,
        include_augmented: bool = False,
        stratified: bool = True,
        split_file: Optional[str | Path] = None,
        forbid_class_dirs: bool = False,
    ):
        self.root = Path(root)
        self.transform = transform
        self.split = split
        self.samples: List[HTPSample] = []

        if require_exists and not self.root.exists():
            raise FileNotFoundError(
                f"Clinical dataset not found at {self.root}. "
                f"See docs/DATASETS.md and datasets/README.md for request/layout notes."
            )

        class_dirs = class_dirs or {"normal": 0, "depressed": 1}
        labels_csv = self.root / "labels.csv"
        use_dd = parse_filename_labels or self._looks_like_dd()
        if labels_csv.exists():
            self.samples = self._from_csv(labels_csv)
        elif use_dd:
            self.samples = self._from_dd_filenames(include_augmented=include_augmented)
        elif forbid_class_dirs:
            raise RuntimeError(
                f"Refusing ImageFolder/class_dirs fallback under {self.root} "
                "(DD-style dumps must use filename Nor/De labels)."
            )
        else:
            self.samples = self._from_class_dirs(class_dirs)

        self.samples = self._apply_split(
            self.samples,
            split,
            split_ratios,
            seed,
            stratified=stratified,
            split_file=Path(split_file) if split_file else None,
        )

    def _looks_like_dd(self) -> bool:
        return (self.root / "0").is_dir() and any(self.root.rglob("*-Nor-*.png"))

    def _from_class_dirs(self, class_dirs: Dict[str, int]) -> List[HTPSample]:
        out: List[HTPSample] = []
        for name, label in class_dirs.items():
            d = self.root / name
            if not d.is_dir():
                continue
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in self.IMAGE_EXTS:
                    out.append(HTPSample(path=p, label=label))
        return out

    def _from_dd_filenames(self, include_augmented: bool = False) -> List[HTPSample]:
        """Scan DD tree; labels from Nor/De filename tags (not folder ids)."""
        out: List[HTPSample] = []
        skip_dirs = {"_duplicates_removed", ".git"}
        if not include_augmented:
            skip_dirs.add("1_augmented")
        for p in sorted(self.root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in self.IMAGE_EXTS:
                continue
            if any(part in skip_dirs for part in p.parts):
                continue
            sample = parse_dd_filename(p)
            if sample is not None:
                out.append(sample)
        return out

    def _from_csv(self, csv_path: Path) -> List[HTPSample]:
        import csv

        out: List[HTPSample] = []
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel = row.get("filename") or row.get("path") or row.get("file")
                if not rel:
                    continue
                path = self.root / rel
                if not path.exists():
                    path = self.root / "images" / Path(rel).name
                raw = (
                    row.get("label")
                    or row.get("depression_label")
                    or row.get("class")
                    or "0"
                )
                label = int(float(raw)) if str(raw).replace(".", "", 1).isdigit() else (
                    1 if str(raw).lower() in {"depressed", "depression", "positive", "pos", "1"} else 0
                )
                so = str(row.get("has_small_object", "0")).lower() in {"1", "true", "yes"}
                bbox = None
                if all(k in row and row[k] for k in ("x", "y", "w", "h")):
                    bbox = tuple(float(row[k]) for k in ("x", "y", "w", "h"))  # type: ignore[assignment]
                out.append(
                    HTPSample(path=path, label=label, has_small_object=so, bbox=bbox, meta=dict(row))
                )
        return out

    def _sample_key(self, s: HTPSample) -> str:
        """Root-relative path so frozen splits sync across machines."""
        assert s.path is not None
        try:
            return str(s.path.resolve().relative_to(self.root.resolve()))
        except ValueError:
            return s.path.name

    def _lookup_key(self, key: str, by_key: Dict[str, HTPSample], samples: List[HTPSample]) -> Optional[HTPSample]:
        if key in by_key:
            return by_key[key]
        # absolute path from another host → try relative / basename
        name = Path(key).name
        # try last two path parts e.g. 0/foo.png
        parts = Path(key).parts
        if len(parts) >= 2:
            rel = str(Path(*parts[-2:]))
            if rel in by_key:
                return by_key[rel]
        matches = [s for s in samples if s.path and s.path.name == name]
        if len(matches) == 1:
            return matches[0]
        return None

    def _apply_split(
        self,
        samples: List[HTPSample],
        split: str,
        ratios: Sequence[float],
        seed: int,
        *,
        stratified: bool,
        split_file: Optional[Path],
    ) -> List[HTPSample]:
        if not samples:
            return []

        split_name = "val" if split in {"val", "valid", "validation"} else split
        by_key = {self._sample_key(s): s for s in samples}

        if split_file is not None and split_file.exists():
            with split_file.open(encoding="utf-8") as f:
                frozen = json.load(f)
            keys = frozen.get(split_name) or frozen.get(split) or []
            out = []
            for k in keys:
                hit = self._lookup_key(k, by_key, samples)
                if hit is not None:
                    out.append(hit)
            return out

        if stratified:
            from loader.data_analysis.common import stratified_split_indices

            labels = [s.label for s in samples]
            idx_map = stratified_split_indices(labels, ratios=ratios, seed=seed)
            sel = idx_map.get(split_name, idx_map.get("test", []))
            chosen = [samples[i] for i in sel]
        else:
            chosen = self._split_random(samples, split_name, ratios, seed)

        if split_file is not None:
            self._write_frozen_split(samples, ratios, seed, stratified, split_file)

        return chosen

    def _write_frozen_split(
        self,
        samples: List[HTPSample],
        ratios: Sequence[float],
        seed: int,
        stratified: bool,
        split_file: Path,
    ) -> None:
        if split_file.exists():
            return
        if stratified:
            from loader.data_analysis.common import stratified_split_indices

            idx_map = stratified_split_indices(
                [s.label for s in samples], ratios=ratios, seed=seed
            )
        else:
            from loader.data_analysis.common import random_split_indices

            idx_map = random_split_indices(len(samples), ratios=ratios, seed=seed)

        payload = {
            "seed": seed,
            "ratios": list(ratios),
            "stratified": stratified,
            "counts": {},
            "train": [],
            "val": [],
            "test": [],
        }
        for name, idxs in idx_map.items():
            keys = [self._sample_key(samples[i]) for i in idxs]
            payload[name] = keys
            payload["counts"][name] = {
                "total": len(idxs),
                "Nor": sum(1 for i in idxs if samples[i].label == 0),
                "De": sum(1 for i in idxs if samples[i].label == 1),
            }
        split_file.parent.mkdir(parents=True, exist_ok=True)
        with split_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[HTPFolderDataset] wrote frozen split → {split_file}")

    @staticmethod
    def _split_random(
        samples: List[HTPSample],
        split: str,
        ratios: Sequence[float],
        seed: int,
    ) -> List[HTPSample]:
        import random

        idx = list(range(len(samples)))
        rng = random.Random(seed)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        if split == "train":
            sel = idx[:n_train]
        elif split == "val":
            sel = idx[n_train : n_train + n_val]
        else:
            sel = idx[n_train + n_val :]
        return [samples[i] for i in sel]

    def __len__(self) -> int:
        return len(self.samples)

    def labels_list(self) -> List[int]:
        return [s.label for s in self.samples]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        from PIL import Image

        s = self.samples[index]
        assert s.path is not None
        img = Image.open(s.path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,
            "label": s.label,
            "so_mask": int(s.has_small_object),
            "path": str(s.path),
        }
