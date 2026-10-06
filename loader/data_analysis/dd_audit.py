"""DD (Depression Drawings) audit — filename labels, folder mistrust, splits."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Allow `python -m loader.data_analysis.check` from repo root
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loader.clinical import parse_dd_filename  # noqa: E402
from loader.data_analysis.common import (  # noqa: E402
    check_corrupted,
    check_exact_and_near_duplicates,
    class_balance_msg,
    iter_images,
    random_split_indices,
    sample_image_stats,
    split_leakage,
    stratified_split_indices,
    summarize_stats,
)


@dataclass
class DDRecord:
    path: Path
    label: int  # 0=Nor, 1=De
    severity: Optional[str]
    folder: str
    folder_implied_label: Optional[int]  # from folder name 0/1 if applicable


@dataclass
class DDAuditResult:
    root: str
    total_png: int = 0
    include_augmented: bool = False
    label_counts: Dict[str, int] = field(default_factory=dict)
    severity_counts: Dict[str, int] = field(default_factory=dict)
    by_folder: Dict[str, Dict[str, int]] = field(default_factory=dict)
    folder_filename_conflicts: List[str] = field(default_factory=list)
    conflict_count: int = 0
    unparsed: List[str] = field(default_factory=list)
    balance_msg: str = ""
    imbalance_ratio_nor_over_de: float = 0.0
    class_weight_suggestion: Dict[str, float] = field(default_factory=dict)
    corrupted: List[str] = field(default_factory=list)
    duplicates: Dict[str, Any] = field(default_factory=dict)
    image_stats: Dict[str, Any] = field(default_factory=dict)
    current_loader_split: Dict[str, Any] = field(default_factory=dict)
    stratified_split: Dict[str, Any] = field(default_factory=dict)
    augmented_leakage_risk: Dict[str, Any] = field(default_factory=dict)
    label_rules: Dict[str, str] = field(default_factory=dict)
    must_fix: List[str] = field(default_factory=list)
    nice_to_have: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def _folder_implied_label(folder: str) -> Optional[int]:
    if folder == "0":
        return 0
    if folder in {"1", "1_augmented"}:
        return 1
    return None


def collect_dd_records(
    root: Path,
    *,
    include_augmented: bool = False,
) -> Tuple[List[DDRecord], List[Path]]:
    skip = set() if include_augmented else {"1_augmented"}
    paths = iter_images(root, skip_dirs=skip)
    records: List[DDRecord] = []
    unparsed_paths: List[Path] = []
    for p in paths:
        sample = parse_dd_filename(p)
        folder = p.parent.name
        if sample is None:
            unparsed_paths.append(p)
            continue
        records.append(
            DDRecord(
                path=p,
                label=int(sample.label),
                severity=sample.meta.get("severity"),
                folder=folder,
                folder_implied_label=_folder_implied_label(folder),
            )
        )
    return records, unparsed_paths


def _split_class_counts(
    records: Sequence[DDRecord],
    indices: Sequence[int],
) -> Dict[str, int]:
    c = Counter(records[i].label for i in indices)
    return {"Nor": c.get(0, 0), "De": c.get(1, 0), "total": len(indices)}


def audit_dd(
    root: Path,
    *,
    include_augmented: bool = False,
    seed: int = 42,
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
    stats_n: int = 100,
    check_dups: bool = True,
    check_corrupt: bool = True,
) -> DDAuditResult:
    root = root.resolve()
    result = DDAuditResult(
        root=str(root),
        include_augmented=include_augmented,
        label_rules={
            "authoritative": "Filename tags: `-Nor-` → 0, `-De-` → 1; "
            "hospital/depression in stem → 1",
            "unreliable": "Folder ids `0/` / `1/` — `0/` mixes Nor+De; never use ImageFolder",
            "severity": "`-TH-{L,M,H}` suffix stored as severity; Nor is mostly L, De is M/H",
            "augmented": "`1_augmented/` = transforms of single hospital De; "
            "default loaders exclude (include_augmented=false)",
        },
    )
    if not root.is_dir():
        result.notes.append(f"DD root missing: {root}")
        result.must_fix.append(
            "Obtain / extract under datasets/raw/htp_depression_dd (delete archive after extract)"
        )
        return result

    records, unparsed = collect_dd_records(root, include_augmented=include_augmented)
    all_png = iter_images(root, skip_dirs=set() if include_augmented else {"1_augmented"})
    result.total_png = len(all_png)
    result.unparsed = [str(p.relative_to(root)) for p in unparsed]

    label_ctr = Counter({0: 0, 1: 0})
    sev_ctr: Counter = Counter()
    by_folder: Dict[str, Counter] = defaultdict(Counter)
    conflicts: List[str] = []

    for r in records:
        key = "Nor" if r.label == 0 else "De"
        label_ctr[r.label] += 1
        by_folder[r.folder][key] += 1
        if r.severity:
            sev_ctr[r.severity] += 1
        if r.folder_implied_label is not None and r.folder_implied_label != r.label:
            conflicts.append(
                f"{r.path.relative_to(root)}: folder implies {r.folder_implied_label}, "
                f"filename label={r.label} ({key})"
            )

    result.label_counts = {"Nor": label_ctr[0], "De": label_ctr[1], "unknown": len(unparsed)}
    result.severity_counts = dict(sev_ctr)
    result.by_folder = {k: dict(v) for k, v in sorted(by_folder.items())}
    result.folder_filename_conflicts = conflicts[:50]
    result.conflict_count = len(conflicts)
    n_nor, n_de = label_ctr[0], label_ctr[1]
    result.imbalance_ratio_nor_over_de = (n_nor / n_de) if n_de else float("inf")
    result.balance_msg = class_balance_msg({"Nor": n_nor, "De": n_de})
    # inverse-frequency weights normalized to mean 1
    total = max(n_nor + n_de, 1)
    if n_nor and n_de:
        w0 = total / (2.0 * n_nor)
        w1 = total / (2.0 * n_de)
        result.class_weight_suggestion = {"Nor_0": round(w0, 4), "De_1": round(w1, 4)}

    # Stats — prefer all De samples
    de_paths = [r.path for r in records if r.label == 1]
    stats = sample_image_stats(
        [r.path for r in records],
        max_n=stats_n,
        prefer=de_paths,
        seed=seed,
    )
    result.image_stats = summarize_stats(stats)

    if check_corrupt:
        result.corrupted = check_corrupted([r.path for r in records])

    if check_dups:
        dup = check_exact_and_near_duplicates([r.path for r in records], near=True)
        result.duplicates = {
            "exact_sets": dup.exact_sets,
            "exact_extra_files": dup.exact_extra_files,
            "near_sets": dup.near_sets,
            "examples": dup.examples,
        }

    # Current loader split (global shuffle) vs stratified
    labels = [r.label for r in records]
    cur = random_split_indices(len(records), ratios=ratios, seed=seed)
    strat = stratified_split_indices(labels, ratios=ratios, seed=seed)

    def pack(split_idx: Dict[str, List[int]]) -> Dict[str, Any]:
        paths = {
            name: [records[i].path for i in idxs] for name, idxs in split_idx.items()
        }
        leak = split_leakage(paths)
        return {
            "per_split": {k: _split_class_counts(records, v) for k, v in split_idx.items()},
            "leakage": leak,
            "empty_minority_splits": [
                name
                for name, idxs in split_idx.items()
                if _split_class_counts(records, idxs).get("De", 0) == 0
            ],
        }

    result.current_loader_split = pack(cur)
    result.stratified_split = pack(strat)

    # Augmented leakage: if include_augmented, stem family across splits
    if include_augmented:
        family_to_splits: Dict[str, set] = defaultdict(set)
        for name, idxs in cur.items():
            for i in idxs:
                r = records[i]
                # strip common aug suffixes
                stem = r.path.stem
                base = stem.split("_rotate")[0].split("_flip")[0].split("_brightness")[0].split(
                    "_contrast"
                )[0].split("_original")[0]
                family_to_splits[base].add(name)
        cross = {k: sorted(v) for k, v in family_to_splits.items() if len(v) > 1}
        result.augmented_leakage_risk = {
            "families_spanning_splits": len(cross),
            "examples": dict(list(cross.items())[:5]),
            "warning": (
                "Augmented variants of the same hospital image appear in multiple splits"
                if cross
                else "No cross-split augmented families under current seed"
            ),
        }

    # Prioritized work items
    if result.conflict_count:
        result.must_fix.append(
            f"DD: {result.conflict_count} folder↔filename label conflicts "
            f"(all De-in-folder-0). Loaders MUST parse Nor/De from filenames; "
            "never trust ImageFolder on 0/1."
        )
    if n_de and result.imbalance_ratio_nor_over_de >= 5:
        result.must_fix.append(
            f"DD: severe imbalance Nor={n_nor} De={n_de} "
            f"(ratio≈{result.imbalance_ratio_nor_over_de:.1f}:1). "
            f"Suggested class weights {result.class_weight_suggestion} "
            "or WeightedRandomSampler / focal loss before any real train."
        )
    if result.current_loader_split.get("empty_minority_splits"):
        result.must_fix.append(
            "DD: current HTPFolderDataset._split is NOT stratified — "
            f"empty De in splits {result.current_loader_split['empty_minority_splits']} "
            f"at seed={seed}. Switch to stratified split before training."
        )
    else:
        val_de = result.current_loader_split["per_split"].get("val", {}).get("De", 0)
        test_de = result.current_loader_split["per_split"].get("test", {}).get("De", 0)
        if val_de < 5 or test_de < 5:
            result.must_fix.append(
                "DD: HTPFolderDataset._split is global-shuffle (not stratified); "
                f"seed={seed} yields val De={val_de}, test De={test_de} "
                f"({result.current_loader_split['per_split']}). "
                "Implement stratified split before trusting val/test metrics."
            )

    if result.corrupted:
        result.must_fix.append(f"DD: {len(result.corrupted)} corrupted images — remove/fix.")

    if result.unparsed:
        result.must_fix.append(
            f"DD: {len(result.unparsed)} PNGs with unparsed labels: {result.unparsed[:5]}"
        )

    if include_augmented and result.augmented_leakage_risk.get("families_spanning_splits"):
        result.must_fix.append(
            "DD: with include_augmented=True, augmented copies leak across splits — "
            "keep augs train-only or disable."
        )

    if result.duplicates.get("exact_sets"):
        result.nice_to_have.append(
            f"DD: {result.duplicates['exact_sets']} exact duplicate sets — dedupe before train."
        )
    if result.duplicates.get("near_sets"):
        result.nice_to_have.append(
            f"DD: {result.duplicates['near_sets']} near-duplicate aHash sets — review manually."
        )

    stats = result.image_stats
    if stats.get("brightness_warn") == "high/white-bg":
        result.nice_to_have.append(
            "DD: very bright canvases (mean intensity high) — faint strokes; "
            "consider contrast stretch / invert-aware norms."
        )
    if stats.get("unique_sizes", 0) > 1:
        result.nice_to_have.append("DD: mixed resolutions — resize/pad in transforms (already expected).")

    result.nice_to_have.append(
        "DD: no small-object boxes yet — Label Studio / HTPSeg before SOE-FBANet SO claims."
    )
    result.notes.append(
        "Severity L on Nor (n≈472) vs M/H on De — treat L/M/H as filename metadata, "
        "not a three-way classification target unless clinically validated."
    )
    return result
