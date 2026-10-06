#!/usr/bin/env python3
"""Viola dataset audit (meddef check.py rigor, adapted for HTP / DD / sketches).

Meddef ``loader/data_analysis/check.py`` map (what we mirror):
  - class distribution + imbalance warning
  - image dimensions / channels / color space
  - corrupted-file scan
  - pixel range / brightness / contrast / sharpness proxy
  - exact (+ near) duplicate detection
  - label sanity
  - text + CSV result dumps under analysis/

Viola-specific additions (not in meddef medical ImageFolder flow):
  - DD filename Nor/De parsing vs folder 0/1 mistrust
  - stratified vs current random split integrity / minority emptiness
  - augmented-family leakage if include_augmented
  - QuickDraw .npy shape/range; SketchyScene train/val/test leakage;
    TU-Berlin HTP-adjacent SVG counts

Examples:
  python -m loader.data_analysis.check
  python -m loader.data_analysis.check --raw datasets/raw --include-augmented
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loader.data_analysis.dd_audit import audit_dd  # noqa: E402
from loader.data_analysis.public_audit import audit_public_sketches  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "analysis"
TXT_RESULTS = OUT_DIR / "dataset_analysis_results.txt"
CSV_RESULTS = OUT_DIR / "dataset_analysis_results.csv"
JSON_RESULTS = OUT_DIR / "dataset_analysis_results.json"


def run_full_audit(
    raw: Path,
    *,
    include_augmented: bool = False,
    seed: int = 42,
    skip_dups: bool = False,
    skip_corrupt: bool = False,
) -> Dict[str, Any]:
    raw = raw.resolve()
    dd = audit_dd(
        raw / "htp_depression_dd",
        include_augmented=include_augmented,
        seed=seed,
        check_dups=not skip_dups,
        check_corrupt=not skip_corrupt,
    )
    public = audit_public_sketches(raw)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "raw": str(raw),
        "dd": dd.to_dict(),
        "public": {k: v.to_dict() for k, v in public.items()},
        "prioritized_work": _prioritized(dd, public),
    }
    return payload


def _prioritized(dd, public) -> Dict[str, List[str]]:
    must: List[str] = list(dd.must_fix)
    nice: List[str] = list(dd.nice_to_have)
    for name, res in public.items():
        must.extend(res.must_fix)
        nice.extend(res.nice_to_have)
    # de-dupe preserve order
    def uniq(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return {"must_fix_before_train": uniq(must), "nice_to_have": uniq(nice)}


def save_outputs(payload: Dict[str, Any], out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_analysis_results.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )

    dd = payload["dd"]
    lines = [
        f"Timestamp: {payload['timestamp']}",
        f"Raw: {payload['raw']}",
        "",
        "=== DD ===",
        f"total_png={dd.get('total_png')} labels={dd.get('label_counts')}",
        f"severity={dd.get('severity_counts')}",
        f"by_folder={dd.get('by_folder')}",
        f"conflict_count={dd.get('conflict_count')}",
        f"balance: {dd.get('balance_msg')}",
        f"class_weights: {dd.get('class_weight_suggestion')}",
        f"image_stats: {dd.get('image_stats')}",
        f"duplicates: {dd.get('duplicates')}",
        f"corrupted_n={len(dd.get('corrupted') or [])}",
        f"current_loader_split: {dd.get('current_loader_split')}",
        f"stratified_split: {dd.get('stratified_split')}",
        "",
        "=== Public ===",
    ]
    for name, block in payload["public"].items():
        lines.append(f"-- {name} available={block.get('available')}")
        lines.append(f"   summary_keys={list((block.get('summary') or {}).keys())}")
        if block.get("must_fix"):
            lines.append(f"   must_fix={block['must_fix']}")
    lines += ["", "=== Prioritized work ==="]
    for k, items in payload["prioritized_work"].items():
        lines.append(f"{k}:")
        for it in items:
            lines.append(f"  - {it}")
    lines.append("-" * 50)
    txt_path = out_dir / "dataset_analysis_results.txt"
    with txt_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Flat CSV row (meddef-style append)
    header = [
        "Timestamp",
        "DD_Nor",
        "DD_De",
        "DD_Unknown",
        "DD_Imbalance_Nor_over_De",
        "DD_Conflicts",
        "DD_ExactDupSets",
        "DD_Corrupted",
        "DD_EmptyMinoritySplits",
        "QD_house",
        "QD_tree",
        "QD_face",
        "SS_train",
        "SS_val",
        "SS_test",
        "MustFixCount",
    ]
    qd = (payload["public"].get("quickdraw") or {}).get("summary", {}).get("categories", {})
    ss = (payload["public"].get("sketchyscene") or {}).get("summary", {}).get("counts", {})
    row = {
        "Timestamp": payload["timestamp"],
        "DD_Nor": dd.get("label_counts", {}).get("Nor"),
        "DD_De": dd.get("label_counts", {}).get("De"),
        "DD_Unknown": dd.get("label_counts", {}).get("unknown"),
        "DD_Imbalance_Nor_over_De": dd.get("imbalance_ratio_nor_over_de"),
        "DD_Conflicts": dd.get("conflict_count"),
        "DD_ExactDupSets": (dd.get("duplicates") or {}).get("exact_sets"),
        "DD_Corrupted": len(dd.get("corrupted") or []),
        "DD_EmptyMinoritySplits": "|".join(
            (dd.get("current_loader_split") or {}).get("empty_minority_splits") or []
        ),
        "QD_house": (qd.get("house") or {}).get("n"),
        "QD_tree": (qd.get("tree") or {}).get("n"),
        "QD_face": (qd.get("face") or {}).get("n"),
        "SS_train": ss.get("train"),
        "SS_val": ss.get("val"),
        "SS_test": ss.get("test"),
        "MustFixCount": len(payload["prioritized_work"]["must_fix_before_train"]),
    }
    csv_path = out_dir / "dataset_analysis_results.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Viola meddef-style dataset audit")
    parser.add_argument("--raw", type=Path, default=ROOT / "datasets" / "raw")
    parser.add_argument("--include-augmented", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-dups", action="store_true", help="Skip MD5/aHash (faster)")
    parser.add_argument("--skip-corrupt", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Where to write txt/csv/json (default: loader/data_analysis/analysis/)",
    )
    args = parser.parse_args()

    print(f"Auditing raw={args.raw.resolve()}")
    payload = run_full_audit(
        args.raw,
        include_augmented=args.include_augmented,
        seed=args.seed,
        skip_dups=args.skip_dups,
        skip_corrupt=args.skip_corrupt,
    )
    save_outputs(payload, out_dir=args.out_dir.resolve())

    dd = payload["dd"]
    print(
        f"DD: Nor={dd['label_counts'].get('Nor')} De={dd['label_counts'].get('De')} "
        f"conflicts={dd['conflict_count']} imbalance={dd['imbalance_ratio_nor_over_de']:.2f}"
    )
    print("Current loader split:", dd.get("current_loader_split", {}).get("per_split"))
    print("Stratified split:", dd.get("stratified_split", {}).get("per_split"))
    print("Must-fix:")
    for item in payload["prioritized_work"]["must_fix_before_train"]:
        print(f"  ! {item}")
    print(f"Wrote {args.out_dir / 'dataset_analysis_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
