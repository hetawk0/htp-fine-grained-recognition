#!/usr/bin/env python3
"""
Optional download of QuickDraw house/tree/face bitmaps (~330 MB total).

Usage:
  python scripts/download_quickdraw.py
  python scripts/download_quickdraw.py --categories house tree
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "datasets/raw/quickdraw/numpy_bitmap"
URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy"
CATEGORIES = ("house", "tree", "face")


def download(category: str, dest_dir: Path, force: bool = False) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{category}.npy"
    if out.exists() and not force:
        print(f"skip existing {out}")
        return out
    url = URL.format(category=category)
    print(f"downloading {url} -> {out}")
    urllib.request.urlretrieve(url, out)
    print(f"done {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=str(DEFAULT_DIR))
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    dest = Path(args.out)
    for cat in args.categories:
        try:
            download(cat, dest, force=args.force)
        except Exception as exc:
            print(f"FAILED {cat}: {exc}", file=sys.stderr)
            sys.exit(1)
    print("QuickDraw download complete.")


if __name__ == "__main__":
    main()
