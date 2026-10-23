#!/usr/bin/env python3
"""Download a versioned Roboflow HTP export into the ignored datasets folder."""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORMAT = "coco"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        type=int,
        required=True,
        help="Roboflow Universe dataset version number",
    )
    parser.add_argument(
        "--format",
        choices=("coco", "yolov8"),
        default=DEFAULT_FORMAT,
        help="Export format (default: coco, suitable for multiple detector frameworks)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/roboflow_htp"),
        help="Output directory, relative to the repository root unless absolute",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()
    datasets_dir = (ROOT / "datasets").resolve()
    if datasets_dir not in output_dir.parents:
        parser.error(f"output directory must be inside {datasets_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(
            f"output directory is not empty: {output_dir}; choose a new directory "
            "to avoid overwriting or mixing dataset versions"
        )

    try:
        from roboflow import Roboflow
    except ImportError:
        parser.error(
            "Roboflow SDK is not installed; install it with "
            "'python -m pip install roboflow'"
        )

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Roboflow API key (input hidden): ").strip()
    if not api_key:
        parser.error("a Roboflow API key is required")

    output_dir.mkdir(parents=True, exist_ok=True)
    roboflow = Roboflow(api_key=api_key)
    project = roboflow.workspace("htp").project("htp-lewxu")
    version = project.version(args.version)
    dataset = version.download(args.format, location=str(output_dir))

    print(
        f"Downloaded Roboflow htp/htp-lewxu version {args.version} "
        f"({args.format}) to {getattr(dataset, 'location', output_dir)}"
    )
    print("The downloaded dataset is local-only and excluded from Git.")


if __name__ == "__main__":
    main()
