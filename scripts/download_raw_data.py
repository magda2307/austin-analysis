"""CLI for downloading AAC raw CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aac_adoption.data.download_data import DATASET_SOURCES, download_aac_raw_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Austin Animal Center raw CSV files")
    parser.add_argument(
        "--source",
        choices=sorted(DATASET_SOURCES),
        default="historical",
        help="AAC Socrata source to download",
    )
    parser.add_argument("--output-dir", default="data/raw", help="Directory for raw CSV files")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing CSV files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = DATASET_SOURCES[args.source]
    intakes_path, outcomes_path = download_aac_raw_data(
        output_dir=args.output_dir,
        source_name=args.source,
        overwrite=args.overwrite,
    )
    print(f"Source: {source.description}")
    print(f"Intakes: {intakes_path}")
    print(f"Outcomes: {outcomes_path}")


if __name__ == "__main__":
    main()

