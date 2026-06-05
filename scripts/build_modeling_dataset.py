"""CLI for building the AAC modeling dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aac_adoption.data.build_dataset import build_modeling_dataset_from_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AAC adoption modeling dataset")
    parser.add_argument("--intakes", default="data/raw/intakes.csv", help="Path to raw intakes CSV")
    parser.add_argument("--outcomes", default="data/raw/outcomes.csv", help="Path to raw outcomes CSV")
    parser.add_argument(
        "--output",
        default="data/processed/modeling_dataset.csv",
        help="Path for processed modeling dataset CSV",
    )
    parser.add_argument(
        "--context-data-dir",
        default="",
        help="Optional directory with cached context CSVs for enriched features",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_modeling_dataset_from_files(
        args.intakes,
        args.outcomes,
        args.output,
        context_data_dir=args.context_data_dir or None,
    )
    print(f"Wrote {args.output}")
    print(f"Matched rows: {result.matched_rows}")
    print(f"Unmatched intakes skipped: {result.unmatched_intakes}")


if __name__ == "__main__":
    main()
