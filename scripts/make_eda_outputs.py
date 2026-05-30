"""CLI for creating initial AAC EDA outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aac_adoption.visualization.plots import create_eda_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create initial AAC EDA tables and figures")
    parser.add_argument(
        "--data",
        default="data/processed/modeling_dataset.csv",
        help="Path to processed modeling dataset",
    )
    parser.add_argument("--tables", default="reports/tables", help="Output directory for CSV tables")
    parser.add_argument("--figures", default="reports/figures", help="Output directory for PNG figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_eda_outputs(args.data, args.tables, args.figures)
    print(f"Wrote EDA tables to {args.tables}")
    print(f"Wrote EDA figures to {args.figures}")


if __name__ == "__main__":
    main()

