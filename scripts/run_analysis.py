"""Run thesis analysis table generation from existing outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aac_adoption.analysis.hypothesis_tables import create_hypothesis_support_tables
from aac_adoption.analysis.model_comparison import create_model_comparison_tables
from aac_adoption.visualization.plots import create_eda_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AAC thesis analysis outputs")
    parser.add_argument("--data", default="data/processed/modeling_dataset.csv")
    parser.add_argument("--metrics-dir", default="reports/metrics")
    parser.add_argument("--tables-dir", default="reports/tables")
    parser.add_argument("--figures-dir", default="reports/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_eda_outputs(args.data, args.tables_dir, args.figures_dir)
    create_model_comparison_tables(args.metrics_dir, args.tables_dir)
    create_hypothesis_support_tables(args.data, args.tables_dir)
    print(f"Wrote analysis tables to {args.tables_dir}")
    print(f"Wrote figures to {args.figures_dir}")


if __name__ == "__main__":
    main()

