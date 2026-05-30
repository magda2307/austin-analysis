"""CLI for training AAC gradient boosting models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aac_adoption.models.train_boosting import train_all_boosting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AAC adoption gradient boosting models")
    parser.add_argument("--data", default="data/processed/modeling_dataset.csv")
    parser.add_argument("--metrics-dir", default="reports/metrics")
    parser.add_argument("--models-dir", default="models/boosting")
    parser.add_argument("--tables-dir", default="reports/tables")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--permutation-repeats", type=int, default=3)
    parser.add_argument("--permutation-max-rows", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = train_all_boosting(
        data_path=args.data,
        metrics_dir=args.metrics_dir,
        models_dir=args.models_dir,
        tables_dir=args.tables_dir,
        max_rows=args.max_rows if args.max_rows > 0 else None,
        permutation_repeats=args.permutation_repeats,
        permutation_max_rows=args.permutation_max_rows,
    )
    print(f"Wrote boosting metrics to {args.metrics_dir}")
    print(f"Wrote boosting model artifacts to {args.models_dir}")
    print(outputs.classification_metrics.to_string(index=False))
    print(outputs.regression_metrics.to_string(index=False))


if __name__ == "__main__":
    main()

