"""CLI for training AAC baseline models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aac_adoption.models.train_baseline import train_all_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AAC adoption baseline models")
    parser.add_argument(
        "--data",
        default="data/processed/modeling_dataset.csv",
        help="Path to processed modeling dataset",
    )
    parser.add_argument(
        "--output",
        default="reports/metrics/baseline_metrics.csv",
        help="Optional combined baseline metrics CSV path",
    )
    parser.add_argument(
        "--metrics-dir",
        default="reports/metrics",
        help="Directory for classification/regression metrics CSVs",
    )
    parser.add_argument(
        "--models-dir",
        default="models/baseline",
        help="Directory for fitted baseline model artifacts",
    )
    parser.add_argument(
        "--tables-dir",
        default="reports/tables",
        help="Directory for interpretability tables",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Reproducible row sample for fast first runs; use 0 for full dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_rows = args.max_rows if args.max_rows > 0 else None
    outputs = train_all_baselines(
        data_path=args.data,
        metrics_dir=args.metrics_dir,
        models_dir=args.models_dir,
        tables_dir=args.tables_dir,
        max_rows=max_rows,
        output_path=args.output,
    )
    print(f"Wrote combined metrics to {args.output}")
    print(f"Wrote split metrics to {args.metrics_dir}")
    print(f"Wrote model artifacts to {args.models_dir}")
    print(outputs.classification_metrics.to_string(index=False))
    print(outputs.regression_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
