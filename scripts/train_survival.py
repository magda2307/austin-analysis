"""CLI for training survival models for LOS analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aac_adoption.models.train_survival import train_all_survival


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train survival models for AAC LOS analysis")
    parser.add_argument("--data", dest="data_path", type=Path, default="data/processed/modeling_dataset.csv")
    parser.add_argument("--metrics-dir", default="reports/metrics")
    parser.add_argument("--models-dir", default="models/survival")
    parser.add_argument("--tables-dir", default="reports/tables")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--smoothing", type=float, default=10.0, help="Smoothing parameter for target encoder")
    parser.add_argument(
        "--tuned-params-path",
        type=Path,
        default=None,
        help="Path to JSON file with tuned hyperparameters",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_path.exists():
        print(f"Error: {args.data_path} not found.")
        sys.exit(1)

    print(f"Training survival models using {args.data_path}...")
    outputs = train_all_survival(
        data_path=args.data_path,
        metrics_dir=args.metrics_dir,
        models_dir=args.models_dir,
        tables_dir=args.tables_dir,
        max_rows=args.max_rows if args.max_rows > 0 else None,
        smoothing=args.smoothing,
        tuned_params_path=args.tuned_params_path,
    )
    print(f"Wrote survival metrics to {args.metrics_dir}")
    print(f"Wrote survival model artifacts to {args.models_dir}")
    print(f"Wrote survival tables to {args.tables_dir}")
    print(outputs.concordance_metrics.to_string(index=False))
    print(outputs.brier_metrics.to_string(index=False))
    print(outputs.calibration_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
