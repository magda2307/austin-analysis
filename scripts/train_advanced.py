"""CLI for training advanced AAC CatBoost models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aac_adoption.models.train_advanced import train_all_advanced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train advanced AAC CatBoost models")
    parser.add_argument("--data", dest="data_path", type=Path, default="data/processed/modeling_dataset.csv")
    parser.add_argument("--metrics-dir", default="reports/metrics")
    parser.add_argument("--models-dir", default="models/advanced")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
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

    print(f"Training advanced models using {args.data_path}...")
    outputs = train_all_advanced(
        data_path=args.data_path,
        metrics_dir=args.metrics_dir,
        models_dir=args.models_dir,
        max_rows=args.max_rows if args.max_rows > 0 else None,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        early_stopping_rounds=args.early_stopping_rounds,
        tuned_params_path=args.tuned_params_path,
    )
    print(f"Wrote advanced metrics to {args.metrics_dir}")
    print(f"Wrote advanced model artifacts to {args.models_dir}")
    print(outputs.classification_metrics.to_string(index=False))
    print(outputs.regression_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
