"""Train CatBoost regressors strictly for adopted animals."""

import argparse
from pathlib import Path
import sys

from aac_adoption.models.train_adopted_regression import train_all_adopted


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train adopted-only regression models.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/modeling_dataset.csv"),
        help="Path to engineered modeling features",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("reports/metrics"),
        help="Directory to save metric outputs",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models/advanced"),
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit rows for quick testing",
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)

    args = parser.parse_args()

    if not args.data_path.exists():
        print(f"Error: {args.data_path} not found.")
        sys.exit(1)

    print(f"Training adopted-only regression models from {args.data_path}...")
    outputs = train_all_adopted(
        data_path=args.data_path,
        metrics_dir=args.metrics_dir,
        models_dir=args.models_dir,
        max_rows=args.max_rows,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
    )
    
    print("\nAdopted-only Regression Metrics:")
    print(outputs.regression_metrics[["animal_subset", "mae", "rmse", "r2", "train_period", "test_period"]].to_string(index=False))

    print(f"\nSaved metrics to {args.metrics_dir}")
    print(f"Saved artifacts to {args.models_dir}")


if __name__ == "__main__":
    main()
