"""Tune hyperparameters for core models."""

import argparse
from pathlib import Path
import sys
import pandas as pd

from aac_adoption.models.tune import tune_models, save_tuned_params
from aac_adoption.models.train_baseline import limit_rows

def main():
    parser = argparse.ArgumentParser(description="Tune hyperparameters for CatBoost and HistGradientBoosting.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/modeling_dataset.csv"),
        help="Path to engineered modeling features",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("models/tuning/best_params.json"),
        help="Path to save best hyperparameters json",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit rows for quick testing",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of optuna trials per model",
    )

    args = parser.parse_args()

    if not args.data_path.exists():
        print(f"Error: {args.data_path} not found.")
        sys.exit(1)

    print(f"Tuning hyperparameters using {args.data_path}...")
    
    header = pd.read_csv(args.data_path, nrows=0)
    parse_dates = [col for col in ["intake_datetime", "outcome_datetime"] if col in header.columns]
    df = pd.read_csv(args.data_path, parse_dates=parse_dates)
    df = limit_rows(df, args.max_rows)
    
    best_params = tune_models(df, n_trials=args.n_trials)
    
    save_tuned_params(best_params, args.output_path)
    print(f"Saved tuned hyperparameters to {args.output_path}")

if __name__ == "__main__":
    main()
