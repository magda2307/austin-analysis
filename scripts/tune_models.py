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
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5000,
        help="Maximum boosting iterations per trial",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of chronological cross-validation folds",
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
    
    best_params, studies = tune_models(
        df,
        n_trials=args.n_trials,
        max_iterations=args.max_iterations,
        cv_splits=args.cv_splits,
    )
    
    save_tuned_params(best_params, args.output_path)
    print(f"Saved tuned hyperparameters to {args.output_path}")

    # Generate Roadmap Requirements
    tuning_results = []
    selected_reasons = []
    best_params_rows = []

    for name, study in studies.items():
        df_trials = study.trials_dataframe()
        df_trials["model"] = name
        tuning_results.append(df_trials)

        best_score = study.best_value
        direction = "maximize" if "classification" in name else "minimize"
        selected_reasons.append(f"### {name}\nOptuna selected the best parameters by trying to {direction} the objective. The best score achieved was {best_score:.4f}.\nSelected Parameters: {study.best_params}\n")

        for k, v in study.best_params.items():
            best_params_rows.append({"model": name, "parameter": k, "value": v})

    tuning_dir = args.output_path.parent
    tuning_dir.mkdir(parents=True, exist_ok=True)
    
    pd.concat(tuning_results, ignore_index=True).to_csv(tuning_dir / "tuning_results.csv", index=False)
    pd.DataFrame(best_params_rows).to_csv(tuning_dir / "best_params.csv", index=False)
    (tuning_dir / "selected_model_reason.md").write_text("# Selected Model Reason\n\n" + "\n".join(selected_reasons), encoding="utf-8")
    
    print(f"Saved roadmap tuning artifacts to {tuning_dir}/")

if __name__ == "__main__":
    main()
