"""Calibrate trained classifiers on the validation split."""

import argparse
from pathlib import Path
import sys

from aac_adoption.models.calibrate import calibrate_classifiers


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Calibrate trained classifiers.")
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
        default=Path("models/calibrated"),
        help="Directory to save calibrated model artifacts",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit rows for quick testing",
    )

    args = parser.parse_args()

    if not args.data_path.exists():
        print(f"Error: {args.data_path} not found.")
        sys.exit(1)

    print(f"Calibrating classifiers from models/advanced and models/boosting...")
    
    source_artifacts = [
        ("models/advanced", "catboost"),
        ("models/boosting", "hist_gradient_boosting"),
    ]

    outputs = calibrate_classifiers(
        data_path=args.data_path,
        source_artifacts=source_artifacts,
        metrics_dir=args.metrics_dir,
        models_dir=args.models_dir,
        max_rows=args.max_rows,
    )
    
    if not outputs.classification_metrics.empty:
        print("\nCalibrated Classification Metrics:")
        cols = ["animal_subset", "model_name", "pr_auc", "roc_auc", "brier_score", "expected_calibration_error"]
        print(outputs.classification_metrics[[c for c in cols if c in outputs.classification_metrics.columns]].to_string(index=False))
    else:
        print("\nNo models were calibrated. (Were the source artifacts missing?)")

    print(f"\nSaved metrics to {args.metrics_dir}")
    print(f"Saved artifacts to {args.models_dir}")


if __name__ == "__main__":
    main()
