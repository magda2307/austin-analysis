"""Model comparison tables for baseline and boosting runs."""

from pathlib import Path

import pandas as pd


def _read_existing(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def create_model_comparison_tables(
    metrics_dir: str | Path = "reports/metrics",
    tables_dir: str | Path = "reports/tables",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create classification and regression model comparison tables."""
    metrics = Path(metrics_dir)
    tables = Path(tables_dir)
    tables.mkdir(parents=True, exist_ok=True)

    classification = _read_existing(
        [
            metrics / "classification_metrics.csv",
            metrics / "boosting_classification_metrics.csv",
        ]
    )
    regression = _read_existing(
        [
            metrics / "regression_metrics.csv",
            metrics / "boosting_regression_metrics.csv",
        ]
    )

    if not classification.empty:
        classification["roc_auc_rank"] = classification.groupby("animal_subset")["roc_auc"].rank(
            ascending=False,
            method="min",
        )
        classification["f1_rank"] = classification.groupby("animal_subset")["f1"].rank(
            ascending=False,
            method="min",
        )
        classification = classification.sort_values(
            ["animal_subset", "roc_auc", "f1"],
            ascending=[True, False, False],
        )
        classification.to_csv(tables / "model_comparison_classification.csv", index=False)

    if not regression.empty:
        regression["mae_rank"] = regression.groupby("animal_subset")["mae"].rank(
            ascending=True,
            method="min",
        )
        regression["rmse_rank"] = regression.groupby("animal_subset")["rmse"].rank(
            ascending=True,
            method="min",
        )
        regression = regression.sort_values(
            ["animal_subset", "mae", "rmse"],
            ascending=[True, True, True],
        )
        regression.to_csv(tables / "model_comparison_regression.csv", index=False)

    return classification, regression

