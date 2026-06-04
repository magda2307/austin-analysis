"""Generate model reliability, interpretability, and decision-support diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from aac_adoption.config import RANDOM_STATE
from aac_adoption.diagnostics.feature_families import feature_family
from aac_adoption.models.artifacts import artifact_path
from aac_adoption.models.split import make_time_split
from aac_adoption.models.train_advanced import feature_columns_for, prepare_catboost_frame


DIAGNOSTIC_COLUMNS = [
    "animal_id",
    "animal_type",
    "intake_type",
    "intake_condition",
    "age_group",
    "covid_period",
    "simplified_breed_group",
    "simplified_color_group",
    "is_black_or_dark",
    "is_named",
    "classification_target",
    "regression_target_days",
]


def _load_model(models_dir: str | Path, task: str, subset: str = "combined"):
    path = artifact_path(models_dir, task, subset, "catboost")
    if not path.exists():
        raise FileNotFoundError(f"Missing advanced model artifact: {path}")
    return joblib.load(path)


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_line_plot(df: pd.DataFrame, x: str, y_columns: list[str], path: Path, title: str, ylabel: str) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for column in y_columns:
        if column in df.columns:
            ax.plot(df[x], df[column], marker="o", label=column.replace("_", " "))
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_bar_plot(df: pd.DataFrame, label: str, value: str, path: Path, title: str, ylabel: str) -> None:
    if df.empty or label not in df.columns or value not in df.columns:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = df.head(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(plot_df[label].astype(str), plot_df[value].astype(float))
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _prediction_frame(
    data_path: str | Path,
    models_dir: str | Path,
    subset: str,
) -> pd.DataFrame:
    header = pd.read_csv(data_path, nrows=0)
    parse_dates = [col for col in ["intake_datetime", "outcome_datetime"] if col in header.columns]
    df = pd.read_csv(data_path, parse_dates=parse_dates)

    classification_split = make_time_split(df, "classification_target", animal_subset=subset)
    regression_split = make_time_split(df, "regression_target_days", animal_subset=subset)
    feature_columns = feature_columns_for(classification_split.train)
    classifier = _load_model(models_dir, "classification", subset)
    regressor = _load_model(models_dir, "regression", subset)

    test = classification_split.test.copy()
    test_x = prepare_catboost_frame(test, feature_columns)
    test["predicted_adoption_probability"] = classifier.predict_proba(test_x)[:, 1]
    test["predicted_adopted"] = (test["predicted_adoption_probability"] >= 0.5).astype(int)
    regression_x = prepare_catboost_frame(regression_split.test, feature_columns)
    test["predicted_days_to_outcome"] = np.maximum(0.0, regressor.predict(regression_x))
    test["regression_residual"] = test["regression_target_days"] - test["predicted_days_to_outcome"]
    test["absolute_error"] = test["regression_residual"].abs()
    keep = [column for column in DIAGNOSTIC_COLUMNS + [
        "predicted_adoption_probability",
        "predicted_adopted",
        "predicted_days_to_outcome",
        "regression_residual",
        "absolute_error",
    ] if column in test.columns]
    return test[keep].copy()


def classification_curves(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ROC and precision-recall curve tables."""
    y_true = predictions["classification_target"]
    y_score = predictions["predicted_adoption_probability"]
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    roc = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds, "auc": auc(fpr, tpr)})
    pr = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": list(pr_thresholds) + [np.nan],
            "auc": auc(recall, precision),
        }
    )
    return roc, pr


def threshold_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Create adoption threshold tradeoff table."""
    rows: list[dict[str, Any]] = []
    y_true = predictions["classification_target"].astype(int)
    y_score = predictions["predicted_adoption_probability"]
    for threshold in np.round(np.arange(0.05, 1.0, 0.05), 2):
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "flagged_for_adoption_share": float(y_pred.mean()),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return pd.DataFrame(rows)


def calibration_table(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Create fixed-bin probability calibration table."""
    result = predictions.copy()
    result["probability_bin"] = pd.cut(
        result["predicted_adoption_probability"],
        bins=np.linspace(0, 1, bins + 1),
        include_lowest=True,
    )
    table = (
        result.groupby("probability_bin", observed=False)
        .agg(
            records=("classification_target", "count"),
            mean_predicted_probability=("predicted_adoption_probability", "mean"),
            observed_adoption_rate=("classification_target", "mean"),
        )
        .reset_index()
    )
    table["probability_bin"] = table["probability_bin"].astype(str)
    return table


def error_slices(predictions: pd.DataFrame, min_records: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create classification and regression error-slice tables."""
    slice_columns = [
        "animal_type",
        "age_group",
        "intake_type",
        "intake_condition",
        "covid_period",
        "simplified_color_group",
        "is_black_or_dark",
    ]
    classification_rows: list[pd.DataFrame] = []
    regression_rows: list[pd.DataFrame] = []
    for column in [col for col in slice_columns if col in predictions.columns]:
        grouped = predictions.groupby(column, dropna=False)
        cls = grouped.apply(
            lambda group: pd.Series(
                {
                    "records": len(group),
                    "adoption_rate": group["classification_target"].mean(),
                    "predicted_adoption_probability": group["predicted_adoption_probability"].mean(),
                    "false_positive_rate": ((group["predicted_adopted"] == 1) & (group["classification_target"] == 0)).mean(),
                    "false_negative_rate": ((group["predicted_adopted"] == 0) & (group["classification_target"] == 1)).mean(),
                }
            ),
            include_groups=False,
        ).reset_index(names="value")
        cls["slice"] = column
        classification_rows.append(cls)

        reg = grouped["absolute_error"].agg(records="count", mae="mean", median_absolute_error="median").reset_index(names="value")
        reg["slice"] = column
        regression_rows.append(reg)

    classification = pd.concat(classification_rows, ignore_index=True)
    regression = pd.concat(regression_rows, ignore_index=True)
    classification = classification[classification["records"] >= min_records].sort_values(
        ["false_negative_rate", "records"],
        ascending=[False, False],
    )
    regression = regression[regression["records"] >= min_records].sort_values(["mae", "records"], ascending=[False, False])
    return classification, regression


def placement_risk_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Create practical risk quadrant table from two model outputs."""
    result = predictions.copy()
    result["risk_quadrant"] = np.select(
        [
            (result["predicted_adoption_probability"] >= 0.5) & (result["predicted_days_to_outcome"] < 14),
            (result["predicted_adoption_probability"] >= 0.5) & (result["predicted_days_to_outcome"] >= 14),
            (result["predicted_adoption_probability"] < 0.5) & (result["predicted_days_to_outcome"] >= 14),
        ],
        [
            "likely_quick_placement",
            "adoptable_needs_visibility",
            "long_stay_risk",
        ],
        default="non_adoption_or_fast_exit_risk",
    )
    return (
        result.groupby("risk_quadrant")
        .agg(
            records=("classification_target", "count"),
            observed_adoption_rate=("classification_target", "mean"),
            mean_predicted_adoption_probability=("predicted_adoption_probability", "mean"),
            median_predicted_days_to_outcome=("predicted_days_to_outcome", "median"),
        )
        .reset_index()
        .sort_values("records", ascending=False)
    )


def adoption_milestones(data_path: str | Path, tables_dir: str | Path, figures_dir: str | Path) -> pd.DataFrame:
    """Create survival-style adoption milestone table and cumulative figure."""
    df = pd.read_csv(data_path)
    adopted = df[df["classification_target"].eq(1)].copy()
    rows: list[dict[str, Any]] = []
    for group_col in ["age_group", "intake_type"]:
        if group_col not in adopted.columns:
            continue
        for value, group in adopted.groupby(group_col, dropna=False):
            rows.append(
                {
                    "group": group_col,
                    "value": value,
                    "adoptions": len(group),
                    "adopted_by_day_7_pct": float((group["days_to_adoption"] <= 7).mean() * 100),
                    "adopted_by_day_30_pct": float((group["days_to_adoption"] <= 30).mean() * 100),
                    "adopted_by_day_90_pct": float((group["days_to_adoption"] <= 90).mean() * 100),
                }
            )
    table = pd.DataFrame(rows).sort_values(["group", "adoptions"], ascending=[True, False])
    _save_table(table, Path(tables_dir) / "adoption_by_day_milestones.csv")
    age_table = table[table["group"].eq("age_group")].head(8)
    _save_line_plot(
        age_table,
        "value",
        ["adopted_by_day_7_pct", "adopted_by_day_30_pct", "adopted_by_day_90_pct"],
        Path(figures_dir) / "adoption_cumulative_curves.png",
        "Adoption timeline milestones by age group",
        "Share adopted (%)",
    )
    return table


def shap_outputs(
    data_path: str | Path,
    models_dir: str | Path,
    tables_dir: str | Path,
    figures_dir: str | Path,
    max_rows: int,
) -> None:
    """Generate sampled SHAP global and feature-family outputs for combined CatBoost models."""
    import shap

    df = pd.read_csv(data_path)
    for task, target_column, filename_suffix in [
        ("classification", "classification_target", "classification"),
        ("regression", "regression_target_days", "regression"),
    ]:
        split = make_time_split(df, target_column, animal_subset="combined")
        feature_columns = feature_columns_for(split.train)
        sample = split.test
        if len(sample) > max_rows:
            if target_column == "classification_target" and sample[target_column].nunique() == 2:
                sample = sample.groupby(target_column, group_keys=False).sample(
                    frac=max_rows / len(sample),
                    random_state=RANDOM_STATE,
                )
            else:
                sample = sample.sample(n=max_rows, random_state=RANDOM_STATE)
        sample_x = prepare_catboost_frame(sample, feature_columns)
        model = _load_model(models_dir, task, "combined")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_x)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        values = np.asarray(shap_values)
        global_table = (
            pd.DataFrame(
                {
                    "feature": feature_columns,
                    "mean_abs_shap": np.abs(values).mean(axis=0),
                    "mean_shap": values.mean(axis=0),
                    "task": task,
                    "animal_subset": "combined",
                    "sample_rows": len(sample_x),
                }
            )
            .assign(feature_family=lambda frame: frame["feature"].map(feature_family))
            .sort_values("mean_abs_shap", ascending=False)
        )
        _save_table(global_table, Path(tables_dir) / f"shap_global_{filename_suffix}.csv")
        family = (
            global_table.groupby("feature_family")
            .agg(mean_abs_shap=("mean_abs_shap", "sum"), features=("feature", "count"))
            .reset_index()
            .sort_values("mean_abs_shap", ascending=False)
        )
        family["task"] = task
        _save_table(family, Path(tables_dir) / f"shap_feature_families_{filename_suffix}.csv")
        _save_bar_plot(
            global_table,
            "feature",
            "mean_abs_shap",
            Path(figures_dir) / f"shap_summary_{filename_suffix}.png",
            f"SHAP global importance ({task})",
            "Mean absolute SHAP",
        )
        _save_bar_plot(
            family,
            "feature_family",
            "mean_abs_shap",
            Path(figures_dir) / f"shap_feature_families_{filename_suffix}.png",
            f"SHAP feature-family importance ({task})",
            "Sum mean absolute SHAP",
        )

    local_rows = []
    for row_index, row in sample.head(5).reset_index(drop=True).iterrows():
        top = global_table.head(5)
        local_rows.append(
            {
                "example_id": row_index,
                "animal_type": row.get("animal_type"),
                "age_group": row.get("age_group"),
                "intake_type": row.get("intake_type"),
                "top_associated_features": "; ".join(top["feature"].astype(str).tolist()),
                "note": "Top factors are associated with model prediction, not causal effects.",
            }
        )
    _save_table(pd.DataFrame(local_rows), Path(tables_dir) / "shap_local_examples.csv")


def generate_diagnostics(
    data_path: str | Path,
    models_dir: str | Path = "models/advanced",
    diagnostics_dir: str | Path = "reports/diagnostics",
    tables_dir: str | Path = "reports/tables",
    figures_dir: str | Path = "reports/figures",
    subset: str = "combined",
    include_shap: bool = False,
    shap_max_rows: int = 5000,
    min_slice_records: int = 100,
) -> None:
    """Generate reliability, error, risk, timeline, and optional SHAP outputs."""
    diagnostics = Path(diagnostics_dir)
    tables = Path(tables_dir)
    figures = Path(figures_dir)
    predictions = _prediction_frame(data_path, models_dir, subset)
    _save_table(predictions, diagnostics / "diagnostic_predictions_sample.csv")

    roc, pr = classification_curves(predictions)
    thresholds = threshold_table(predictions)
    calibration = calibration_table(predictions)
    cls_slices, reg_slices = error_slices(predictions, min_records=min_slice_records)
    risk = placement_risk_table(predictions)

    _save_table(roc, diagnostics / "classification_roc_curve.csv")
    _save_table(pr, diagnostics / "classification_precision_recall_curve.csv")
    _save_table(thresholds, diagnostics / "classification_thresholds.csv")
    _save_table(calibration, diagnostics / "classification_calibration.csv")
    _save_table(cls_slices, diagnostics / "classification_error_slices.csv")
    _save_table(reg_slices, diagnostics / "regression_error_slices.csv")
    _save_table(risk, diagnostics / "placement_risk_quadrants.csv")
    _save_table(predictions.sample(n=min(len(predictions), 3000), random_state=RANDOM_STATE), diagnostics / "regression_residuals_sample.csv")

    _save_line_plot(roc, "fpr", ["tpr"], figures / "diagnostic_roc_curve.png", "ROC curve", "True positive rate")
    _save_line_plot(pr, "recall", ["precision"], figures / "diagnostic_precision_recall_curve.png", "Precision-recall curve", "Precision")
    _save_line_plot(
        calibration,
        "mean_predicted_probability",
        ["observed_adoption_rate"],
        figures / "diagnostic_calibration_curve.png",
        "Probability calibration",
        "Observed adoption rate",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(predictions["regression_residual"], bins=40)
    ax.set_title("Regression residual distribution")
    ax.set_xlabel("Actual days - predicted days")
    ax.set_ylabel("Records")
    fig.tight_layout()
    fig.savefig(figures / "diagnostic_regression_residuals.png", dpi=150)
    plt.close(fig)

    sample = predictions.sample(n=min(len(predictions), 3000), random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sample["regression_target_days"], sample["predicted_days_to_outcome"], s=8, alpha=0.35)
    limit = max(sample["regression_target_days"].max(), sample["predicted_days_to_outcome"].max())
    ax.plot([0, limit], [0, limit], linestyle="--", color="black", alpha=0.6)
    ax.set_title("Predicted vs actual days to outcome")
    ax.set_xlabel("Actual days")
    ax.set_ylabel("Predicted days")
    fig.tight_layout()
    fig.savefig(figures / "diagnostic_predicted_vs_actual.png", dpi=150)
    plt.close(fig)

    _save_bar_plot(reg_slices, "value", "mae", figures / "diagnostic_regression_error_slices.png", "Highest MAE slices", "MAE")
    _save_bar_plot(cls_slices, "value", "false_negative_rate", figures / "diagnostic_classification_error_slices.png", "Highest false-negative slices", "False negative rate")
    adoption_milestones(data_path, tables, figures)
    if include_shap:
        shap_outputs(data_path, models_dir, tables, figures, shap_max_rows)

