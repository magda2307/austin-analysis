"""Generate model reliability, interpretability, and decision-support diagnostics."""

from __future__ import annotations

import json
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
from aac_adoption.features.feature_sets import model_feature_columns
from aac_adoption.models.train_advanced import prepare_catboost_frame


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


_MODEL_ROOTS = {
    "catboost": "advanced",
    "hist_gradient_boosting": "boosting",
    "random_forest": "baseline",
    "logistic_regression": "baseline",
    "ridge": "baseline",
    "dummy_most_frequent": "baseline",
    "dummy_median": "baseline",
}


def _models_root(models_dir: str | Path) -> Path:
    base = Path(models_dir)
    return base.parent if base.name in {"advanced", "boosting", "baseline"} else base


def _selected_model_row(
    tables_dir: str | Path,
    task: str,
    subset: str,
) -> pd.Series | None:
    selection_path = Path(tables_dir) / "final_model_selection.csv"
    if not selection_path.exists():
        return None
    selected = pd.read_csv(selection_path)
    if selected.empty or "selected" not in selected.columns:
        return None
    subset_column = "subset" if "subset" in selected.columns else "animal_subset"
    if subset_column not in selected.columns:
        return None
    mask = (
        selected["task"].astype(str).eq(task)
        & selected[subset_column].astype(str).eq(subset)
        & selected["selected"].astype(str).str.lower().isin(["true", "1", "yes"])
    )
    rows = selected[mask]
    if rows.empty:
        return None
    return rows.iloc[0]


def _candidate_artifact_paths(
    models_dir: str | Path,
    task: str,
    subset: str,
    model_name: str,
    selected_row: pd.Series | None,
) -> list[Path]:
    candidates: list[Path] = []
    if selected_row is not None and "artifact_path" in selected_row and pd.notna(selected_row["artifact_path"]):
        candidates.append(Path(str(selected_row["artifact_path"])))

    base = Path(models_dir)
    root = _models_root(base)
    preferred = _MODEL_ROOTS.get(model_name)
    if preferred:
        candidates.append(artifact_path(root / preferred, task, subset, model_name))
    candidates.append(artifact_path(base, task, subset, model_name))
    for family_dir in ["advanced", "boosting", "baseline"]:
        candidates.append(artifact_path(root / family_dir, task, subset, model_name))

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _load_model(
    models_dir: str | Path,
    task: str,
    subset: str = "combined",
    tables_dir: str | Path = "reports/tables",
) -> tuple[Any, dict[str, Any]]:
    selected_row = _selected_model_row(tables_dir, task, subset)
    model_name = str(selected_row["model_name"]) if selected_row is not None else "catboost"
    tried = _candidate_artifact_paths(models_dir, task, subset, model_name, selected_row)
    for path in tried:
        if path.exists():
            metadata_path = path.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            metadata.update(
                {
                    "model_name": model_name,
                    "task": task,
                    "animal_subset": subset,
                    "artifact_path": str(path),
                    "selection_source": str(Path(tables_dir) / "final_model_selection.csv")
                    if selected_row is not None
                    else "fallback_catboost",
                    "validation_tactic": (
                        "Resolve selected model from final_model_selection.csv, load matching artifact, "
                        "and record artifact path in diagnostics_model_selection.csv."
                    ),
                }
            )
            return joblib.load(path), metadata
    searched = ", ".join(str(path) for path in tried)
    raise FileNotFoundError(f"Missing selected model artifact for {task}/{subset}/{model_name}. Tried: {searched}")


def _feature_columns_from_metadata(metadata: dict[str, Any], fallback: list[str]) -> list[str]:
    columns = metadata.get("feature_columns")
    if isinstance(columns, list) and columns:
        return [str(column) for column in columns]
    return fallback


def _model_frame(model_name: str, df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    if model_name == "catboost":
        return prepare_catboost_frame(df, feature_columns)
    return df[feature_columns].copy()


def _model_selection_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "task",
        "animal_subset",
        "model_name",
        "artifact_path",
        "selection_source",
        "validation_tactic",
    ]
    return pd.DataFrame(rows)[columns]


def diagnostics_validation_tactics(include_shap: bool) -> pd.DataFrame:
    """Return validation tactics for every generated diagnostics part."""
    rows = [
        ("selected_model_resolution", "Check diagnostics_model_selection.csv lists one loaded artifact per task/subset."),
        ("prediction_frame", "Check predictions use test-period rows from make_time_split and selected model artifacts."),
        ("classification_curves", "Check ROC/PR curves use predicted probabilities and classification_target from the same test rows."),
        ("threshold_grid", "Check threshold metrics are diagnostic only; final operating thresholds must come from validation-period analysis."),
        ("calibration_table", "Check observed adoption rate is compared with mean predicted probability by fixed probability bins."),
        ("error_slices", "Check slices below min_slice_records are excluded from interpretation."),
        ("risk_quadrants", "Check quadrants are descriptive summaries of selected classification and regression outputs."),
        ("regression_residuals", "Check residual artifacts compare predicted days to held-out regression_target_days."),
        ("adoption_milestones", "Check milestones use observed adopted-only timelines, not model predictions."),
    ]
    if include_shap:
        rows.append(("shap_outputs", "Generate SHAP only for selected CatBoost models; otherwise write an explicit skip note."))
    return pd.DataFrame(rows, columns=["diagnostic_part", "validation_tactic"])


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
    tables_dir: str | Path,
    subset: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    header = pd.read_csv(data_path, nrows=0)
    parse_dates = [col for col in ["intake_datetime", "outcome_datetime"] if col in header.columns]
    df = pd.read_csv(data_path, parse_dates=parse_dates)

    classification_split = make_time_split(df, "classification_target", animal_subset=subset)
    regression_split = make_time_split(df, "regression_target_days", animal_subset=subset)
    fallback_features = model_feature_columns(classification_split.train)
    classifier, classifier_metadata = _load_model(models_dir, "classification", subset, tables_dir)
    regressor, regressor_metadata = _load_model(models_dir, "regression", subset, tables_dir)
    classification_features = _feature_columns_from_metadata(classifier_metadata, fallback_features)
    regression_features = _feature_columns_from_metadata(regressor_metadata, fallback_features)

    test = classification_split.test.copy()
    test_x = _model_frame(classifier_metadata["model_name"], test, classification_features)
    test["predicted_adoption_probability"] = classifier.predict_proba(test_x)[:, 1]
    test["predicted_adopted"] = (test["predicted_adoption_probability"] >= 0.5).astype(int)
    regression_test = regression_split.test.copy()
    regression_x = _model_frame(regressor_metadata["model_name"], regression_test, regression_features)
    regression_test["predicted_days_to_outcome"] = np.maximum(0.0, regressor.predict(regression_x))
    regression_test["regression_residual"] = (
        regression_test["regression_target_days"] - regression_test["predicted_days_to_outcome"]
    )
    regression_outputs = regression_test[["predicted_days_to_outcome", "regression_residual"]]
    test[["predicted_days_to_outcome", "regression_residual"]] = regression_outputs.reindex(test.index)
    test["absolute_error"] = test["regression_residual"].abs()
    keep = [column for column in DIAGNOSTIC_COLUMNS + [
        "predicted_adoption_probability",
        "predicted_adopted",
        "predicted_days_to_outcome",
        "regression_residual",
        "absolute_error",
    ] if column in test.columns]
    model_selection = _model_selection_table([classifier_metadata, regressor_metadata])
    return test[keep].copy(), model_selection


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
    selection_tables_dir: str | Path,
) -> None:
    """Generate sampled SHAP outputs only when the selected combined model is CatBoost."""
    import shap

    df = pd.read_csv(data_path)
    for task, target_column, filename_suffix in [
        ("classification", "classification_target", "classification"),
        ("regression", "regression_target_days", "regression"),
    ]:
        selected_row = _selected_model_row(selection_tables_dir, task, "combined")
        selected_name = str(selected_row["model_name"]) if selected_row is not None else "catboost"
        if selected_name != "catboost":
            for stale_path in [
                Path(tables_dir) / f"shap_global_{filename_suffix}.csv",
                Path(tables_dir) / f"shap_feature_families_{filename_suffix}.csv",
                Path(tables_dir) / f"feature_family_importance_{filename_suffix}.csv",
                Path(figures_dir) / f"shap_summary_{filename_suffix}.png",
                Path(figures_dir) / f"shap_feature_families_{filename_suffix}.png",
                Path(figures_dir) / f"feature_family_importance_{filename_suffix}.png",
            ]:
                stale_path.unlink(missing_ok=True)
            _save_table(
                pd.DataFrame(
                    [
                        {
                            "task": task,
                            "animal_subset": "combined",
                            "selected_model": selected_name,
                            "status": "skipped",
                            "reason": "SHAP diagnostics are generated only for selected CatBoost models to avoid explaining an unselected artifact.",
                            "validation_tactic": "Confirm skip note matches final_model_selection.csv when selected model is not CatBoost.",
                        }
                    ]
                ),
                Path(tables_dir) / f"shap_{filename_suffix}_skip_note.csv",
            )
            continue
        split = make_time_split(df, target_column, animal_subset="combined")
        feature_columns = model_feature_columns(split.train)
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
        model, _metadata = _load_model(models_dir, task, "combined", selection_tables_dir)
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

    if "sample" not in locals() or "global_table" not in locals():
        return

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
    predictions, model_selection = _prediction_frame(data_path, models_dir, tables, subset)
    _save_table(predictions, diagnostics / "diagnostic_predictions_sample.csv")
    _save_table(model_selection, diagnostics / "diagnostics_model_selection.csv")
    _save_table(diagnostics_validation_tactics(include_shap), diagnostics / "diagnostics_validation_tactics.csv")

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
        shap_outputs(data_path, models_dir, tables, figures, shap_max_rows, tables)

