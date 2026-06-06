"""Generate model evidence pack tables and thesis-ready summary text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, roc_auc_score

from aac_adoption.analysis.animal_profiles import add_animal_descriptors
from aac_adoption.config import RANDOM_STATE
from aac_adoption.dashboard.data import (
    build_profile_prediction_record,
    local_shap_explanations,
    predict_from_record,
    profile_global_shap_reasons,
    similar_historical_cases,
    visibility_need_from_prediction,
)


EVIDENCE_COLUMNS = [
    "section",
    "item",
    "metric",
    "value",
    "interpretation",
]

LIMITATION_COLUMNS = [
    "cohort",
    "value",
    "records",
    "small_cohort_flag",
    "observed_adoption_rate",
    "mean_predicted_adoption_probability",
    "calibration_gap",
    "brier_score",
    "is_reliable",
    "mae",
    "false_positive_rate",
    "false_negative_rate",
]

CI_COLUMNS = ["metric", "animal_subset", "lower", "estimate", "upper", "bootstrap_samples"]

SUBGROUP_CI_COLUMNS = [
    "cohort",
    "value",
    "metric",
    "records",
    "lower",
    "estimate",
    "upper",
    "bootstrap_samples",
    "status",
]

MILESTONE_COLUMNS = [
    "cohort",
    "value",
    "records",
    "adoptions",
    "adoption_rate_pct",
    "adopted_by_day_7_pct",
    "adopted_by_day_30_pct",
    "adopted_by_day_60_pct",
    "adopted_by_day_90_pct",
]

FAILURE_MODE_COLUMNS = [
    "failure_mode",
    "cohort",
    "value",
    "records",
    "metric",
    "value_score",
    "interpretation",
]

JOURNEY_COLUMNS = [
    "profile_label",
    "records",
    "observed_adoption_rate_pct",
    "historical_median_days_to_outcome",
    "predicted_adoption_probability",
    "predicted_days_to_outcome",
    "visibility_label",
    "similar_case_summary",
    "top_shap_reasons",
    "caveat",
]

LOCAL_EXPLANATION_COLUMNS = [
    "example_id",
    "explanation_type",
    "profile_label",
    "records",
    "observed_adoption_rate_pct",
    "predicted_adoption_probability",
    "predicted_days_to_outcome",
    "similar_historical_cases",
    "shap_model_reasons",
    "limitation_note",
]


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _fmt(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def best_model_evidence(classification: pd.DataFrame, regression: pd.DataFrame) -> pd.DataFrame:
    """Create compact best-model evidence rows from comparison tables."""
    rows: list[dict[str, Any]] = []
    if not classification.empty and {"animal_subset", "model_name", "roc_auc"}.issubset(classification.columns):
        for subset, group in classification.dropna(subset=["roc_auc"]).groupby("animal_subset"):
            best = group.sort_values("roc_auc", ascending=False).iloc[0]
            rows.append(
                {
                    "section": "model_choice",
                    "item": f"{subset} classification",
                    "metric": "roc_auc",
                    "value": best["roc_auc"],
                    "interpretation": (
                        f"{best['model_name']} is the strongest ranking model for {subset}; "
                        "compare PR-AUC and calibration before using a decision threshold."
                    ),
                }
            )
            if "pr_auc" in best and not pd.isna(best["pr_auc"]):
                rows.append(
                    {
                        "section": "model_choice",
                        "item": f"{subset} classification",
                        "metric": "pr_auc",
                        "value": best["pr_auc"],
                        "interpretation": "PR-AUC summarizes adoption-positive precision/recall behavior across thresholds.",
                    }
                )
    if not regression.empty and {"animal_subset", "model_name", "mae"}.issubset(regression.columns):
        for subset, group in regression.dropna(subset=["mae"]).groupby("animal_subset"):
            best = group.sort_values("mae", ascending=True).iloc[0]
            rows.append(
                {
                    "section": "model_choice",
                    "item": f"{subset} regression",
                    "metric": "mae",
                    "value": best["mae"],
                    "interpretation": f"{best['model_name']} has the lowest average absolute days-to-outcome error for {subset}.",
                }
            )
    return pd.DataFrame(rows, columns=EVIDENCE_COLUMNS)


def _bootstrap_interval(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan, np.nan, np.nan
    return (
        float(np.nanpercentile(values, 2.5)),
        float(np.nanmean(values)),
        float(np.nanpercentile(values, 97.5)),
    )


def bootstrap_metric_intervals(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int = 200,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Bootstrap core metrics from diagnostic prediction rows."""
    if predictions.empty:
        return pd.DataFrame(columns=CI_COLUMNS)
    required = {
        "classification_target",
        "predicted_adoption_probability",
        "predicted_adopted",
        "regression_target_days",
        "predicted_days_to_outcome",
    }
    if not required.issubset(predictions.columns):
        return pd.DataFrame(columns=CI_COLUMNS)

    rng = np.random.default_rng(random_state)
    metric_values = {"roc_auc": [], "pr_auc": [], "f1_at_0_50": [], "mae": []}
    frame = predictions.reset_index(drop=True)
    
    if "animal_id" in frame.columns and frame["animal_id"].notna().any():
        # Cluster-aware bootstrap
        animals = frame["animal_id"].dropna().unique()
        # Pre-group indices for fast lookup
        id_to_indices = frame.groupby("animal_id").indices
        
        for _ in range(n_bootstrap):
            sampled_ids = rng.choice(animals, size=len(animals), replace=True)
            # Flatten indices
            sample_indices = np.concatenate([id_to_indices[aid] for aid in sampled_ids if aid in id_to_indices])
            sample = frame.iloc[sample_indices]
            
            y_true = sample["classification_target"].astype(int)
            y_score = sample["predicted_adoption_probability"]
            y_pred = sample["predicted_adopted"].astype(int)
            if y_true.nunique() == 2:
                metric_values["roc_auc"].append(roc_auc_score(y_true, y_score))
                metric_values["pr_auc"].append(average_precision_score(y_true, y_score))
            metric_values["f1_at_0_50"].append(f1_score(y_true, y_pred, zero_division=0))
            metric_values["mae"].append(mean_absolute_error(sample["regression_target_days"], sample["predicted_days_to_outcome"]))
    else:
        # Fallback to row-level bootstrap
        for _ in range(n_bootstrap):
            sample = frame.iloc[rng.integers(0, len(frame), len(frame))]
            y_true = sample["classification_target"].astype(int)
            y_score = sample["predicted_adoption_probability"]
            y_pred = sample["predicted_adopted"].astype(int)
            if y_true.nunique() == 2:
                metric_values["roc_auc"].append(roc_auc_score(y_true, y_score))
                metric_values["pr_auc"].append(average_precision_score(y_true, y_score))
            metric_values["f1_at_0_50"].append(f1_score(y_true, y_pred, zero_division=0))
            metric_values["mae"].append(mean_absolute_error(sample["regression_target_days"], sample["predicted_days_to_outcome"]))

    rows = []
    for metric, values in metric_values.items():
        lower, estimate, upper = _bootstrap_interval(np.asarray(values, dtype=float))
        rows.append(
            {
                "metric": metric,
                "animal_subset": "combined",
                "lower": lower,
                "estimate": estimate,
                "upper": upper,
                "bootstrap_samples": len(values),
            }
        )
    return pd.DataFrame(rows, columns=CI_COLUMNS)


def model_limitations_by_cohort(predictions: pd.DataFrame, min_records: int = 100) -> pd.DataFrame:
    """Summarize calibration and errors by available cohort fields."""
    if predictions.empty:
        return pd.DataFrame(columns=LIMITATION_COLUMNS)
    frame = add_animal_descriptors(predictions)
    candidate_columns = [
        "animal_type",
        "age_group",
        "intake_type",
        "intake_condition",
        "health_profile",
        "behavior_support_flag",
        "simplified_breed_group",
        "simplified_color_group",
        "is_named",
        "covid_period",
    ]
    rows: list[dict[str, Any]] = []
    for column in [col for col in candidate_columns if col in frame.columns]:
        for value, group in frame.groupby(column, dropna=False):
            records = len(group)
            observed = float(group["classification_target"].mean())
            predicted = float(group["predicted_adoption_probability"].mean())
            calibration_gap = abs(observed - predicted)
            false_positive = float(((group["predicted_adopted"] == 1) & (group["classification_target"] == 0)).mean())
            false_negative = float(((group["predicted_adopted"] == 0) & (group["classification_target"] == 1)).mean())
            brier_score = float(((group["predicted_adoption_probability"] - group["classification_target"])**2).mean())
            is_reliable = (records >= min_records) and (calibration_gap <= 0.10) and (brier_score <= 0.25)
            rows.append(
                {
                    "cohort": column,
                    "value": value,
                    "records": records,
                    "small_cohort_flag": records < min_records,
                    "observed_adoption_rate": observed,
                    "mean_predicted_adoption_probability": predicted,
                    "calibration_gap": calibration_gap,
                    "brier_score": brier_score,
                    "is_reliable": is_reliable,
                    "mae": float(group["absolute_error"].mean()),
                    "false_positive_rate": false_positive,
                    "false_negative_rate": false_negative,
                }
            )
    result = pd.DataFrame(rows, columns=LIMITATION_COLUMNS)
    if result.empty:
        return result
    return result.sort_values(["small_cohort_flag", "is_reliable", "calibration_gap", "records"], ascending=[True, True, False, False])


def subgroup_reliability(predictions: pd.DataFrame, min_records: int = 100) -> pd.DataFrame:
    """Create the primary subgroup reliability table."""
    return model_limitations_by_cohort(predictions, min_records=min_records)


def subgroup_metric_intervals(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int = 100,
    min_records: int = 100,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Bootstrap metrics by subgroup when enough records and class variety exist."""
    if predictions.empty:
        return pd.DataFrame(columns=SUBGROUP_CI_COLUMNS)
    frame = add_animal_descriptors(predictions)
    candidate_columns = [
        "animal_type",
        "age_group",
        "intake_type",
        "health_profile",
        "simplified_breed_group",
        "simplified_color_group",
        "is_named",
    ]
    rows: list[dict[str, Any]] = []
    for column in [col for col in candidate_columns if col in frame.columns]:
        for value, group in frame.groupby(column, dropna=False):
            records = len(group)
            if records < min_records:
                rows.append(
                    {
                        "cohort": column,
                        "value": value,
                        "metric": "all",
                        "records": records,
                        "lower": np.nan,
                        "estimate": np.nan,
                        "upper": np.nan,
                        "bootstrap_samples": 0,
                        "status": "small_cohort",
                    }
                )
                continue
            intervals = bootstrap_metric_intervals(group, n_bootstrap=n_bootstrap, random_state=random_state)
            if intervals.empty:
                rows.append(
                    {
                        "cohort": column,
                        "value": value,
                        "metric": "all",
                        "records": records,
                        "lower": np.nan,
                        "estimate": np.nan,
                        "upper": np.nan,
                        "bootstrap_samples": 0,
                        "status": "insufficient_schema",
                    }
                )
                continue
            for _, interval in intervals.iterrows():
                status = "ok"
                if pd.isna(interval["estimate"]):
                    status = "insufficient_class_variety"
                rows.append(
                    {
                        "cohort": column,
                        "value": value,
                        "metric": interval["metric"],
                        "records": records,
                        "lower": interval["lower"],
                        "estimate": interval["estimate"],
                        "upper": interval["upper"],
                        "bootstrap_samples": interval["bootstrap_samples"],
                        "status": status,
                    }
                )
    return pd.DataFrame(rows, columns=SUBGROUP_CI_COLUMNS)


def subgroup_adoption_milestones(data_path: str | Path, min_records: int = 50) -> pd.DataFrame:
    """Create descriptive adoption milestone table for key subgroup dimensions."""
    path = Path(data_path)
    if not path.exists():
        return pd.DataFrame(columns=MILESTONE_COLUMNS)
    df = pd.read_csv(path)
    if df.empty or "classification_target" not in df.columns:
        return pd.DataFrame(columns=MILESTONE_COLUMNS)
    df = add_animal_descriptors(df)
    days_column = "days_to_adoption" if "days_to_adoption" in df.columns else "days_to_outcome"
    if days_column not in df.columns:
        return pd.DataFrame(columns=MILESTONE_COLUMNS)
    group_columns = ["animal_type", "age_group", "intake_type", "health_profile"]
    rows: list[dict[str, Any]] = []
    for column in [col for col in group_columns if col in df.columns]:
        for value, group in df.groupby(column, dropna=False):
            records = len(group)
            adopted = group[group["classification_target"].eq(1)]
            adoptions = len(adopted)
            if records < min_records:
                continue
            rows.append(
                {
                    "cohort": column,
                    "value": value,
                    "records": records,
                    "adoptions": adoptions,
                    "adoption_rate_pct": float(group["classification_target"].mean() * 100),
                    "adopted_by_day_7_pct": float((adopted[days_column] <= 7).mean() * 100) if adoptions else 0.0,
                    "adopted_by_day_30_pct": float((adopted[days_column] <= 30).mean() * 100) if adoptions else 0.0,
                    "adopted_by_day_60_pct": float((adopted[days_column] <= 60).mean() * 100) if adoptions else 0.0,
                    "adopted_by_day_90_pct": float((adopted[days_column] <= 90).mean() * 100) if adoptions else 0.0,
                }
            )
    if not rows:
        return pd.DataFrame(columns=MILESTONE_COLUMNS)
    return pd.DataFrame(rows, columns=MILESTONE_COLUMNS).sort_values(["cohort", "records"], ascending=[True, False])


def model_failure_modes(reliability: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Extract top failure-mode rows from subgroup reliability."""
    if reliability.empty:
        return pd.DataFrame(columns=FAILURE_MODE_COLUMNS)
    frame = reliability[~reliability["small_cohort_flag"].astype(bool)].copy()
    if frame.empty:
        return pd.DataFrame(columns=FAILURE_MODE_COLUMNS)
    specs = [
        ("calibration_gap", "calibration_gap", "Predicted probability differs from observed adoption rate."),
        ("high_mae", "mae", "Days-to-outcome error is high for this cohort."),
        ("false_negative_rate", "false_negative_rate", "Adopted animals are sometimes predicted below the default threshold."),
        ("false_positive_rate", "false_positive_rate", "Non-adoption outcomes are sometimes predicted above the default threshold."),
    ]
    rows: list[dict[str, Any]] = []
    for mode, metric, interpretation in specs:
        if metric not in frame.columns:
            continue
        for _, row in frame.sort_values(metric, ascending=False).head(top_n).iterrows():
            rows.append(
                {
                    "failure_mode": mode,
                    "cohort": row["cohort"],
                    "value": row["value"],
                    "records": row["records"],
                    "metric": metric,
                    "value_score": row[metric],
                    "interpretation": interpretation,
                }
            )
    return pd.DataFrame(rows, columns=FAILURE_MODE_COLUMNS)


def shap_family_evidence(shap_family_classification: pd.DataFrame, shap_family_regression: pd.DataFrame) -> pd.DataFrame:
    """Convert SHAP feature-family tables into evidence rows."""
    rows: list[dict[str, Any]] = []
    for task, table in [("classification", shap_family_classification), ("regression", shap_family_regression)]:
        if table.empty or not {"feature_family", "mean_abs_shap"}.issubset(table.columns):
            continue
        for _, row in table.sort_values("mean_abs_shap", ascending=False).head(5).iterrows():
            rows.append(
                {
                    "section": "interpretability",
                    "item": f"{task}: {row['feature_family']}",
                    "metric": "mean_abs_shap",
                    "value": row["mean_abs_shap"],
                    "interpretation": "Feature-family contribution is associated with model prediction, not a causal effect.",
                }
            )
    return pd.DataFrame(rows, columns=EVIDENCE_COLUMNS)


def animal_risk_evidence(vulnerable_profiles: pd.DataFrame) -> pd.DataFrame:
    """Create evidence rows for the highest-priority animal profiles."""
    if vulnerable_profiles.empty:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)
    rows = []
    for _, row in vulnerable_profiles.head(5).iterrows():
        rows.append(
            {
                "section": "animal_profiles",
                "item": row.get("profile_label", "animal profile"),
                "metric": "vulnerability_score",
                "value": row.get("vulnerability_score", np.nan),
                "interpretation": (
                    f"{int(row.get('records', 0))} records; observed adoption rate "
                    f"{_fmt(row.get('adoption_rate_pct'), 1)}%; label {row.get('visibility_need', 'n/a')}."
                ),
            }
        )
    return pd.DataFrame(rows, columns=EVIDENCE_COLUMNS)


def animal_journey_examples(
    animal_archetypes: pd.DataFrame,
    shap_global_classification: pd.DataFrame,
    data_path: str | Path,
    models_dir: str | Path,
    max_examples: int = 5,
) -> pd.DataFrame:
    """Export selected journey-card evidence rows for thesis/demo use."""
    if animal_archetypes.empty:
        return pd.DataFrame(columns=JOURNEY_COLUMNS)
    rows = []
    for _, profile in animal_archetypes.head(max_examples).iterrows():
        record = build_profile_prediction_record(profile)
        prediction: dict[str, float | None] = {"adoption_probability": np.nan, "predicted_days_to_outcome": np.nan}
        local_shap = pd.DataFrame()
        try:
            prediction = predict_from_record(record, models_dir)
            local_shap = local_shap_explanations(record, models_dir, task="classification", top_n=5)
        except (FileNotFoundError, ImportError):
            local_shap = pd.DataFrame()

        if local_shap.empty:
            reasons = profile_global_shap_reasons(profile, shap_global_classification, top_n=5)
            top_reasons = "; ".join(reasons.get("feature", pd.Series(dtype=str)).astype(str).tolist())
        else:
            top_reasons = "; ".join(
                f"{row['feature']}={row['value']} ({row['association']})"
                for _, row in local_shap.iterrows()
            )

        similar = similar_historical_cases(data_path, record)
        similar_summary = "not available"
        if not similar.empty:
            similar_row = similar.iloc[0]
            similar_summary = (
                f"{int(similar_row['similar_records'])} cases; "
                f"{_fmt(similar_row['historical_adoption_rate_pct'], 1)}% adoption; "
                f"median {_fmt(similar_row['median_days_to_outcome'], 1)} days; "
                f"{similar_row['matching_level']}"
            )

        probability = prediction.get("adoption_probability", np.nan)
        days = prediction.get("predicted_days_to_outcome", np.nan)
        label = (
            visibility_need_from_prediction(float(probability), float(days))
            if not pd.isna(probability) and not pd.isna(days)
            else profile.get("visibility_need", "n/a")
        )
        rows.append(
            {
                "profile_label": profile.get("profile_label"),
                "records": profile.get("records"),
                "observed_adoption_rate_pct": profile.get("adoption_rate_pct"),
                "historical_median_days_to_outcome": profile.get("median_days_to_outcome"),
                "predicted_adoption_probability": probability,
                "predicted_days_to_outcome": days,
                "visibility_label": label,
                "similar_case_summary": similar_summary,
                "top_shap_reasons": top_reasons,
                "caveat": "Predictions and SHAP reasons are associated with model behavior, not causal proof.",
            }
        )
    return pd.DataFrame(rows, columns=JOURNEY_COLUMNS)


def local_explanation_examples(journeys: pd.DataFrame) -> pd.DataFrame:
    """Create acceptance-facing local explanation examples from journey cards."""
    if journeys.empty:
        return pd.DataFrame(columns=LOCAL_EXPLANATION_COLUMNS)
    rows = []
    for example_id, (_, row) in enumerate(journeys.iterrows(), start=1):
        rows.append(
            {
                "example_id": example_id,
                "explanation_type": "profile-level SHAP and similar historical cases",
                "profile_label": row.get("profile_label", "animal profile"),
                "records": row.get("records", np.nan),
                "observed_adoption_rate_pct": row.get("observed_adoption_rate_pct", np.nan),
                "predicted_adoption_probability": row.get("predicted_adoption_probability", np.nan),
                "predicted_days_to_outcome": row.get("predicted_days_to_outcome", np.nan),
                "similar_historical_cases": row.get("similar_case_summary", "not available"),
                "shap_model_reasons": row.get("top_shap_reasons", "not available"),
                "limitation_note": (
                    "Illustrative, non-causal example. Similar historical cases and SHAP/model reasons "
                    "describe model behavior and cohort history, not causal effects or individual certainty."
                ),
            }
        )
    return pd.DataFrame(rows, columns=LOCAL_EXPLANATION_COLUMNS)


def _local_explanation_markdown(local_examples: pd.DataFrame) -> str:
    lines = [
        "# Local Explanation Examples",
        "",
        "These examples are illustrative and non-causal. They combine similar historical cases with local SHAP/model reasons to show how the trained model behaves for representative animal profiles.",
        "",
    ]
    if local_examples.empty:
        lines.append("- No local explanation examples are available until animal profile evidence is generated.")
    else:
        for _, row in local_examples.head(5).iterrows():
            lines.append(
                f"- {row['profile_label']}: {row['similar_historical_cases']}; "
                f"model reasons: {row['shap_model_reasons']}."
            )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Similar historical cases summarize past cohorts and may not match a future animal exactly.",
            "- SHAP/model reasons explain associations learned by the model, not causal drivers of adoption.",
            "- Predictions are decision-support evidence and should be reviewed with shelter context before action.",
            "",
        ]
    )
    return "\n".join(lines)


def _summary_markdown(
    evidence: pd.DataFrame,
    limitations: pd.DataFrame,
    intervals: pd.DataFrame,
    journeys: pd.DataFrame,
    subgroup_intervals: pd.DataFrame | None = None,
    subgroup_milestones: pd.DataFrame | None = None,
    failure_modes: pd.DataFrame | None = None,
) -> str:
    lines = [
        "# Model Evidence Pack",
        "",
        "This evidence pack summarizes model choice, uncertainty, reliability limits, SHAP interpretation, and animal-centered examples. SHAP reasons are associated with model behavior and model predictions, not causal effects.",
        "",
        "## Model Choice",
        "",
    ]
    model_rows = evidence[evidence["section"].eq("model_choice")] if not evidence.empty else pd.DataFrame()
    if model_rows.empty:
        lines.append("- Model comparison artifacts are missing.")
    else:
        for _, row in model_rows.head(12).iterrows():
            lines.append(f"- {row['item']}: {row['metric']} = {_fmt(row['value'])}. {row['interpretation']}")

    lines.extend(["", "## Uncertainty", ""])
    if intervals.empty:
        lines.append("- Bootstrap confidence intervals are unavailable until diagnostic predictions exist.")
    else:
        for _, row in intervals.iterrows():
            lines.append(
                f"- {row['metric']} ({row['animal_subset']}): {_fmt(row['estimate'])} "
                f"[{_fmt(row['lower'])}, {_fmt(row['upper'])}] from {int(row['bootstrap_samples'])} bootstrap samples."
            )

    lines.extend(["", "## Trust and Limits", ""])
    if limitations.empty:
        lines.append("- Cohort limitation table is unavailable until diagnostics exist.")
    else:
        for _, row in limitations[~limitations["small_cohort_flag"]].head(8).iterrows():
            lines.append(
                f"- {row['cohort']}={row['value']}: {int(row['records'])} records, "
                f"calibration gap {_fmt(row['calibration_gap'])}, MAE {_fmt(row['mae'], 2)}."
            )

    if failure_modes is not None and not failure_modes.empty:
        lines.extend(["", "Top model failure modes:"])
        for _, row in failure_modes.head(8).iterrows():
            lines.append(
                f"- {row['failure_mode']}: {row['cohort']}={row['value']} "
                f"({int(row['records'])} records, {row['metric']} {_fmt(row['value_score'])})."
            )

    if subgroup_intervals is not None and not subgroup_intervals.empty:
        ok = subgroup_intervals[subgroup_intervals["status"].eq("ok")]
        if not ok.empty:
            lines.extend(["", "Subgroup metric intervals are available for cohorts with enough records and class variety."])

    if subgroup_milestones is not None and not subgroup_milestones.empty:
        lines.extend(["", "## Time-to-Adoption Milestones", ""])
        for _, row in subgroup_milestones.head(8).iterrows():
            lines.append(
                f"- {row['cohort']}={row['value']}: {int(row['adoptions'])} adoptions; "
                f"{_fmt(row['adopted_by_day_30_pct'], 1)}% adopted by day 30 and "
                f"{_fmt(row['adopted_by_day_90_pct'], 1)}% by day 90."
            )

    lines.extend(["", "## SHAP and Animal Stories", ""])
    shap_rows = evidence[evidence["section"].eq("interpretability")] if not evidence.empty else pd.DataFrame()
    if shap_rows.empty:
        lines.append("- SHAP family evidence is unavailable until diagnostics run with `--include-shap`.")
    else:
        for _, row in shap_rows.head(8).iterrows():
            lines.append(f"- {row['item']}: {_fmt(row['value'])}. {row['interpretation']}")
    if not journeys.empty:
        lines.append("")
        lines.append("Animal Journey examples:")
        for _, row in journeys.head(5).iterrows():
            lines.append(f"- {row['profile_label']}: {row['similar_case_summary']}; label {row['visibility_label']}.")

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- CatBoost is included because shelter data is categorical-heavy; histogram gradient boosting may still win when its validation/test ranking is stronger.",
            "- Probability estimates require calibration review before operational threshold use.",
            "- Health and behavior fields are administrative care-context proxies, not complete temperament labels.",
            "- Regression predicts days to outcome, not a full survival model for time to adoption.",
            "",
        ]
    )
    return "\n".join(lines)


def create_evidence_pack(
    data_path: str | Path = "data/processed/modeling_dataset.csv",
    tables_dir: str | Path = "reports/tables",
    diagnostics_dir: str | Path = "reports/diagnostics",
    summary_dir: str | Path = "reports/summary",
    models_dir: str | Path = "models/advanced",
    *,
    bootstrap_samples: int = 200,
    min_cohort_records: int = 100,
    milestone_min_records: int = 50,
) -> dict[str, Path]:
    """Generate evidence pack CSV and Markdown artifacts."""
    tables = Path(tables_dir)
    diagnostics = Path(diagnostics_dir)
    summary = Path(summary_dir)
    tables.mkdir(parents=True, exist_ok=True)
    summary.mkdir(parents=True, exist_ok=True)

    classification = _read_table(tables / "model_comparison_classification.csv")
    regression = _read_table(tables / "model_comparison_regression.csv")
    predictions = _read_table(diagnostics / "diagnostic_predictions_sample.csv")
    shap_family_classification = _read_table(tables / "shap_feature_families_classification.csv")
    shap_family_regression = _read_table(tables / "shap_feature_families_regression.csv")
    shap_global_classification = _read_table(tables / "shap_global_classification.csv")
    vulnerable_profiles = _read_table(tables / "vulnerable_profiles.csv")
    animal_archetypes = _read_table(tables / "animal_archetypes.csv")

    evidence = pd.concat(
        [
            best_model_evidence(classification, regression),
            shap_family_evidence(shap_family_classification, shap_family_regression),
            animal_risk_evidence(vulnerable_profiles),
        ],
        ignore_index=True,
    )
    if evidence.empty:
        evidence = pd.DataFrame(columns=EVIDENCE_COLUMNS)
    limitations = model_limitations_by_cohort(predictions, min_records=min_cohort_records)
    subgroup = subgroup_reliability(predictions, min_records=min_cohort_records)
    intervals = bootstrap_metric_intervals(predictions, n_bootstrap=bootstrap_samples)
    subgroup_intervals = subgroup_metric_intervals(
        predictions,
        n_bootstrap=max(25, min(bootstrap_samples, 100)),
        min_records=min_cohort_records,
    )
    milestones = subgroup_adoption_milestones(data_path, min_records=milestone_min_records)
    failures = model_failure_modes(subgroup)
    journeys = animal_journey_examples(animal_archetypes, shap_global_classification, data_path, models_dir)
    local_examples = local_explanation_examples(journeys)

    paths = {
        "evidence": tables / "model_evidence_pack.csv",
        "limitations": tables / "model_limitations_by_cohort.csv",
        "intervals": tables / "metric_confidence_intervals.csv",
        "subgroup_reliability": tables / "subgroup_reliability.csv",
        "subgroup_intervals": tables / "subgroup_metric_confidence_intervals.csv",
        "subgroup_milestones": tables / "subgroup_adoption_milestones.csv",
        "failure_modes": tables / "model_failure_modes.csv",
        "journeys": tables / "animal_journey_examples.csv",
        "local_explanations": tables / "local_explanation_examples.csv",
        "summary": summary / "model_evidence_pack.md",
        "subgroup_summary": summary / "subgroup_reliability.md",
        "local_explanation_summary": summary / "local_explanation_examples.md",
    }
    evidence.to_csv(paths["evidence"], index=False)
    limitations.to_csv(paths["limitations"], index=False)
    intervals.to_csv(paths["intervals"], index=False)
    subgroup.to_csv(paths["subgroup_reliability"], index=False)
    subgroup_intervals.to_csv(paths["subgroup_intervals"], index=False)
    milestones.to_csv(paths["subgroup_milestones"], index=False)
    failures.to_csv(paths["failure_modes"], index=False)
    journeys.to_csv(paths["journeys"], index=False)
    local_examples.to_csv(paths["local_explanations"], index=False)
    subgroup_text = _summary_markdown(
        evidence,
        subgroup,
        intervals,
        journeys,
        subgroup_intervals=subgroup_intervals,
        subgroup_milestones=milestones,
        failure_modes=failures,
    )
    paths["summary"].write_text(subgroup_text, encoding="utf-8")
    paths["subgroup_summary"].write_text(subgroup_text, encoding="utf-8")
    paths["local_explanation_summary"].write_text(_local_explanation_markdown(local_examples), encoding="utf-8")
    return paths
