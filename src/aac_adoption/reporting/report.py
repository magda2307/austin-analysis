"""Generate thesis-ready Markdown summaries and figures from pipeline outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SUBSET_ORDER = ["combined", "dogs", "cats"]


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _format_number(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_days(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    try:
        return f"{float(value):.2f} days"
    except (TypeError, ValueError):
        return str(value)


def _ordered_subsets(values: pd.Series) -> list[str]:
    present = set(values.dropna().astype(str))
    ordered = [subset for subset in SUBSET_ORDER if subset in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _save_grouped_metric_plot(
    df: pd.DataFrame,
    metric: str,
    path: Path,
    title: str,
    ylabel: str,
    lower_is_better: bool = False,
) -> None:
    required = {"animal_subset", "model_name", metric}
    if df.empty or not required.issubset(df.columns):
        return

    plot_df = df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return

    subsets = _ordered_subsets(plot_df["animal_subset"])
    models = list(dict.fromkeys(plot_df["model_name"].astype(str)))
    width = 0.8 / max(len(models), 1)
    x_positions = list(range(len(subsets)))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for index, model in enumerate(models):
        values = []
        for subset in subsets:
            match = plot_df[
                (plot_df["animal_subset"].astype(str) == subset)
                & (plot_df["model_name"].astype(str) == model)
            ]
            values.append(float(match.iloc[0][metric]) if not match.empty else 0.0)
        offset = (index - (len(models) - 1) / 2) * width
        ax.bar([x + offset for x in x_positions], values, width=width, label=model.replace("_", " "))

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(subsets)
    ax.legend(loc="best", fontsize=8)
    if not lower_is_better:
        ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_hypothesis_bar_plot(
    df: pd.DataFrame,
    value_column: str,
    metric: str,
    path: Path,
    title: str,
    ylabel: str,
    variable: str | None = None,
    max_categories: int = 12,
) -> None:
    required = {value_column, metric}
    if df.empty or not required.issubset(df.columns):
        return

    plot_df = df.copy()
    if variable and "variable" in plot_df.columns:
        plot_df = plot_df[plot_df["variable"] == variable]
    plot_df = plot_df.dropna(subset=[metric])
    if plot_df.empty:
        return

    if "records" in plot_df.columns:
        plot_df = plot_df.sort_values("records", ascending=False).head(max_categories)
    else:
        plot_df = plot_df.head(max_categories)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(plot_df[value_column].astype(str), plot_df[metric].astype(float))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _best_rows(df: pd.DataFrame, metric: str, ascending: bool) -> pd.DataFrame:
    if df.empty or metric not in df.columns or "animal_subset" not in df.columns:
        return pd.DataFrame()
    ranked = df.dropna(subset=[metric]).copy()
    ranked["subset_order"] = ranked["animal_subset"].map(
        {subset: index for index, subset in enumerate(SUBSET_ORDER)}
    )
    ranked["subset_order"] = ranked["subset_order"].fillna(len(SUBSET_ORDER))
    ranked = ranked.sort_values(
        ["subset_order", "animal_subset", metric],
        ascending=[True, True, ascending],
    )
    return ranked.groupby("animal_subset", as_index=False).head(1).drop(columns=["subset_order"])


def _summary_lines(
    classification: pd.DataFrame,
    regression: pd.DataFrame,
    context_comparison: pd.DataFrame,
    h1: pd.DataFrame,
    h3: pd.DataFrame,
    h5: pd.DataFrame,
    calibration: pd.DataFrame,
    thresholds: pd.DataFrame,
    risk: pd.DataFrame,
    shap_classification: pd.DataFrame,
    evidence_summary: str = "",
) -> list[str]:
    lines = [
        "# Generated Current Results Summary",
        "",
        "This summary is generated from existing pipeline outputs. Treat these results as reproducible working outputs, not final causal conclusions.",
        "",
        "## Model Comparison",
        "",
    ]

    best_classification = _best_rows(classification, "roc_auc", ascending=False)
    if best_classification.empty:
        lines.append("Classification comparison table was not available.")
    else:
        lines.append("Best classification models by ROC-AUC:")
        lines.append("")
        for _, row in best_classification.iterrows():
            lines.append(
                f"- {row['animal_subset']}: {row['model_name']} "
                f"(ROC-AUC {_format_number(row.get('roc_auc'))}, F1 {_format_number(row.get('f1'))})"
            )

    lines.append("")
    best_regression = _best_rows(regression, "mae", ascending=True)
    if best_regression.empty:
        lines.append("Regression comparison table was not available.")
    else:
        lines.append("Best regression models by MAE:")
        lines.append("")
        for _, row in best_regression.iterrows():
            lines.append(
                f"- {row['animal_subset']}: {row['model_name']} "
                f"(MAE {_format_days(row.get('mae'))}, RMSE {_format_days(row.get('rmse'))})"
            )

    if not context_comparison.empty and {"task", "animal_subset", "model_name", "delta", "higher_is_better"}.issubset(context_comparison.columns):
        lines.extend(["## External Context Feature Test", ""])
        lines.append(
            "Context features use intake-date weather plus prior-window 311 and shelter intake volumes; rolling counts use only dates before intake."
        )
        lines.append("Overall, context effects should be treated as small unless the metric delta is large enough to matter operationally.")
        lines.append("")
        for task, task_rows in context_comparison.sort_values(["task", "animal_subset", "model_name"]).groupby("task"):
            lines.append(f"{task.title()} context deltas:")
            lines.append("")
            for _, row in task_rows.head(8).iterrows():
                if abs(float(row["delta"])) < 0.0005:
                    direction = "changed negligibly"
                elif (row["delta"] > 0 and row["higher_is_better"]) or (row["delta"] < 0 and not row["higher_is_better"]):
                    direction = "improved"
                else:
                    direction = "worsened"
                lines.append(
                    f"- {row['animal_subset']} / {row['model_name']}: "
                    f"context {direction} {row['primary_metric']} by {_format_number(abs(row['delta']))}."
                )
            lines.append("")

    lines.extend(["## Hypothesis Signals", ""])

    if not h1.empty and {"variable", "records", "adoption_rate_pct"}.issubset(h1.columns):
        intake_rows = h1[h1["variable"] == "intake_type"].sort_values("records", ascending=False)
        if not intake_rows.empty:
            lines.append("H1 intake-type patterns:")
            lines.append("")
            for _, row in intake_rows.head(5).iterrows():
                lines.append(
                    f"- {row['value']}: {int(row['records'])} records, "
                    f"{_format_number(row['adoption_rate_pct'], 1)}% adoption rate"
                )
            lines.append("")

    if not h3.empty and {"value", "adoption_rate_pct", "median_days_to_outcome"}.issubset(h3.columns):
        lines.append("H3 age-group patterns:")
        lines.append("")
        for _, row in h3.head(5).iterrows():
            lines.append(
                f"- {row['value']}: {_format_number(row['adoption_rate_pct'], 1)}% adoption rate, "
                f"median outcome time {_format_days(row['median_days_to_outcome'])}"
            )
        lines.append("")

    if not h5.empty and {"value", "adoption_rate_pct", "median_days_to_outcome"}.issubset(h5.columns):
        lines.append("H5 COVID-period patterns:")
        lines.append("")
        for _, row in h5.head(5).iterrows():
            lines.append(
                f"- {row['value']}: {_format_number(row['adoption_rate_pct'], 1)}% adoption rate, "
                f"median outcome time {_format_days(row['median_days_to_outcome'])}"
            )
        lines.append("")

    if not calibration.empty and {"mean_predicted_probability", "observed_adoption_rate"}.issubset(calibration.columns):
        valid = calibration.dropna(subset=["mean_predicted_probability", "observed_adoption_rate"])
        if not valid.empty:
            mean_gap = (valid["observed_adoption_rate"] - valid["mean_predicted_probability"]).abs().mean()
            lines.extend(
                [
                    "## Reliability Diagnostics",
                    "",
                    f"- Mean absolute calibration gap across probability bins: {_format_number(mean_gap, 3)}.",
                ]
            )
            if not thresholds.empty and {"threshold", "precision", "recall", "f1"}.issubset(thresholds.columns):
                default_threshold = thresholds.iloc[(thresholds["threshold"] - 0.5).abs().argsort()[:1]]
                if not default_threshold.empty:
                    row = default_threshold.iloc[0]
                    lines.append(
                        f"- At threshold {_format_number(row['threshold'], 2)}, precision is {_format_number(row['precision'])}, "
                        f"recall is {_format_number(row['recall'])}, and F1 is {_format_number(row['f1'])}."
                    )
            if not risk.empty and {"risk_quadrant", "records"}.issubset(risk.columns):
                top_risk = risk.sort_values("records", ascending=False).iloc[0]
                lines.append(f"- Largest placement-risk quadrant: {top_risk['risk_quadrant']} ({int(top_risk['records'])} records).")
            lines.append("")

    if not shap_classification.empty and {"feature", "mean_abs_shap", "feature_family"}.issubset(shap_classification.columns):
        top = shap_classification.sort_values("mean_abs_shap", ascending=False).head(5)
        lines.extend(["## Interpretability Signals", ""])
        for _, row in top.iterrows():
            lines.append(
                f"- {row['feature']} ({row['feature_family']}): mean absolute SHAP {_format_number(row['mean_abs_shap'])}."
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation Guardrails",
            "",
            "- Use model outputs as predictive association evidence, not proof of causal effects.",
            "- Emphasize the time-aware train/validation/test split when discussing evaluation.",
            "- Keep H1, H3, and H5 central; use H2 and H4 as descriptive supporting analyses unless stronger tests are added.",
            "- Regression should be discussed primarily through MAE because it is easiest to explain operationally.",
            "",
        ]
    )
    if evidence_summary:
        lines.extend(
            [
                "## Model Evidence Pack",
                "",
                "A separate evidence pack has been generated for model choice, uncertainty, cohort limits, SHAP interpretation, and animal journey examples.",
                "",
            ]
        )
        evidence_lines = [
            line
            for line in evidence_summary.splitlines()
            if not line.startswith("# Model Evidence Pack")
        ]
        while evidence_lines and not evidence_lines[0].strip():
            evidence_lines.pop(0)
        lines.extend(evidence_lines[:100])
        lines.append("")
    return lines


def create_report_outputs(
    tables_dir: str | Path = "reports/tables",
    figures_dir: str | Path = "reports/figures",
    summary_dir: str | Path = "reports/summary",
) -> Path:
    """Create Markdown and figure outputs from existing analysis tables."""
    tables = Path(tables_dir)
    figures = Path(figures_dir)
    summary = Path(summary_dir)
    summary.mkdir(parents=True, exist_ok=True)

    classification = _read_table(tables / "model_comparison_classification.csv")
    regression = _read_table(tables / "model_comparison_regression.csv")
    context_comparison = _read_table(tables / "context_model_comparison.csv")
    h1 = _read_table(tables / "h1_intake_vs_appearance.csv")
    h3 = _read_table(tables / "h3_age_adoption_speed.csv")
    h5 = _read_table(tables / "h5_covid_period.csv")
    diagnostics = tables.parent / "diagnostics"
    calibration = _read_table(diagnostics / "classification_calibration.csv")
    thresholds = _read_table(diagnostics / "classification_thresholds.csv")
    risk = _read_table(diagnostics / "placement_risk_quadrants.csv")
    shap_classification = _read_table(tables / "shap_global_classification.csv")
    evidence_summary_path = summary / "model_evidence_pack.md"
    evidence_summary = evidence_summary_path.read_text(encoding="utf-8") if evidence_summary_path.exists() else ""

    _save_grouped_metric_plot(
        classification,
        "roc_auc",
        figures / "model_comparison_classification_roc_auc.png",
        "Classification ROC-AUC by model and subset",
        "ROC-AUC",
    )
    _save_grouped_metric_plot(
        classification,
        "f1",
        figures / "model_comparison_classification_f1.png",
        "Classification F1 by model and subset",
        "F1",
    )
    _save_grouped_metric_plot(
        regression,
        "mae",
        figures / "model_comparison_regression_mae.png",
        "Regression MAE by model and subset",
        "MAE in days",
        lower_is_better=True,
    )
    _save_grouped_metric_plot(
        context_comparison,
        "delta",
        figures / "context_model_delta.png",
        "Context feature delta by model and subset",
        "Context minus base metric delta",
        lower_is_better=True,
    )
    _save_grouped_metric_plot(
        regression,
        "rmse",
        figures / "model_comparison_regression_rmse.png",
        "Regression RMSE by model and subset",
        "RMSE in days",
        lower_is_better=True,
    )

    _save_hypothesis_bar_plot(
        h1,
        "value",
        "adoption_rate_pct",
        figures / "h1_intake_type_adoption_rate.png",
        "H1 adoption rate by intake type",
        "Adoption rate (%)",
        variable="intake_type",
    )
    _save_hypothesis_bar_plot(
        h1,
        "value",
        "adoption_rate_pct",
        figures / "h1_intake_condition_adoption_rate.png",
        "H1 adoption rate by intake condition",
        "Adoption rate (%)",
        variable="intake_condition",
    )
    _save_hypothesis_bar_plot(
        h3,
        "value",
        "adoption_rate_pct",
        figures / "h3_age_group_adoption_rate.png",
        "H3 adoption rate by age group",
        "Adoption rate (%)",
    )
    _save_hypothesis_bar_plot(
        h3,
        "value",
        "median_days_to_outcome",
        figures / "h3_age_group_median_days.png",
        "H3 median outcome time by age group",
        "Median days to outcome",
    )
    _save_hypothesis_bar_plot(
        h5,
        "value",
        "adoption_rate_pct",
        figures / "h5_covid_period_adoption_rate.png",
        "H5 adoption rate by COVID period",
        "Adoption rate (%)",
    )
    _save_hypothesis_bar_plot(
        h5,
        "value",
        "median_days_to_outcome",
        figures / "h5_covid_period_median_days.png",
        "H5 median outcome time by COVID period",
        "Median days to outcome",
    )

    summary_path = summary / "current_results.md"
    summary_path.write_text(
        "\n".join(
            _summary_lines(
                classification,
                regression,
                context_comparison,
                h1,
                h3,
                h5,
                calibration,
                thresholds,
                risk,
                shap_classification,
                evidence_summary,
            )
        ),
        encoding="utf-8",
    )
    return summary_path
