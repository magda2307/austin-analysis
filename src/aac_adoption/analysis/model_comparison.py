"""Model comparison tables and plots for baseline and boosting runs."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _read_existing(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _context_delta_table(classification: pd.DataFrame, regression: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if not classification.empty and {"feature_set", "animal_subset", "model_name", "roc_auc", "f1"}.issubset(classification.columns):
        for keys, group in classification.groupby(["animal_subset", "model_name"], dropna=False):
            by_feature = group.dropna(subset=["feature_set"]).set_index("feature_set")
            if {"intake_time_v1", "intake_time_context_v1"}.issubset(by_feature.index):
                base = by_feature.loc["intake_time_v1"].iloc[0] if isinstance(by_feature.loc["intake_time_v1"], pd.DataFrame) else by_feature.loc["intake_time_v1"]
                context = by_feature.loc["intake_time_context_v1"].iloc[0] if isinstance(by_feature.loc["intake_time_context_v1"], pd.DataFrame) else by_feature.loc["intake_time_context_v1"]
                rows.append(
                    {
                        "task": "classification",
                        "animal_subset": keys[0],
                        "model_name": keys[1],
                        "primary_metric": "roc_auc",
                        "base_score": base["roc_auc"],
                        "context_score": context["roc_auc"],
                        "delta": context["roc_auc"] - base["roc_auc"],
                        "secondary_metric": "f1",
                        "secondary_base_score": base["f1"],
                        "secondary_context_score": context["f1"],
                        "secondary_delta": context["f1"] - base["f1"],
                        "higher_is_better": True,
                    }
                )
    if not regression.empty and {"feature_set", "animal_subset", "model_name", "mae", "rmse"}.issubset(regression.columns):
        for keys, group in regression.groupby(["animal_subset", "model_name"], dropna=False):
            by_feature = group.dropna(subset=["feature_set"]).set_index("feature_set")
            if {"intake_time_v1", "intake_time_context_v1"}.issubset(by_feature.index):
                base = by_feature.loc["intake_time_v1"].iloc[0] if isinstance(by_feature.loc["intake_time_v1"], pd.DataFrame) else by_feature.loc["intake_time_v1"]
                context = by_feature.loc["intake_time_context_v1"].iloc[0] if isinstance(by_feature.loc["intake_time_context_v1"], pd.DataFrame) else by_feature.loc["intake_time_context_v1"]
                rows.append(
                    {
                        "task": "regression",
                        "animal_subset": keys[0],
                        "model_name": keys[1],
                        "primary_metric": "mae",
                        "base_score": base["mae"],
                        "context_score": context["mae"],
                        "delta": context["mae"] - base["mae"],
                        "secondary_metric": "rmse",
                        "secondary_base_score": base["rmse"],
                        "secondary_context_score": context["rmse"],
                        "secondary_delta": context["rmse"] - base["rmse"],
                        "higher_is_better": False,
                    }
                )
    return pd.DataFrame(rows)


def _plot_pr_auc(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty or "pr_auc" not in df.columns:
        return

    plot_df = df.dropna(subset=["pr_auc"]).copy()
    if plot_df.empty:
        return

    subsets = ["combined", "dogs", "cats"]
    present_subsets = [s for s in subsets if s in plot_df["animal_subset"].unique()]
    models = list(dict.fromkeys(plot_df["model_name"].astype(str)))
    width = 0.8 / max(len(models), 1)
    x_positions = list(range(len(present_subsets)))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, model in enumerate(models):
        values = []
        for subset in present_subsets:
            match = plot_df[
                (plot_df["animal_subset"].astype(str) == subset)
                & (plot_df["model_name"].astype(str) == model)
            ]
            values.append(float(match.iloc[0]["pr_auc"]) if not match.empty else 0.0)
        offset = (index - (len(models) - 1) / 2) * width
        ax.bar([x + offset for x in x_positions], values, width=width, label=model.replace("_", " "))

    ax.set_title("Classification PR-AUC by model and subset")
    ax.set_ylabel("PR-AUC")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(present_subsets)
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(bottom=0, top=1.0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
            metrics / "advanced_classification_metrics.csv",
            metrics / "calibrated_classification_metrics.csv",
        ]
    )
    regression = _read_existing(
        [
            metrics / "regression_metrics.csv",
            metrics / "boosting_regression_metrics.csv",
            metrics / "advanced_regression_metrics.csv",
        ]
    )

    if not classification.empty:
        classification["roc_auc_rank"] = classification.groupby("animal_subset")["roc_auc"].rank(
            ascending=False,
            method="min",
        )
        if "pr_auc" in classification.columns:
            classification["pr_auc_rank"] = classification.groupby("animal_subset")["pr_auc"].rank(
                ascending=False,
                method="min",
            )
        classification["f1_rank"] = classification.groupby("animal_subset")["f1"].rank(
            ascending=False,
            method="min",
        )
        sort_cols = ["animal_subset", "roc_auc", "f1"]
        sort_asc = [True, False, False]
        if "pr_auc" in classification.columns:
            sort_cols = ["animal_subset", "pr_auc", "roc_auc", "f1"]
            sort_asc = [True, False, False, False]
        classification = classification.sort_values(
            sort_cols,
            ascending=sort_asc,
        )
        classification.to_csv(tables / "model_comparison_classification.csv", index=False)

        # Plot PR-AUC
        figures_dir = tables.parent / "figures"
        _plot_pr_auc(classification, figures_dir / "model_comparison_classification_pr_auc.png")

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

    context_delta = _context_delta_table(classification, regression)
    if not context_delta.empty:
        context_delta.to_csv(tables / "context_model_comparison.csv", index=False)

    return classification, regression


def create_context_model_comparison_table(
    base_metrics_dir: str | Path,
    context_metrics_dir: str | Path,
    tables_dir: str | Path = "reports/tables",
) -> pd.DataFrame:
    """Compare base and context-enriched model metrics from separate runs."""
    base_metrics = Path(base_metrics_dir)
    context_metrics = Path(context_metrics_dir)
    tables = Path(tables_dir)
    tables.mkdir(parents=True, exist_ok=True)

    classification = _read_existing(
        [
            base_metrics / "classification_metrics.csv",
            base_metrics / "boosting_classification_metrics.csv",
            base_metrics / "advanced_classification_metrics.csv",
            base_metrics / "calibrated_classification_metrics.csv",
            context_metrics / "classification_metrics.csv",
            context_metrics / "boosting_classification_metrics.csv",
            context_metrics / "advanced_classification_metrics.csv",
            context_metrics / "calibrated_classification_metrics.csv",
        ]
    )
    regression = _read_existing(
        [
            base_metrics / "regression_metrics.csv",
            base_metrics / "boosting_regression_metrics.csv",
            base_metrics / "advanced_regression_metrics.csv",
            context_metrics / "regression_metrics.csv",
            context_metrics / "boosting_regression_metrics.csv",
            context_metrics / "advanced_regression_metrics.csv",
        ]
    )
    context_delta = _context_delta_table(classification, regression)
    if not context_delta.empty:
        context_delta.to_csv(tables / "context_model_comparison.csv", index=False)
    return context_delta
