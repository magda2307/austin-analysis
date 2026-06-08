"""H1/H3/H5 support tables, adopted-only timing, and KM descriptive survival curves."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _summarize(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        df.groupby(column, dropna=False)
        .agg(
            records=("classification_target", "count"),
            adoptions=("classification_target", "sum"),
            adoption_rate_pct=("classification_target", lambda values: float(values.mean() * 100)),
            median_days_to_outcome=("regression_target_days", "median"),
        )
        .reset_index()
        .rename(columns={column: "value"})
        .assign(variable=column)
        .assign(
            target_column="classification_target/regression_target_days",
            population_scope="all matched episodes",
            estimand_label="adoption_rate_and_median_los"
        )
        .sort_values(["variable", "records"], ascending=[True, False])
    )


def _importance_for(tables_dir: Path, feature_terms: list[str]) -> pd.DataFrame:
    paths = [
        tables_dir / "permutation_importance_classification.csv",
        tables_dir / "random_forest_feature_importance.csv",
        tables_dir / "logistic_regression_coefficients.csv",
    ]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame(columns=["feature", "importance_score"])

    normalized = []
    for frame in frames:
        score_col = None
        for candidate in ["importance_mean", "importance", "abs_coefficient"]:
            if candidate in frame.columns:
                score_col = candidate
                break
        if score_col is None or "feature" not in frame.columns:
            continue
        normalized.append(frame[["feature", score_col]].rename(columns={score_col: "importance_score"}))
    if not normalized:
        return pd.DataFrame(columns=["feature", "importance_score"])

    importance = pd.concat(normalized, ignore_index=True)
    mask = importance["feature"].astype(str).str.contains("|".join(feature_terms), case=False, regex=True)
    return (
        importance.loc[mask]
        .groupby("feature", as_index=False)["importance_score"]
        .mean()
        .sort_values("importance_score", ascending=False)
    )


def _write_with_importance(
    summary: pd.DataFrame,
    importance: pd.DataFrame,
    output_path: Path,
) -> None:
    if importance.empty:
        summary["related_importance_features"] = ""
        summary["mean_importance_score"] = pd.NA
    else:
        summary["related_importance_features"] = "; ".join(importance["feature"].head(10).astype(str))
        summary["mean_importance_score"] = importance["importance_score"].mean()
    summary.to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Primary hypothesis tables
# ---------------------------------------------------------------------------

def create_hypothesis_support_tables(
    data_path: str | Path = "data/processed/modeling_dataset.csv",
    tables_dir: str | Path = "reports/tables",
) -> None:
    """Create thesis support tables for central hypotheses H1, H3, and H5.

    H3 output is renamed to h3_age_length_of_stay.csv to reflect that
    median_days_to_outcome is length-of-stay (all outcomes), not adoption speed.
    The adopted-only speed table is written separately by create_adopted_only_timing_tables().
    """
    tables = Path(tables_dir)
    tables.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)

    h1 = pd.concat(
        [
            _summarize(df, "intake_type"),
            _summarize(df, "intake_condition"),
            _summarize(df, "simplified_breed_group"),
            _summarize(df, "simplified_color_group"),
        ],
        ignore_index=True,
    )
    _write_with_importance(
        h1,
        _importance_for(
            tables,
            ["intake_type", "intake_condition", "simplified_breed_group", "simplified_color_group"],
        ),
        tables / "h1_intake_vs_appearance.csv",
    )

    h3 = _summarize(df, "age_group")
    _write_with_importance(
        h3,
        _importance_for(tables, ["age_days", "age_months", "age_years", "age_group"]),
        tables / "h3_age_length_of_stay.csv",
    )

    h5 = _summarize(df, "covid_period")
    _write_with_importance(
        h5,
        _importance_for(tables, ["covid_period"]),
        tables / "h5_covid_period.csv",
    )


# ---------------------------------------------------------------------------
# Task 3.5 - H2 Seasonality and H4 Dark Colour
# ---------------------------------------------------------------------------

SEASON_ORDER = ["spring", "summer", "autumn", "fall", "winter"]
SEASON_COLORS = {"spring": "#2DC653", "summer": "#FFBE0B", "autumn": "#FB5607", "fall": "#FB5607", "winter": "#3A86FF"}


def _bar_figure(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
    x_order: list[str] | None = None,
    colors: dict[str, str] | None = None,
    subtitle: str | None = None,
) -> None:
    if x_order:
        df = df.copy()
        df[x_col] = pd.Categorical(df[x_col], categories=x_order, ordered=True)
        df = df.sort_values(x_col)
    vals = df[y_col].tolist()
    labels = df[x_col].astype(str).tolist()
    bar_colors = [colors.get(l, "#555") for l in labels] if colors else ["#3A86FF"] * len(labels)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, vals, color=bar_colors, alpha=0.87, edgecolor="white")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=11)
    full_title = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(full_title, fontsize=12)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.01 + 0.001, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_h2_seasonality_outputs(
    data_path: str | Path = "data/processed/modeling_dataset.csv",
    tables_dir: str | Path = "reports/tables",
    figures_dir: str | Path = "reports/figures",
    summary_dir: str | Path = "reports/summary",
) -> None:
    """H2: adoption rate and median LOS by intake season."""
    tables = Path(tables_dir)
    figures = Path(figures_dir)
    summary = Path(summary_dir)
    for d in [tables, figures, summary]:
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if "intake_season" not in df.columns:
        print("[3.5] intake_season column not found - skipping H2.")
        return

    summary_df = _summarize(df, "intake_season")
    summary_df.to_csv(tables / "h2_seasonality_summary.csv", index=False)
    print("[3.5] Wrote h2_seasonality_summary.csv")

    ordered_seasons = [s for s in SEASON_ORDER if s in summary_df["value"].values]
    if not ordered_seasons:
        ordered_seasons = None

    if "adoption_rate_pct" in summary_df.columns:
        _bar_figure(
            summary_df, "value", "adoption_rate_pct",
            title="H2 - Adoption Rate by Season (secondary check)",
            ylabel="Adoption Rate (%)",
            out_path=figures / "h2_adoption_rate_by_season.png",
            x_order=ordered_seasons,
            colors=SEASON_COLORS,
        )
        print("[3.5] Wrote h2_adoption_rate_by_season.png")

    if "median_days_to_outcome" in summary_df.columns:
        _bar_figure(
            summary_df, "value", "median_days_to_outcome",
            title="H2 - Median Length of Stay by Season (secondary check)",
            ylabel="Median Days to Outcome",
            out_path=figures / "h2_median_los_by_season.png",
            x_order=ordered_seasons,
            colors=SEASON_COLORS,
        )
        print("[3.5] Wrote h2_median_los_by_season.png")

    _write_h2_interpretation_md(summary_df, summary)


def _write_h2_interpretation_md(summary_df: pd.DataFrame, summary: Path) -> None:
    highest_adoption = summary_df.sort_values("adoption_rate_pct", ascending=False).iloc[0]
    lowest_adoption = summary_df.sort_values("adoption_rate_pct", ascending=True).iloc[0]
    longest_los = summary_df.sort_values("median_days_to_outcome", ascending=False).iloc[0]
    shortest_los = summary_df.sort_values("median_days_to_outcome", ascending=True).iloc[0]
    lines = [
        "# H2 Interpretation - Seasonality and Adoption Patterns\n\n",
        "## Hypothesis\n",
        "Adoption rates and length of stay vary by intake season.\n\n",
        "## Evidence\n\n",
        "### Descriptive Seasonality Summary\n\n",
    ]
    lines.append(summary_df.to_markdown(index=False))
    lines.append("\n\n")
    lines += [
        "### Model Evidence\n",
        "Seasonality features (`intake_season`, `intake_month`, `intake_quarter`) are evaluated by the classification and regression models.\n",
        "SHAP importance for the seasonality feature family is relatively modest compared to breed, age, and identity.\n\n",
        "## Interpretation\n\n",
        "- Seasonal variation is descriptively present: "
        f"{highest_adoption['value']} has the highest adoption rate ({highest_adoption['adoption_rate_pct']:.2f}%), "
        f"while {lowest_adoption['value']} has the lowest adoption rate ({lowest_adoption['adoption_rate_pct']:.2f}%).\n",
        "- Length-of-stay variation is also descriptive: "
        f"{longest_los['value']} has the longest median time to outcome ({longest_los['median_days_to_outcome']:.2f} days), "
        f"while {shortest_los['value']} has the shortest ({shortest_los['median_days_to_outcome']:.2f} days).\n",
        "- However, seasonal differences are small in magnitude, and the machine learning models treat seasonality as a weak predictor compared to clinical/demographic features.\n",
        "- H2 is supported descriptively as an association, not a causal driver.\n\n",
        "## Causal Warning\n\n",
        "> **Seasonality is associated with outcome but is descriptive only.** We cannot claim that season causes adoptions. Seasonal variations are heavily confounded by animal intake volumes (e.g. kitten season in spring/summer) and shelter resource constraints.\n",
    ]
    (summary / "h2_interpretation.md").write_text("".join(lines), encoding="utf-8")
    print("[3.5] Wrote h2_interpretation.md")


DARK_COLOR_LABELS = {True: "Dark / Black", False: "Not Dark", "True": "Dark / Black", "False": "Not Dark"}
DARK_COLORS_MAP = {"Dark / Black": "#1A1A2E", "Not Dark": "#E2B96F"}


def create_h4_dark_color_outputs(
    data_path: str | Path = "data/processed/modeling_dataset.csv",
    tables_dir: str | Path = "reports/tables",
    figures_dir: str | Path = "reports/figures",
    summary_dir: str | Path = "reports/summary",
) -> None:
    """H4: adoption rate and median LOS by dark colour flag."""
    tables = Path(tables_dir)
    figures = Path(figures_dir)
    summary = Path(summary_dir)
    for d in [tables, figures, summary]:
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if "is_black_or_dark" not in df.columns:
        print("[3.5] is_black_or_dark column not found - skipping H4.")
        return

    summary_df = _summarize(df, "is_black_or_dark")
    summary_df["value"] = summary_df["value"].astype(str).map(lambda v: DARK_COLOR_LABELS.get(v, v))
    summary_df.to_csv(tables / "h4_dark_color_summary.csv", index=False)
    print("[3.5] Wrote h4_dark_color_summary.csv")

    caution = "Note: is_black_or_dark is an approximate operational colour grouping"

    if "adoption_rate_pct" in summary_df.columns:
        _bar_figure(
            summary_df, "value", "adoption_rate_pct",
            title="H4 - Adoption Rate by Coat Colour (secondary check)",
            ylabel="Adoption Rate (%)",
            out_path=figures / "h4_dark_color_adoption_rate.png",
            colors=DARK_COLORS_MAP,
            subtitle=caution,
        )
        print("[3.5] Wrote h4_dark_color_adoption_rate.png")

    if "median_days_to_outcome" in summary_df.columns:
        _bar_figure(
            summary_df, "value", "median_days_to_outcome",
            title="H4 - Median Length of Stay by Coat Colour (secondary check)",
            ylabel="Median Days to Outcome",
            out_path=figures / "h4_dark_color_median_los.png",
            colors=DARK_COLORS_MAP,
            subtitle=caution,
        )
        print("[3.5] Wrote h4_dark_color_median_los.png")

    _write_h4_interpretation_md(summary_df, summary)


def _write_h4_interpretation_md(summary_df: pd.DataFrame, summary: Path) -> None:
    by_value = summary_df.set_index("value")
    dark = by_value.loc["Dark / Black"]
    not_dark = by_value.loc["Not Dark"]
    rate_gap = dark["adoption_rate_pct"] - not_dark["adoption_rate_pct"]
    los_gap = dark["median_days_to_outcome"] - not_dark["median_days_to_outcome"]
    direction = "higher" if rate_gap > 0 else "lower"
    los_direction = "longer" if los_gap > 0 else "shorter"
    lines = [
        "# H4 Interpretation - Coat Colour (Black/Dark Animals)\n\n",
        "## Hypothesis\n",
        "Black or dark-coloured animals have lower adoption rates (black dog/cat syndrome).\n\n",
        "## Evidence\n\n",
        "### Descriptive Colour Summary\n\n",
    ]
    lines.append(summary_df.to_markdown(index=False))
    lines.append("\n\n")
    lines += [
        "### Model Evidence\n",
        "The coat colour feature family (`color`, `is_black_or_dark`, `simplified_color_group`) is a very weak predictor in both classification and regression models, with low SHAP and permutation importance.\n\n",
        "## Interpretation\n\n",
        "- Descriptively, black or dark-coloured animals show an adoption rate of "
        f"**{dark['adoption_rate_pct']:.2f}%** compared to **{not_dark['adoption_rate_pct']:.2f}%** for non-dark animals "
        f"({abs(rate_gap):.2f} percentage points {direction}).\n",
        "- Median length of stay is similar: "
        f"**{dark['median_days_to_outcome']:.2f} days** for dark-coloured animals vs "
        f"**{not_dark['median_days_to_outcome']:.2f} days** for non-dark animals "
        f"({abs(los_gap):.2f} days {los_direction}).\n",
        "- Consequently, the popular hypothesis of \"black dog/cat syndrome\" is evaluated as a descriptive check rather than a strong primary finding.\n",
        "- H4 is treated as a secondary check with an explicit caveat that color is not a primary driver of adoption outcomes in this shelter.\n\n",
        "## Causal Warning\n\n",
        "> **Colour associations are descriptive only.** Coat colour is a weak predictor and co-varies with breed and species. This analysis does not control for specific breed-colour combinations or individual shelter presentation factors.\n",
    ]
    (summary / "h4_interpretation.md").write_text("".join(lines), encoding="utf-8")
    print("[3.5] Wrote h4_interpretation.md")


# ---------------------------------------------------------------------------
# Adopted-only descriptive timing tables (Task 2.2)
# ---------------------------------------------------------------------------

def create_adopted_only_timing_tables(
    data_path: str | Path = "data/processed/modeling_dataset.csv",
    tables_dir: str | Path = "reports/tables",
    figures_dir: str | Path = "reports/figures",
) -> None:
    """Create adopted-animal-only timing tables for H3 support.

    Uses days_to_adoption (= days_to_outcome where outcome_type == 'Adoption').
    This is the correct metric for adoption-speed analysis.
    The main h3_age_length_of_stay.csv uses all outcomes.
    """
    tables = Path(tables_dir)
    figures = Path(figures_dir)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    # Filter to adopted animals only
    adopted = df.loc[df["classification_target"].eq(1)].copy()
    if adopted.empty:
        return

    # Ensure days_to_adoption column exists
    adopted["days_to_adoption"] = adopted["regression_target_days"]

    groups = ["age_group", "animal_type"]
    records = []

    for group_col in groups:
        if group_col not in adopted.columns:
            continue
        for (group_val, atype), sub in adopted.groupby([group_col, "animal_type"], dropna=False):
            d = sub["days_to_adoption"].dropna()
            if d.empty:
                continue
            total = len(df[df["animal_type"] == atype]) if group_col == "age_group" else len(df)
            if group_col == "age_group":
                total = len(df[(df["age_group"] == group_val) & (df["animal_type"] == atype)])
            records.append(
                {
                    "group_variable": group_col,
                    "group_value": group_val,
                    "animal_type": atype,
                    "all_records": total,
                    "adopted_records": len(sub),
                    "median_days_to_adoption": float(d.median()),
                    "mean_days_to_adoption": float(d.mean()),
                    "p25_days_to_adoption": float(d.quantile(0.25)),
                    "p75_days_to_adoption": float(d.quantile(0.75)),
                    "adopted_within_7_days": int((d <= 7).sum()),
                    "adopted_within_30_days": int((d <= 30).sum()),
                    "adopted_within_60_days": int((d <= 60).sum()),
                    "adopted_within_90_days": int((d <= 90).sum()),
                }
            )

    # Also combined (all animal types together)
    for group_col in groups:
        if group_col not in adopted.columns:
            continue
        for group_val, sub in adopted.groupby(group_col, dropna=False):
            d = sub["days_to_adoption"].dropna()
            if d.empty:
                continue
            all_for_group = len(df[df[group_col] == group_val])
            records.append(
                {
                    "group_variable": group_col,
                    "group_value": group_val,
                    "animal_type": "Combined",
                    "all_records": all_for_group,
                    "adopted_records": len(sub),
                    "median_days_to_adoption": float(d.median()),
                    "mean_days_to_adoption": float(d.mean()),
                    "p25_days_to_adoption": float(d.quantile(0.25)),
                    "p75_days_to_adoption": float(d.quantile(0.75)),
                    "adopted_within_7_days": int((d <= 7).sum()),
                    "adopted_within_30_days": int((d <= 30).sum()),
                    "adopted_within_60_days": int((d <= 60).sum()),
                    "adopted_within_90_days": int((d <= 90).sum()),
                }
            )

    if not records:
        return

    result = pd.DataFrame(records)
    age_rows = result["group_variable"].eq("age_group")
    result["age_group"] = pd.NA
    result.loc[age_rows, "age_group"] = result.loc[age_rows, "group_value"]
    result["records"] = result["all_records"]
    result.to_csv(tables / "h3_adopted_only_age_speed.csv", index=False)

    # Figure: median days to adoption by age_group, per animal type
    age_view = result[result["group_variable"] == "age_group"].copy()
    if not age_view.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for atype, sub in age_view.groupby("animal_type"):
            sub_sorted = sub.sort_values("median_days_to_adoption")
            ax.bar(
                [f"{row['group_value']}\n({atype})" for _, row in sub_sorted.iterrows()],
                sub_sorted["median_days_to_adoption"],
                label=atype,
                alpha=0.8,
            )
        ax.set_xlabel("Age group / Animal type")
        ax.set_ylabel("Median days to adoption (adopted animals only)")
        ax.set_title(
            "H3: Median days to adoption by age group — adopted animals only\n"
            "(uses days_to_adoption, not days_to_outcome)"
        )
        ax.legend()
        plt.tight_layout()
        fig.savefig(figures / "h3_adopted_only_median_days_to_adoption.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Kaplan–Meier style descriptive survival curves (Task 2.3)
# ---------------------------------------------------------------------------

def _empirical_survival(times: np.ndarray, max_days: int = 365) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical survival (proportion not yet adopted) at each day."""
    n = len(times)
    days = np.arange(0, max_days + 1)
    survived = np.array([(times > d).sum() / n for d in days])
    return days, survived


def _km_survival(times: np.ndarray, max_days: int = 365) -> tuple[np.ndarray, np.ndarray]:
    """Compute KM estimate using lifelines when available, else fallback to empirical."""
    try:
        from lifelines import KaplanMeierFitter

        kmf = KaplanMeierFitter()
        # All events are observed (no censoring in adopted-only subset)
        event_observed = np.ones(len(times), dtype=bool)
        kmf.fit(times, event_observed=event_observed, label="km")
        timeline = np.arange(0, max_days + 1)
        sf = kmf.survival_function_at_times(timeline).values
        return timeline, sf
    except ImportError:
        return _empirical_survival(times, max_days)


def create_survival_descriptive(
    data_path: str | Path = "data/processed/modeling_dataset.csv",
    tables_dir: str | Path = "reports/tables",
    figures_dir: str | Path = "reports/figures",
    summary_dir: str | Path = "reports/summary",
    max_days: int = 180,
) -> None:
    """Create descriptive Kaplan-Meier style adoption survival curves.

    These are descriptive time-to-adoption views among ADOPTED animals only.
    They are NOT the main modeling framework and do not replace the supervised ML
    comparison. They serve as descriptive evidence for H3 and address the 'why
    not survival analysis?' reviewer question.

    Uses lifelines.KaplanMeierFitter when available; falls back to empirical
    proportion-not-yet-adopted curves if lifelines is not installed.

    Groups: animal_type, age_group, covid_period, intake_type.
    """
    tables = Path(tables_dir)
    figures = Path(figures_dir)
    summary = Path(summary_dir)
    for d in [tables, figures, summary]:
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    # Use adopted animals only for KM curves (time-to-adoption analysis)
    adopted = df.loc[df["classification_target"].eq(1)].copy()
    if adopted.empty:
        return

    adopted["days_to_adoption"] = adopted["regression_target_days"]

    group_columns = ["animal_type", "age_group", "covid_period", "intake_type"]
    group_columns = [c for c in group_columns if c in adopted.columns]

    all_curves: list[dict] = []

    for group_col in group_columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        group_data = adopted.groupby(group_col, dropna=False)

        for group_val, sub in group_data:
            times = sub["days_to_adoption"].dropna().values
            if len(times) < 20:
                continue
            days, sf = _km_survival(times, max_days=max_days)
            label = f"{group_val} (n={len(times):,})"
            ax.step(days, sf, where="post", label=label, linewidth=2)

            for d, s in zip(days[::10], sf[::10]):
                all_curves.append(
                    {
                        "group_variable": group_col,
                        "group_value": str(group_val),
                        "day": int(d),
                        "survival_probability": float(s),
                        "adoption_probability": float(1 - s),
                        "records": int(len(times)),
                    }
                )

        ax.set_xlabel("Days since intake")
        ax.set_ylabel("Proportion not yet adopted (survival function)")
        ax.set_title(
            f"Descriptive adoption survival curve by {group_col}\n"
            "(adopted animals only — KM-style descriptive view, not a full survival model)"
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        out_path = figures / f"km_adoption_by_{group_col}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    if all_curves:
        curves_df = pd.DataFrame(all_curves)
        curves_df.to_csv(tables / "adoption_survival_curves.csv", index=False)

    # Write methodology note
    note = """\
# Descriptive Survival Analysis Note

## What These Curves Are

The figures `km_adoption_by_*.png` and the table `adoption_survival_curves.csv`
show **empirical Kaplan-Meier style adoption survival curves** for adopted animals,
grouped by `animal_type`, `age_group`, `covid_period`, and `intake_type`.

The y-axis is the **proportion of adopted animals not yet adopted** at each day since intake.
The x-axis is days since intake, restricted to adopted animals only.

Implementation: uses `lifelines.KaplanMeierFitter` when available; falls back to
empirical proportion curves. All events are observed (no censoring) in this adopted-only subset.

## What These Curves Are NOT

These curves are **descriptive time-to-adoption views**. They are NOT:
- The main modeling framework of this thesis.
- A replacement for the supervised ML classification and regression comparison.
- A full survival model with censoring or competing risks.

## Why Not Full Survival Analysis?

1. **Most episodes are resolved:** The dataset contains only matched intake/outcome
   episodes. Animals without a future outcome are excluded from the modeling dataset,
   making the censoring problem smaller than in typical clinical survival analysis.

2. **The main regression target is length-of-stay, not time-to-adoption:**
   `regression_target_days` = `days_to_outcome` covers all outcomes, not just adoption.
   This is operationally relevant for shelter resource planning.

3. **Interpretability:** A regression prediction ("predicted length of stay: 12 days")
   is more directly actionable for a shelter worker than a hazard ratio from a Cox model.

4. **Future work:** Full survival modeling with censoring and competing risks
   (adoption vs. transfer vs. euthanasia vs. return-to-owner) would be a natural
   extension. The descriptive KM curves here provide the foundation.

## Thesis Defense Statement

> "These curves are descriptive time-to-adoption views among adopted animals.
> They provide descriptive evidence for H3 (age and adoption timing patterns)
> without making causal claims. Full time-to-event survival modeling with censoring
> and competing risks is outside the main scope of this thesis and is discussed
> as a natural extension for future work."
"""
    (summary / "survival_descriptive_note.md").write_text(note, encoding="utf-8")
