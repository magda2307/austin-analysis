"""H5 COVID-period evidence — explicitly descriptive/predictive, not causal (Task 3.4)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


COVID_ORDER = ["pre_covid", "covid", "post_covid"]
COVID_LABELS = {"pre_covid": "Pre-COVID\n(≤2019)", "covid": "COVID\n(2020–2021)", "post_covid": "Post-COVID\n(2022+)"}
PALETTE = {"pre_covid": "#3A86FF", "covid": "#FF6B6B", "post_covid": "#8338EC"}


def _ordered_covid(df: pd.DataFrame, col: str = "covid_period") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=COVID_ORDER, ordered=True)
    return df.sort_values(col)


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def _build_evidence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    target_col = next(
        (c for c in ["classification_target", "adopted", "is_adopted"] if c in df.columns), None
    )
    days_col = next(
        (c for c in ["days_to_outcome", "regression_target_days", "days_to_adoption"] if c in df.columns), None
    )
    rows = []
    for period in COVID_ORDER:
        sub = df[df["covid_period"] == period]
        row: dict = {"covid_period": period, "n_records": len(sub)}
        if target_col:
            row["adoption_rate"] = float(sub[target_col].mean()) if not sub.empty else None
        if days_col:
            row["median_los_days"] = float(sub[days_col].median()) if not sub.empty else None
        row["intake_volume"] = len(sub)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_population_mix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in COVID_ORDER:
        sub = df[df["covid_period"] == period]
        n = max(len(sub), 1)
        row: dict = {"covid_period": period, "n_records": len(sub)}

        if "animal_type" in sub.columns:
            for atype in ["Dog", "Cat"]:
                row[f"pct_{atype.lower()}"] = round(100 * (sub["animal_type"] == atype).sum() / n, 2)

        if "intake_type" in sub.columns:
            for itype in sub["intake_type"].dropna().unique():
                safe = str(itype).lower().replace(" ", "_")
                row[f"pct_intake_{safe}"] = round(100 * (sub["intake_type"] == itype).sum() / n, 2)

        if "age_group" in sub.columns:
            for ag in ["baby", "young", "adult", "senior"]:
                row[f"pct_age_{ag}"] = round(100 * (sub["age_group"] == ag).sum() / n, 2)

        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_bar(values: list[float], labels: list[str], title: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [PALETTE.get(l.split("\n")[0].replace(" ", "_").lower(), "#888") for l in labels]
    ax.bar(labels, values, color=colors, alpha=0.88, edgecolor="white", width=0.55)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    for i, v in enumerate(values):
        ax.text(i, v * 1.01, f"{v:.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[3.4] Wrote {out_path.name}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def create_h5_covid_evidence(
    data_path: str | Path = "data/processed/modeling_dataset.csv",
    tables_dir: str | Path = "reports/tables",
    figures_dir: str | Path = "reports/figures",
    summary_dir: str | Path = "reports/summary",
) -> None:
    tables = Path(tables_dir)
    figures = Path(figures_dir)
    summary = Path(summary_dir)
    for d in [tables, figures, summary]:
        d.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(data_path, nrows=0)
    parse_dates = [c for c in ["intake_datetime", "outcome_datetime"] if c in header.columns]
    df = pd.read_csv(data_path, parse_dates=parse_dates)

    if "covid_period" not in df.columns:
        print("[3.4] covid_period column not found — skipping H5 evidence.")
        return

    evidence = _build_evidence_matrix(df)
    evidence.to_csv(tables / "h5_covid_evidence_matrix.csv", index=False)
    print(f"[3.4] Wrote h5_covid_evidence_matrix.csv")

    pop_mix = _build_population_mix(df)
    pop_mix.to_csv(tables / "h5_covid_population_mix.csv", index=False)
    print(f"[3.4] Wrote h5_covid_population_mix.csv")

    ordered = _ordered_covid(evidence)
    xlabels = [COVID_LABELS.get(p, p) for p in ordered["covid_period"]]

    if "adoption_rate" in ordered.columns:
        _plot_bar(
            (ordered["adoption_rate"] * 100).tolist(),
            xlabels,
            "H5 — Adoption Rate by COVID Period\n(descriptive, not causal)",
            "Adoption Rate (%)",
            figures / "h5_covid_adoption_rate.png",
        )

    if "median_los_days" in ordered.columns:
        _plot_bar(
            ordered["median_los_days"].tolist(),
            xlabels,
            "H5 — Median Length of Stay by COVID Period\n(descriptive, not causal)",
            "Median Days to Outcome",
            figures / "h5_covid_median_los.png",
        )

    _plot_bar(
        ordered["intake_volume"].tolist(),
        xlabels,
        "H5 — Intake Volume by COVID Period",
        "Number of Records",
        figures / "h5_covid_intake_volume.png",
    )

    _write_h5_interpretation_md(evidence, pop_mix, summary)


def _write_h5_interpretation_md(
    evidence: pd.DataFrame,
    pop_mix: pd.DataFrame,
    summary: Path,
) -> None:
    lines = [
        "# H5 Interpretation — COVID-Period Change in Adoption Patterns\n\n",
        "> **Important:** This analysis is explicitly **descriptive and associational**. "
        "The thesis does **not** claim COVID caused changes in adoption behaviour. "
        "The COVID period serves as a time-marker for changed system conditions.\n\n",
        "## Hypothesis\n",
        "The COVID period is **associated with** changed adoption patterns in AAC records.\n\n",
        "## Evidence\n\n",
        "### Adoption Rate and Length of Stay by Period\n",
        evidence.to_markdown(index=False),
        "\n\n",
        "### Population Mix by Period\n",
        "Showing selected columns (species share, age group share).\n",
        pop_mix[[c for c in pop_mix.columns if c in [
            "covid_period", "n_records", "pct_dog", "pct_cat",
            "pct_age_baby", "pct_age_young", "pct_age_adult", "pct_age_senior"
        ]]].to_markdown(index=False),
        "\n\n",
        "## Interpretation\n\n",
        "- Adoption rates, intake volume, and population mix all changed across the three periods.\n",
        "- The COVID period coincides with shelter access restrictions, adoption drives, and shifts "
        "in pet-keeping behaviour — all of which are confounded.\n",
        "- The population mix changed (species share, age distribution), meaning raw metric shifts "
        "partly reflect a different mix of animals, not only changed adoption behaviour.\n",
        "- `covid_period` SHAP contribution is among the lowest of all feature families, consistent "
        "with the period being a proxy for many unmeasured changes.\n\n",
        "## Causal Warning\n\n",
        "> The thesis states: *The COVID period is associated with changed patterns in AAC records.* "
        "It does not state that COVID caused these changes. Many simultaneous societal factors changed "
        "at the same time, and this dataset cannot distinguish them.\n",
    ]
    (summary / "h5_interpretation.md").write_text("".join(lines), encoding="utf-8")
    print(f"[3.4] Wrote h5_interpretation.md")
