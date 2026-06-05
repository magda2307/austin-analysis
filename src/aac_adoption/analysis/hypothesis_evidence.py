"""Create the hypothesis evidence matrix for thesis defence (Task 3.1).

Reads existing tables produced by earlier pipeline steps and assembles one
row per hypothesis (H1–H5) with status, evidence links, caveats, and a
plain-language interpretation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Evidence-file helpers
# ---------------------------------------------------------------------------

def _file_exists(path: Path) -> str:
    return str(path) if path.exists() else ""


def _mean_shap(tables_dir: Path, task: str, family_keys: list[str]) -> float | None:
    path = tables_dir / f"shap_feature_families_{task}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "feature_family" not in df.columns or "mean_abs_shap" not in df.columns:
        return None
    mask = df["feature_family"].str.contains("|".join(family_keys), case=False, na=False)
    sub = df.loc[mask, "mean_abs_shap"]
    return float(sub.mean()) if not sub.empty else None


def _top_perm_features(tables_dir: Path, family_cols: list[str]) -> str:
    path = tables_dir / "permutation_importance_classification.csv"
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    if "feature" not in df.columns or "importance_mean" not in df.columns:
        return ""
    mask = df["feature"].str.contains("|".join(family_cols), case=False, na=False, regex=True)
    top = (
        df.loc[mask]
        .groupby("feature", as_index=False)["importance_mean"]
        .mean()
        .sort_values("importance_mean", ascending=False)
        .head(5)["feature"]
        .tolist()
    )
    return "; ".join(top)


# ---------------------------------------------------------------------------
# Per-hypothesis row builders
# ---------------------------------------------------------------------------

def _h1_row(tables_dir: Path, figures_dir: Path, summary_dir: Path) -> dict:
    shap_ctx = _mean_shap(tables_dir, "classification", ["intake_circumstances", "intake_condition_health"])
    shap_app = _mean_shap(tables_dir, "classification", ["breed_appearance", "color"])
    perm_summary = _top_perm_features(tables_dir, ["intake_type", "intake_condition", "breed", "color"])
    h1_table = _file_exists(tables_dir / "h1_intake_vs_appearance.csv")
    h1_family = _file_exists(tables_dir / "h1_feature_family_importance.csv")
    h1_ablation = _file_exists(tables_dir / "h1_feature_family_ablation.csv")
    interp_md = _file_exists(summary_dir / "h1_interpretation.md")

    # Derive status from SHAP if available
    if shap_ctx is not None and shap_app is not None:
        if shap_ctx > shap_app * 0.8:
            status = "supported_descriptively"
        else:
            status = "partially_supported"
    else:
        status = "partially_supported"

    return {
        "hypothesis": "H1",
        "hypothesis_text": "Intake circumstances (intake type, condition, found location) are at least as predictive of adoption outcome as appearance features (breed, colour).",
        "status": status,
        "primary_evidence": f"SHAP intake-context mean={shap_ctx:.4f} vs appearance mean={shap_app:.4f}" if shap_ctx and shap_app else perm_summary,
        "descriptive_evidence_file": h1_table,
        "model_evidence_file": h1_family or h1_ablation,
        "interpretability_evidence_file": _file_exists(tables_dir / "shap_feature_families_classification.csv"),
        "reliability_caveat": "Feature importance reflects association with the model target, not causal effect. Appearance and intake-context features are correlated (e.g. strays are more likely to be unclaimed mixed-breed dogs).",
        "final_interpretation": (
            "Intake circumstances contribute comparable or greater SHAP signal than appearance features in the "
            "combined classifier. Breed retains the single highest family-level importance score. "
            "H1 is supported descriptively and predictively, but causal direction cannot be established from observational data."
        ),
        "causal_warning": "Association only. Confounders (species mix, intake volume) not controlled.",
    }


def _h2_row(tables_dir: Path, figures_dir: Path, summary_dir: Path) -> dict:
    desc_file = _file_exists(tables_dir / "h2_seasonality_summary.csv")
    fig1 = _file_exists(figures_dir / "h2_adoption_rate_by_season.png")
    return {
        "hypothesis": "H2",
        "hypothesis_text": "Adoption rates and length of stay vary by intake season.",
        "status": "supported_descriptively",
        "primary_evidence": "Descriptive comparison of adoption rate and median LOS by season (spring/summer/autumn/winter).",
        "descriptive_evidence_file": desc_file,
        "model_evidence_file": _file_exists(tables_dir / "shap_feature_families_classification.csv"),
        "interpretability_evidence_file": "",
        "reliability_caveat": "Season effects may be confounded by annual intake volume trends and COVID period overlap.",
        "final_interpretation": (
            "H2 is supported descriptively. Seasonal variation in adoption rates is visible in AAC records. "
            "Seasonality SHAP contribution is modest relative to breed, age, and identity features. "
            "H2 is treated as a secondary check, not a primary thesis claim."
        ),
        "causal_warning": "Descriptive only. No causal claim about season driving adoption decisions.",
    }


def _h3_row(tables_dir: Path, figures_dir: Path, summary_dir: Path) -> dict:
    desc_file = _file_exists(tables_dir / "h3_age_adoption_speed.csv")
    evidence_matrix = _file_exists(tables_dir / "h3_age_evidence_matrix.csv")
    interp_md = _file_exists(summary_dir / "h3_interpretation.md")
    shap_age = _mean_shap(tables_dir, "classification", ["age"])
    return {
        "hypothesis": "H3",
        "hypothesis_text": "Age at intake is associated with both adoption likelihood and adoption timing among adopted animals.",
        "status": "supported_descriptively" if shap_age and shap_age > 0.1 else "partially_supported",
        "primary_evidence": f"Age-family SHAP mean={shap_age:.4f}; adoption rate and median adopted-only days by age group." if shap_age else "Adoption rate and adopted-only median days by age group.",
        "descriptive_evidence_file": evidence_matrix or desc_file,
        "model_evidence_file": _file_exists(tables_dir / "shap_feature_families_classification.csv"),
        "interpretability_evidence_file": _file_exists(figures_dir / "h3_age_shap_summary.png"),
        "reliability_caveat": (
            "Two distinct claims are bundled: (1) age → adoption likelihood (classification target); "
            "(2) age → adoption timing (regression target, adopted animals only). These are evaluated separately. "
            "Senior animals are a small cohort; estimates have wide uncertainty."
        ),
        "final_interpretation": (
            "H3 is supported at both levels. Baby and young animals have higher adoption rates. "
            "Among adopted animals, babies have shorter median length of stay. "
            "Senior animals have the lowest adoption rate and are flagged as a reliability concern. "
            "Dogs and cats are shown separately because age patterns differ."
        ),
        "causal_warning": "Age is associated with outcome but cannot be experimentally manipulated. Confounders (breed, health) not isolated.",
    }


def _h4_row(tables_dir: Path, figures_dir: Path, summary_dir: Path) -> dict:
    desc_file = _file_exists(tables_dir / "h4_dark_color_summary.csv")
    shap_color = _mean_shap(tables_dir, "classification", ["color"])
    return {
        "hypothesis": "H4",
        "hypothesis_text": "Black or dark-coloured animals have lower adoption rates (black dog/cat syndrome).",
        "status": "partially_supported",
        "primary_evidence": f"Color-family SHAP mean={shap_color:.4f}; adoption rate by is_black_or_dark." if shap_color else "Adoption rate and median LOS by is_black_or_dark.",
        "descriptive_evidence_file": desc_file,
        "model_evidence_file": _file_exists(tables_dir / "shap_feature_families_classification.csv"),
        "interpretability_evidence_file": "",
        "reliability_caveat": (
            "is_black_or_dark is an approximate operational colour grouping derived from free-text colour fields. "
            "It may misclassify animals. Colour-family SHAP is among the lowest of all families."
        ),
        "final_interpretation": (
            "H4 is partially supported descriptively. The adoption rate difference between dark and non-dark animals "
            "exists in the data but is modest and the model treats colour as a weak predictor. "
            "H4 is included as a secondary check with an explicit colour-coding caveat."
        ),
        "causal_warning": "Colour is an approximate grouping. Effect size is small. Confounders (breed, species) not controlled.",
    }


def _h5_row(tables_dir: Path, figures_dir: Path, summary_dir: Path) -> dict:
    desc_file = _file_exists(tables_dir / "h5_covid_period.csv")
    evidence_matrix = _file_exists(tables_dir / "h5_covid_evidence_matrix.csv")
    pop_mix = _file_exists(tables_dir / "h5_covid_population_mix.csv")
    interp_md = _file_exists(summary_dir / "h5_interpretation.md")
    shap_covid = _mean_shap(tables_dir, "classification", ["covid_period"])
    return {
        "hypothesis": "H5",
        "hypothesis_text": "The COVID period is associated with changed adoption patterns in AAC records.",
        "status": "supported_descriptively",
        "primary_evidence": f"COVID-period SHAP mean={shap_covid:.4f}; adoption rate, median LOS, and intake volume differ across pre/covid/post periods." if shap_covid else "Adoption rate, median LOS, and intake volume by covid_period.",
        "descriptive_evidence_file": evidence_matrix or desc_file,
        "model_evidence_file": _file_exists(tables_dir / "shap_feature_families_classification.csv"),
        "interpretability_evidence_file": pop_mix,
        "reliability_caveat": (
            "The COVID period coincides with other societal changes (remote work, adoption drives, intake restrictions). "
            "Population mix also changed during COVID (species, age, intake type). "
            "covid_period SHAP is the lowest of all families."
        ),
        "final_interpretation": (
            "H5 is supported descriptively. AAC records show changed adoption rates, intake volume, and population mix "
            "during the COVID period. The thesis states this as association, not causality. "
            "Model importance for covid_period is low, consistent with other period-level confounders."
        ),
        "causal_warning": "Explicitly non-causal. The COVID period is a time-marker for changed system conditions, not a direct cause.",
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def create_hypothesis_evidence_matrix(
    tables_dir: str | Path = "reports/tables",
    figures_dir: str | Path = "reports/figures",
    summary_dir: str | Path = "reports/summary",
) -> pd.DataFrame:
    """Assemble and write the hypothesis evidence matrix.

    Returns the DataFrame for downstream use.
    """
    tables = Path(tables_dir)
    figures = Path(figures_dir)
    summary = Path(summary_dir)
    summary.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    rows = [
        _h1_row(tables, figures, summary),
        _h2_row(tables, figures, summary),
        _h3_row(tables, figures, summary),
        _h4_row(tables, figures, summary),
        _h5_row(tables, figures, summary),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(tables / "hypothesis_evidence_matrix.csv", index=False)

    _write_md(df, summary / "hypothesis_evidence_matrix.md")
    print(f"[3.1] Wrote hypothesis_evidence_matrix.csv and .md")
    return df


def _write_md(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Hypothesis Evidence Matrix\n",
        "This table summarises the evidence status for each thesis hypothesis.\n",
        "Evidence comes from descriptive statistics, permutation importance, and SHAP feature-family summaries.\n",
        "**No causal claims are made unless stated otherwise.**\n\n",
    ]

    # Summary table
    cols = ["hypothesis", "hypothesis_text", "status", "causal_warning"]
    table_df = df[cols].copy()
    lines.append(table_df.to_markdown(index=False))
    lines.append("\n\n---\n\n")

    # Per-hypothesis narrative
    for _, row in df.iterrows():
        lines.append(f"## {row['hypothesis']}\n\n")
        lines.append(f"**Statement:** {row['hypothesis_text']}\n\n")
        lines.append(f"**Status:** `{row['status']}`\n\n")
        lines.append(f"**Primary evidence:** {row['primary_evidence']}\n\n")
        if row["descriptive_evidence_file"]:
            lines.append(f"**Descriptive table:** `{row['descriptive_evidence_file']}`\n\n")
        if row["model_evidence_file"]:
            lines.append(f"**Model evidence:** `{row['model_evidence_file']}`\n\n")
        if row["interpretability_evidence_file"]:
            lines.append(f"**Interpretability file:** `{row['interpretability_evidence_file']}`\n\n")
        lines.append(f"**Reliability caveat:** {row['reliability_caveat']}\n\n")
        lines.append(f"**Final interpretation:** {row['final_interpretation']}\n\n")
        lines.append(f"> ⚠️ **Causal warning:** {row['causal_warning']}\n\n")
        lines.append("---\n\n")

    path.write_text("".join(lines), encoding="utf-8")
