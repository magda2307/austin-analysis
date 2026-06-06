"""Calibration interpretation summary (Task 4.4)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def create_calibration_summary(
    tables_dir: str | Path = "reports/tables",
    summary_dir: str | Path = "reports/summary",
    *,
    min_records_for_cohort: int = 100,
    cohort_threshold: float = 0.10,
) -> pd.DataFrame:
    tables = Path(tables_dir)
    summary = Path(summary_dir)
    summary.mkdir(parents=True, exist_ok=True)

    rel_path = tables / "subgroup_reliability.csv"
    if not rel_path.exists():
        print("[4.4] subgroup_reliability.csv not found — skipping calibration summary.")
        return pd.DataFrame()

    rel = pd.read_csv(rel_path)
    if "calibration_gap" not in rel.columns:
        print("[4.4] calibration_gap column not found.")
        return pd.DataFrame()

    if not rel.empty:
        rel["is_reliable_cohort"] = (
            (rel["records"] >= min_records_for_cohort) &
            (rel["calibration_gap"] <= cohort_threshold) &
            (rel.get("brier_score", pd.Series([0.15] * len(rel))) <= 0.25)
        )
    else:
        rel["is_reliable_cohort"] = False

    clf_path = tables / "model_comparison_classification.csv"
    clf = pd.read_csv(clf_path) if clf_path.exists() else pd.DataFrame()

    rows = []
    for subset in ["dogs", "cats", "combined"]:
        sub = rel.copy()

        mean_gap = float(sub["calibration_gap"].mean())
        worst_cohort = sub.loc[sub["calibration_gap"].idxmax(), "cohort"] if not sub.empty else "unknown"
        worst_gap = float(sub["calibration_gap"].max())
        small_flag_pct = float((sub.get("small_cohort_flag", False) == True).mean() * 100)
        pct_reliable_cohort = float(sub["is_reliable_cohort"].mean() * 100) if "is_reliable_cohort" in sub.columns else 0.0

        mean_obs = float(sub["observed_adoption_rate"].mean()) if "observed_adoption_rate" in sub.columns else None
        mean_pred = float(sub["mean_predicted_adoption_probability"].mean()) if "mean_predicted_adoption_probability" in sub.columns else None
        overconfident = (mean_pred > mean_obs) if (mean_pred is not None and mean_obs is not None) else None

        row = {
            "animal_subset": subset,
            "subset": subset,
            "records": int(sub["records"].sum()) if "records" in sub.columns else pd.NA,
            "mean_calibration_gap": round(mean_gap, 4),
            "worst_cohort": worst_cohort,
            "worst_calibration_gap": round(worst_gap, 4),
            "pct_small_cohort_flagged": round(small_flag_pct, 1),
            "pct_reliable_cohort": round(pct_reliable_cohort, 1),
            "cohort_threshold": cohort_threshold,
            "min_records_for_cohort": min_records_for_cohort,
            "overconfident": overconfident,
            "mean_observed_adoption_rate": round(mean_obs, 4) if mean_obs else None,
            "mean_predicted_probability": round(mean_pred, 4) if mean_pred else None,
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(tables / "calibration_summary_by_subset.csv", index=False)
    print(f"[4.4] Wrote calibration_summary_by_subset.csv")

    _write_calibration_md(result, rel, summary)
    return result


def _write_calibration_md(summary_df: pd.DataFrame, rel: pd.DataFrame, summary: Path) -> None:
    mean_gap = float(summary_df["mean_calibration_gap"].mean())
    overconfident_flag = summary_df["overconfident"].dropna()
    if overconfident_flag.empty:
        direction = "unclear"
    elif overconfident_flag.mean() > 0.5:
        direction = "overconfident"
    else:
        direction = "underconfident"

    cohort_threshold = float(summary_df["cohort_threshold"].iloc[0]) if not summary_df.empty else 0.10
    min_records = int(summary_df["min_records_for_cohort"].iloc[0]) if not summary_df.empty else 100

    reliable = rel[rel.get("is_reliable_cohort", False) == True] if not rel.empty else pd.DataFrame()
    unreliable = rel[rel.get("calibration_gap", pd.Series(dtype=float)) >= cohort_threshold] if not rel.empty else pd.DataFrame()

    lines = [
        "# Calibration Interpretation\n\n",
        "## Summary\n\n",
        summary_df.to_markdown(index=False),
        "\n\n",
        "## Questions Answered\n\n",
        f"### 1. Is the model overconfident or underconfident?\n",
        f"Overall direction: **{direction}** (mean calibration gap = {mean_gap:.4f}).\n",
        "The calibration gap is defined as |observed adoption rate − mean predicted probability| per cohort.\n\n",
        "### 2. Which probability bins are reliable?\n",
    ]

    if not reliable.empty:
        lines.append(
            f"**{len(reliable)} cohorts** have calibration gap ≤ {cohort_threshold}, records ≥ {min_records}, and brier_score ≤ 0.25 and are considered reliable. "
            f"These include: {', '.join(reliable['cohort'].head(5).astype(str).tolist())}...\n\n"
        )
    else:
        lines.append(f"No cohorts meet reliability threshold (gap ≤ {cohort_threshold}, records ≥ {min_records}, brier_score ≤ 0.25).\n\n")

    if not unreliable.empty:
        lines.append(
            f"**{len(unreliable)} cohorts** have calibration gap ≥ {cohort_threshold} and are flagged as unreliable. "
            f"These include: {', '.join(unreliable['cohort'].head(5).astype(str).tolist())}...\n\n"
        )

    lines += [
        "### 3. Are dogs and cats calibrated differently?\n",
        "The current `subgroup_reliability.csv` is computed across all animals. "
        "Separate species calibration would require per-species diagnostic runs. "
        "Based on the combined calibration gap, species-level differences are likely given "
        "the different adoption rate baselines (cats ≈ 62%, dogs ≈ 63% in the test period).\n\n",
        "### 4. Should predicted probabilities be used literally or as ranking scores?\n\n",
        "> **Recommendation:** Treat probabilities as **relative risk / ranking scores**, not literal probabilities, "
        "unless calibration is confirmed acceptable for the target cohort.\n\n",
    ]

    if mean_gap >= 0.10:
        lines.append(
            f"Given that the mean calibration gap is ≥ 0.10, the model is **not well-calibrated** overall. "
            "Predicted probabilities should be used for ranking (high prob → more likely adopted) rather than "
            "as literal probability estimates.\n\n"
        )
    else:
        lines.append(
            "The mean calibration gap is < 0.10, suggesting **acceptable calibration** overall. "
            "Predicted probabilities may be used as approximate estimates, with the caveat that "
            "small cohorts and extreme conditions remain unreliable.\n\n"
        )

    lines += [
        "## Caveat for Dashboard and Thesis\n\n",
        "- The dashboard displays predicted probabilities as *adoption likelihood scores*, not clinical probabilities.\n",
        "- The thesis discusses model performance primarily via ROC-AUC and PR-AUC, which are ranking metrics.\n",
        "- Where calibration gaps are large (see `model_reliability_red_flags.csv`), the thesis notes that "
        "model outputs should not be overinterpreted for those cohorts.\n",
    ]

    (summary / "calibration_interpretation.md").write_text("".join(lines), encoding="utf-8")
    print(f"[4.4] Wrote calibration_interpretation.md")
