"""Subgroup reliability red flags (Task 4.5).

Reads subgroup_reliability.csv and produces a compact table of where the model
struggles, with risk levels and plain-language interpretation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


_RISK_THRESHOLDS = {
    "high": 0.15,
    "medium": 0.08,
}


def _risk_level(row: pd.Series) -> str:
    gap = row.get("calibration_gap", 0.0)
    small = row.get("small_cohort_flag", False)
    if small or gap >= _RISK_THRESHOLDS["high"]:
        return "high"
    if gap >= _RISK_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def _interpretation(row: pd.Series) -> str:
    cohort = str(row.get("cohort", "?"))
    value = str(row.get("value", "?"))
    gap = row.get("calibration_gap", 0.0)
    small = row.get("small_cohort_flag", False)
    fpr = row.get("false_positive_rate", None)
    fnr = row.get("false_negative_rate", None)
    obs = row.get("observed_adoption_rate", None)
    pred = row.get("mean_predicted_adoption_probability", None)

    parts = []

    if small:
        n = int(row.get("records", 0))
        parts.append(f"Small cohort (n={n}); estimates unreliable.")

    if gap >= _RISK_THRESHOLDS["high"] and not small:
        direction = "overestimates" if (pred is not None and obs is not None and pred > obs) else "underestimates"
        parts.append(
            f"Model {direction} adoption likelihood for {cohort}={value} "
            f"(gap={gap:.2f}, observed={obs:.2f}, predicted={pred:.2f})."
        )
    elif gap >= _RISK_THRESHOLDS["medium"]:
        parts.append(f"Moderate calibration gap ({gap:.2f}) for {cohort}={value}.")

    if fpr is not None and fpr >= 0.15:
        parts.append(f"High false-positive rate ({fpr:.2f}).")
    if fnr is not None and fnr >= 0.20:
        parts.append(f"High false-negative rate ({fnr:.2f}).")

    if not parts:
        parts.append("Within acceptable calibration range.")

    return " ".join(parts)


def create_reliability_red_flags(
    tables_dir: str | Path = "reports/tables",
    summary_dir: str | Path = "reports/summary",
) -> pd.DataFrame:
    tables = Path(tables_dir)
    summary = Path(summary_dir)
    summary.mkdir(parents=True, exist_ok=True)

    rel_path = tables / "subgroup_reliability.csv"
    if not rel_path.exists():
        print("[4.5] subgroup_reliability.csv not found — skipping red flags.")
        return pd.DataFrame()

    rel = pd.read_csv(rel_path)

    # Parse cohort / value if the CSV uses a combined "cohort" column
    if "cohort" in rel.columns and "value" not in rel.columns:
        # cohort column already present; value may be separate or embedded
        pass
    # Rename columns if they follow the subgroup_reliability format
    # cohort, value, records, small_cohort_flag, observed_adoption_rate,
    # mean_predicted_adoption_probability, calibration_gap, mae, false_positive_rate, false_negative_rate

    out = rel.copy()
    out["risk_level"] = out.apply(_risk_level, axis=1)
    out["interpretation"] = out.apply(_interpretation, axis=1)

    # Rename mae to regression_mae for clarity
    if "mae" in out.columns:
        out = out.rename(columns={"mae": "regression_mae"})

    # Reorder columns to match spec
    ordered_cols = [
        "cohort", "value", "records",
        "observed_adoption_rate", "mean_predicted_adoption_probability",
        "calibration_gap", "false_positive_rate", "false_negative_rate",
        "regression_mae", "small_cohort_flag",
        "risk_level", "interpretation",
    ]
    # Handle cases where 'value' is not a separate column (combined format)
    if "value" not in out.columns:
        # subgroup_reliability uses 'cohort' and 'value' as separate columns per the CSV header
        # If not, fall back to available columns
        ordered_cols = [c for c in ordered_cols if c in out.columns]
    else:
        ordered_cols = [c for c in ordered_cols if c in out.columns]

    out = out[ordered_cols].sort_values("calibration_gap", ascending=False)

    out.to_csv(tables / "model_reliability_red_flags.csv", index=False)
    print(f"[4.5] Wrote model_reliability_red_flags.csv ({len(out)} rows)")

    _write_red_flags_md(out, summary)
    return out


def _write_red_flags_md(df: pd.DataFrame, summary: Path) -> None:
    high = df[df["risk_level"] == "high"]
    medium = df[df["risk_level"] == "medium"]
    low = df[df["risk_level"] == "low"]

    lines = [
        "# Model Reliability Red Flags\n\n",
        "Cohorts where the model should **not** be trusted strongly.\n\n",
        f"- **High-risk cohorts:** {len(high)} (calibration gap ≥ 0.15 or small cohort)\n",
        f"- **Medium-risk cohorts:** {len(medium)} (gap 0.08–0.15)\n",
        f"- **Low-risk cohorts:** {len(low)} (gap < 0.08)\n\n",
        "## High-Risk Cohorts\n\n",
    ]

    if not high.empty:
        cols = [c for c in ["cohort", "value", "records", "calibration_gap", "small_cohort_flag", "risk_level", "interpretation"] if c in high.columns]
        lines.append(high[cols].head(20).to_markdown(index=False))
        lines.append("\n\n")
    else:
        lines.append("*None identified.*\n\n")

    lines.append("## Medium-Risk Cohorts\n\n")
    if not medium.empty:
        cols = [c for c in ["cohort", "value", "records", "calibration_gap", "interpretation"] if c in medium.columns]
        lines.append(medium[cols].head(15).to_markdown(index=False))
        lines.append("\n\n")
    else:
        lines.append("*None identified.*\n\n")

    lines += [
        "## Thesis Implications\n\n",
        "- The thesis can discuss model limitations by cohort using this table.\n",
        "- Small-sample cohorts (small_cohort_flag=True) are listed for completeness but should not be over-interpreted.\n",
        "- For high-risk cohorts, the thesis states that model predictions are unreliable and should not drive operational decisions for those groups without additional validation.\n",
        "- Medium-risk cohorts are acknowledged with a caveat but are not disqualified.\n\n",
        "## Note on Scope\n\n",
        "This analysis uses the combined test set. Per-species red flags would require rerunning diagnostics "
        "with species filters applied before cohort slicing.\n",
    ]

    (summary / "model_reliability_red_flags.md").write_text("".join(lines), encoding="utf-8")
    print(f"[4.5] Wrote model_reliability_red_flags.md")
