"""Generate feature-family importance aggregations from existing SHAP/permutation tables.

Usage:
    python scripts/generate_feature_family_importance.py

Outputs:
    reports/tables/feature_family_importance_classification.csv
    reports/tables/feature_family_importance_regression.csv
    reports/figures/feature_family_importance_classification.png
    reports/figures/feature_family_importance_regression.png
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from aac_adoption.interpretation.feature_families import (
    FAMILY_LABELS,
    aggregate_importance_by_family,
)

TABLES_DIR = ROOT / "reports" / "tables"
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Colour palette — one colour per family, consistent across both tasks
FAMILY_COLORS = {
    "age": "#4e79a7",
    "sex_reproductive_status": "#f28e2b",
    "intake_circumstances": "#e15759",
    "intake_condition_health": "#76b7b2",
    "breed_appearance": "#59a14f",
    "color": "#edc948",
    "name_identity": "#b07aa1",
    "location": "#ff9da7",
    "timing_seasonality": "#9c755f",
    "covid_period": "#bab0ac",
    "animal_type": "#54a0ff",
    "external_context": "#2d3436",
    "other": "#636e72",
}


def _plot_family_importance(family_df: pd.DataFrame, task: str, output_path: Path) -> None:
    """Save a horizontal bar chart of family importance."""
    df = family_df.copy()
    df = df[df["sum_importance"] > 0].sort_values("sum_importance", ascending=True)
    labels = [FAMILY_LABELS.get(f, f.replace("_", " ").title()) for f in df["family"]]
    colors = [FAMILY_COLORS.get(f, FAMILY_COLORS["other"]) for f in df["family"]]

    fig, ax = plt.subplots(figsize=(9, max(3, len(df) * 0.55)))
    bars = ax.barh(labels, df["sum_importance"], color=colors, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
    task_label = task.replace("_", " ").title()
    ax.set_title(
        f"Feature Family Importance — {task_label}\n"
        "(sum of mean |SHAP| per family; model contribution, not causal effect)",
        fontsize=10,
        pad=12,
    )
    ax.set_xlabel("Sum of mean |SHAP| per family", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {output_path}")


def process_task(task: str) -> None:
    shap_path = TABLES_DIR / f"shap_global_{task}.csv"
    if not shap_path.exists():
        print(f"  [SKIP] {shap_path.name} not found — run generate_diagnostics.py --include-shap first.")
        return

    shap_df = pd.read_csv(shap_path)
    print(f"  Loaded {len(shap_df)} rows from {shap_path.name}")

    # Use feature_family column already in the file if present; otherwise derive it
    importance_col = "mean_abs_shap"
    family_df = aggregate_importance_by_family(shap_df, feature_col="feature", importance_col=importance_col)
    family_df["task"] = task

    csv_out = TABLES_DIR / f"feature_family_importance_{task}.csv"
    family_df.to_csv(csv_out, index=False)
    print(f"  Saved table: {csv_out}")

    fig_out = FIGURES_DIR / f"feature_family_importance_{task}.png"
    _plot_family_importance(family_df, task, fig_out)


def main() -> None:
    print("=== Feature Family Importance Generator ===")
    for task in ("classification", "regression"):
        print(f"\n[{task.upper()}]")
        process_task(task)
    print("\nDone.")


if __name__ == "__main__":
    main()
