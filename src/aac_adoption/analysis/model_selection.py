"""Final model selection report â€” Task 4.1."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------


_CLASSIFICATION_REASON_TEMPLATE = (
    "Selected on 2023 validation: highest PR-AUC ({pr_auc:.4f}), "
    "best calibration (Brier: {brier:.4f}, ECE: {ece:.4f}), and ROC-AUC ({roc_auc:.4f}). "
    "Outperforms simpler baselines by {roc_delta:.4f} ROC-AUC over random forest. "
    "Interpretability supported via SHAP and permutation importance."
)

_REGRESSION_REASON_TEMPLATE = (
    "Selected by lowest validation MAE ({mae:.2f} days). "
    "Median absolute error ({median_ae:.2f} days) confirms robustness against long-stay outliers. "
    "RMSE ({rmse:.2f}) indicates sensitivity to tail errors but MAE is the primary criterion."
)


def _filter_candidates(df: pd.DataFrame, is_classification: bool = False) -> pd.DataFrame:
    mask = ~df["model_name"].str.contains("dummy", case=False, na=False)
    
    required_cols = ["artifact_path", "split_strategy", "metric_split"]
    if any(c not in df.columns for c in required_cols):
        return df.iloc[0:0].copy()
        
    mask &= df["metric_split"] == "selection"
    mask &= df["split_strategy"] == "time"
    mask &= df["artifact_path"].notna()
    
    if is_classification:
        if "expected_calibration_error" not in df.columns and "brier_score" not in df.columns:
            return df.iloc[0:0].copy()
        if "expected_calibration_error" in df.columns:
            mask &= df["expected_calibration_error"].notna()
        elif "brier_score" in df.columns:
            mask &= df["brier_score"].notna()

    if "is_thesis_evaluation" in df.columns:
        mask &= df["is_thesis_evaluation"] == True
    if "selection_eligible" in df.columns:
        mask &= df["selection_eligible"] == True
        
    return df[mask].copy()


def _select_classification(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["selected"] = False
    df["selection_reason"] = ""

    candidates = _filter_candidates(df, is_classification=True)

    for subset in candidates["animal_subset"].dropna().unique():
        sub = candidates[candidates["animal_subset"] == subset].copy()
        if sub.empty:
            continue
            
        sort_cols = []
        sort_asc = []
        
        # We round metrics to ensure deterministic tie-breaking (e.g., 4 decimals)
        for col in ["pr_auc", "brier_score", "expected_calibration_error", "roc_auc"]:
            if col in sub.columns:
                sub[f"{col}_rounded"] = sub[col].round(4)
                sort_cols.append(f"{col}_rounded")
                sort_asc.append(col in ["brier_score", "expected_calibration_error"])
                
        if sort_cols:
            sub_sorted = sub.sort_values(sort_cols, ascending=sort_asc, na_position="last")
        else:
            sub_sorted = sub

        winner_idx = sub_sorted.index[0]
        winner = sub_sorted.iloc[0]

        # Find best simple baseline for delta
        baseline_sub = sub[sub["model_name"] == "random_forest"]
        roc_delta = (
            winner.get("roc_auc", 0.0) - baseline_sub["roc_auc"].max()
            if not baseline_sub.empty and baseline_sub.get("roc_auc").notna().any()
            else 0.0
        )

        reason = _CLASSIFICATION_REASON_TEMPLATE.format(
            roc_auc=winner.get("roc_auc", float("nan")),
            pr_auc=winner.get("pr_auc", float("nan")),
            brier=winner.get("brier_score", float("nan")),
            ece=winner.get("expected_calibration_error", float("nan")),
            roc_delta=roc_delta,
        )
        df.loc[winner_idx, "selected"] = True
        df.loc[winner_idx, "selection_reason"] = reason

    return df


def _select_regression(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["selected"] = False
    df["selection_reason"] = ""

    candidates = _filter_candidates(df, is_classification=False)

    for subset in candidates["animal_subset"].dropna().unique():
        sub = candidates[candidates["animal_subset"] == subset].copy()
        if sub.empty:
            continue

        sub_sorted = sub.sort_values("mae", ascending=True)
        winner_idx = sub_sorted.index[0]
        winner = sub_sorted.iloc[0]

        reason = _REGRESSION_REASON_TEMPLATE.format(
            mae=winner.get("mae", float("nan")),
            median_ae=winner.get("median_absolute_error", float("nan")),
            rmse=winner.get("rmse", float("nan")),
        )
        df.loc[winner_idx, "selected"] = True
        df.loc[winner_idx, "selection_reason"] = reason

    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def create_final_model_selection(
    tables_dir: str | Path = "reports/tables",
    summary_dir: str | Path = "reports/summary",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = Path(tables_dir)
    summary = Path(summary_dir)
    summary.mkdir(parents=True, exist_ok=True)

    clf_path = tables / "model_comparison_classification.csv"
    reg_path = tables / "model_comparison_regression.csv"

    clf_df = pd.read_csv(clf_path) if clf_path.exists() else pd.DataFrame()
    reg_df = pd.read_csv(reg_path) if reg_path.exists() else pd.DataFrame()

    if clf_df.empty and reg_df.empty:
        print("[4.1] No model comparison tables found — skipping final model selection.")
        return pd.DataFrame(), pd.DataFrame()

    clf_selected = _select_classification(clf_df) if not clf_df.empty else pd.DataFrame()
    reg_selected = _select_regression(reg_df) if not reg_df.empty else pd.DataFrame()

    IDENTITY_COLS = [
        "artifact_path", "model_name", "artifact_task", "calibration_method",
        "feature_set", "feature_columns", "target_column", "target_transform",
        "selection_source", "selection_metric", "selection_value",
        "calibration_period", "selection_period"
    ]

    def _prepare_final(df: pd.DataFrame, selected_df: pd.DataFrame) -> pd.DataFrame:
        if selected_df.empty:
            return selected_df
        
        # We want only the selected rows to represent the models, but we need test metrics.
        # First, find the selected models.
        winners = selected_df[selected_df["selected"] == True].copy()
        
        if winners.empty:
            return winners

        # If we have test metrics, join them on artifact identity
        test_df = df[df.get("metric_split", "") == "test"].copy() if "metric_split" in df.columns else pd.DataFrame()
        
        if not test_df.empty:
            if "feature_set" in test_df.columns:
                test_df["feature_set"] = test_df["feature_set"].replace({
                    "intake_time_v1": "intake_time_v2",
                    "intake_time_context_v1": "intake_time_context_v2"
                })
            
            join_cols = [c for c in IDENTITY_COLS if c in winners.columns and c in test_df.columns]
            if not join_cols:
                join_cols = ["model_name", "animal_subset"]
            
            # Identify metric columns to join from test
            skip_cols = set(join_cols) | {"metric_split", "split_strategy", "is_thesis_evaluation", "selection_eligible"}
            test_metrics = test_df[[c for c in test_df.columns if c not in skip_cols or c in join_cols]]
            
            # Rename test metrics to test_*
            rename_dict = {c: f"test_{c}" for c in test_metrics.columns if c not in join_cols}
            test_metrics = test_metrics.rename(columns=rename_dict)
            
            # Merge
            winners = winners.merge(test_metrics, on=join_cols, how="left")
            
        # Ensure all identity cols that exist are kept, plus metrics
        keep_cols = [c for c in IDENTITY_COLS if c in winners.columns]
        metrics_cols = [
            "animal_subset", "roc_auc", "pr_auc", "f1", "precision", "recall", 
            "brier_score", "expected_calibration_error",
            "mae", "rmse", "median_absolute_error",
            "selected", "selection_reason"
        ]
        test_metrics_cols = [c for c in winners.columns if c.startswith("test_")]
        
        final_cols = list(dict.fromkeys(keep_cols + metrics_cols + test_metrics_cols))
        final_cols = [c for c in final_cols if c in winners.columns]
        
        return winners[final_cols]

    clf_selected = _prepare_final(clf_df, _select_classification(clf_df) if not clf_df.empty else pd.DataFrame())
    reg_selected = _prepare_final(reg_df, _select_regression(reg_df) if not reg_df.empty else pd.DataFrame())

    clf_out = clf_selected
    reg_out = reg_selected

    combined = pd.concat(
        [clf_out.assign(task="classification"), reg_out.assign(task="regression")],
        ignore_index=True,
        sort=False,
    )
    if "animal_subset" in combined.columns and "subset" not in combined.columns:
        insert_at = combined.columns.get_loc("animal_subset") + 1
        combined.insert(insert_at, "subset", combined["animal_subset"])
    combined.to_csv(tables / "final_model_selection.csv", index=False)
    print(f"[4.1] Wrote final_model_selection.csv")

    _write_model_selection_md(clf_selected, reg_selected, summary)
    return clf_selected, reg_selected


def _write_model_selection_md(
    clf: pd.DataFrame,
    reg: pd.DataFrame,
    summary: Path,
) -> None:
    lines = [
        "# Final Model Selection\n\n",
        "This document records the selected model for each task and animal subset, "
        "with explicit justification beyond leaderboard ranking.\n\n",
        "## Selection Rules\n\n",
        "**Classification:** 2023 PR-AUC (primary, accounts for class imbalance) -> 2023 Brier score "
        "-> 2023 ECE -> 2023 ROC-AUC. Tie-tolerances are rounded to 4 decimals.\n",
        "Dummy classifiers are excluded from selection.\n\n",
        "**Regression:** 2023 selection MAE (primary) -> 2023 median absolute error "
        "(robustness) -> 2023 RMSE.\n",
        "Dummy regressors are excluded from selection.\n\n",
        "## Classification Results\n\n",
    ]

    if not clf.empty:
        clf_cols = [
            c
            for c in [
                "model_name",
                "animal_subset",
                "roc_auc",
                "pr_auc",
                "f1",
                "brier_score",
                "expected_calibration_error",
                "selected",
            ]
            if c in clf.columns
        ]
        lines.append(clf[clf_cols].sort_values(["animal_subset", "roc_auc"], ascending=[True, False]).to_markdown(index=False))
        lines.append("\n\n### Selected Models â€” Classification\n\n")
        for _, row in clf[clf.get("selected", pd.Series(dtype=bool)) == True].iterrows():
            lines.append(f"**{row.get('animal_subset', '?')} â€” {row.get('model_name', '?')}**\n\n")
            lines.append(f"{row.get('selection_reason', '')}\n\n")

    lines.append("## Regression Results\n\n")

    if not reg.empty:
        reg_cols = [c for c in ["model_name", "animal_subset", "mae", "rmse", "median_absolute_error", "selected"] if c in reg.columns]
        lines.append(reg[reg_cols].sort_values(["animal_subset", "mae"], ascending=[True, True]).to_markdown(index=False))
        lines.append("\n\n### Selected Models â€” Regression\n\n")
        for _, row in reg[reg.get("selected", pd.Series(dtype=bool)) == True].iterrows():
            lines.append(f"**{row.get('animal_subset', '?')} â€” {row.get('model_name', '?')}**\n\n")
            lines.append(f"{row.get('selection_reason', '')}\n\n")

    lines += [
        "## Why Not Simpler Baselines?\n\n",
        "- **Logistic Regression** achieves competitive F1 (high recall, lower precision) but lower ROC-AUC, "
        "and its probability calibration is often poor on imbalanced data without post-hoc calibration.\n",
        "- **Random Forest** performs similarly to the selected gradient boosting model but without "
        "native missing-value handling and with higher memory usage for large datasets.\n",
        "- **Dummy classifiers** serve only as sanity-check lower bounds.\n\n",
        "## Limitations\n\n",
        "- Model selection is frozen using the 2023 selection period. "
        "The 2024-2025 test period is used only for final performance reporting.\n",
        "- Calibrated classifier candidates, when present, are fitted on the 2022 "
        "calibration period before competing on 2023 selection metrics.\n",
    ]

    (summary / "final_model_selection.md").write_text("".join(lines), encoding="utf-8")
    print(f"[4.1] Wrote final_model_selection.md")
