"""Confusion matrix and threshold discussion (Task 4.3).

Loads the best saved classification model, runs predictions on the test split,
and evaluates multiple operating thresholds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_model(model_path: Path):
    import joblib
    return joblib.load(model_path)


def _find_best_model_path(tables_dir: Path, models_root: Path) -> Path | None:
    """Try to locate the selected model artifact from final_model_selection.csv."""
    sel_path = tables_dir / "final_model_selection.csv"
    if sel_path.exists():
        sel = pd.read_csv(sel_path)
        clf = sel[(sel.get("task", "") == "classification") & (sel.get("selected", False) == True)]
        if not clf.empty:
            # Look for combined subset first, then dogs, then cats
            for subset in ["combined", "dogs", "cats"]:
                row = clf[clf.get("animal_subset", "") == subset]
                if not row.empty:
                    model_name = row.iloc[0]["model_name"]
                    candidate = models_root / model_name / "classification" / subset / f"{model_name}.joblib"
                    if candidate.exists():
                        return candidate
                    # Try alternative paths
                    for subdir in ["boosting", "advanced", "baseline"]:
                        candidate2 = models_root / subdir / "classification" / subset / f"{model_name}.joblib"
                        if candidate2.exists():
                            return candidate2

    # Fallback: search for any classifier
    for pattern in [
        "*/classification/combined/*.joblib",
        "*/classification/cats/*.joblib",
        "*/classification/dogs/*.joblib",
    ]:
        found = list(models_root.glob(pattern))
        if found:
            return found[0]
    return None


def _evaluate_thresholds(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        confusion_matrix, fbeta_score,
    )

    rows = []

    def _row(label: str, threshold: float) -> dict:
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr = fp / max(fp + tn, 1)
        fnr = fn / max(fn + tp, 1)
        return {
            "threshold_label": label,
            "threshold": round(threshold, 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        }

    # Threshold 1: default
    rows.append(_row("default_0.50", 0.50))

    # Threshold 2: maximize F1
    thresholds = np.linspace(0.05, 0.95, 200)
    f1s = [f1_score(y_true, (y_score >= t).astype(int), zero_division=0) for t in thresholds]
    best_f1_t = float(thresholds[int(np.argmax(f1s))])
    rows.append(_row("max_f1", best_f1_t))

    # Threshold 3: recall >= 0.85 (catch most adopted animals)
    recalls = [recall_score(y_true, (y_score >= t).astype(int), zero_division=0) for t in thresholds]
    high_recall_candidates = [(t, r) for t, r in zip(thresholds, recalls) if r >= 0.85]
    if high_recall_candidates:
        # Among recall>=0.85, pick highest precision
        precisions = [precision_score(y_true, (y_score >= t).astype(int), zero_division=0) for t in thresholds]
        best_t = max(
            [t for t, r in high_recall_candidates],
            key=lambda t: precisions[list(thresholds).index(t)],
        )
        rows.append(_row("high_recall_ge85", float(best_t)))
    else:
        rows.append(_row("high_recall_ge85", 0.30))

    # Threshold 4: balanced (precision ≈ recall)
    diffs = [abs(p - r) for p, r in zip(
        [precision_score(y_true, (y_score >= t).astype(int), zero_division=0) for t in thresholds],
        recalls,
    )]
    balanced_t = float(thresholds[int(np.argmin(diffs))])
    rows.append(_row("balanced_precision_recall", balanced_t))

    df = pd.DataFrame(rows)
    df["threshold_name"] = df["threshold_label"]
    return df


def _plot_confusion_matrices(y_true: np.ndarray, y_score: np.ndarray, thresholds_df: pd.DataFrame, out_path: Path) -> None:
    from sklearn.metrics import confusion_matrix

    t_default = 0.50
    t_f1 = float(thresholds_df.loc[thresholds_df["threshold_label"] == "max_f1", "threshold"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (label, t) in zip(axes, [("Default (0.50)", t_default), (f"F1-Max ({t_f1:.2f})", t_f1)]):
        y_pred = (y_score >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"Confusion Matrix\n{label}", fontsize=12)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Not Adopted", "Adopted"])
        ax.set_yticklabels(["Not Adopted", "Adopted"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)

    fig.suptitle("Final Classifier — Confusion Matrices at Two Thresholds", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[4.3] Wrote {out_path.name}")


def create_threshold_analysis(
    data_path: str | Path = "data/processed/modeling_dataset.csv",
    tables_dir: str | Path = "reports/tables",
    figures_dir: str | Path = "reports/figures",
    summary_dir: str | Path = "reports/summary",
    models_dir: str | Path = "models",
) -> None:
    tables = Path(tables_dir)
    figures = Path(figures_dir)
    summary = Path(summary_dir)
    models_root = Path(models_dir)
    for d in [tables, figures, summary]:
        d.mkdir(parents=True, exist_ok=True)

    model_path = _find_best_model_path(tables, models_root)
    if model_path is None:
        print("[4.3] No saved model found — skipping threshold analysis.")
        return

    print(f"[4.3] Loading model from {model_path}")
    model = _load_model(model_path)

    # Reconstruct test split
    from aac_adoption.models.split import make_time_split
    from aac_adoption.features.feature_sets import available_intake_features

    header = pd.read_csv(data_path, nrows=0)
    parse_dates = [c for c in ["intake_datetime", "outcome_datetime"] if c in header.columns]
    df = pd.read_csv(data_path, parse_dates=parse_dates)

    # Determine animal_subset from path
    for subset_candidate in ["dogs", "cats", "combined"]:
        if subset_candidate in str(model_path):
            animal_subset = subset_candidate
            break
    else:
        animal_subset = "combined"

    split = make_time_split(df, "classification_target", animal_subset=animal_subset)
    feature_cols = available_intake_features(list(split.test.columns))

    try:
        y_score = model.predict_proba(split.test[feature_cols])[:, 1]
    except Exception as e:
        print(f"[4.3] predict_proba failed: {e} — skipping.")
        return

    y_true = split.test["classification_target"].values

    thresholds_df = _evaluate_thresholds(y_true, y_score)
    thresholds_df.to_csv(tables / "final_classifier_thresholds.csv", index=False)
    print(f"[4.3] Wrote final_classifier_thresholds.csv")

    _plot_confusion_matrices(y_true, y_score, thresholds_df, figures / "final_confusion_matrix.png")
    _write_threshold_md(thresholds_df, animal_subset, str(model_path), summary)


def _write_threshold_md(df: pd.DataFrame, animal_subset: str, model_path: str, summary: Path) -> None:
    lines = [
        "# Threshold Selection — Classification Model\n\n",
        f"**Model:** `{model_path}`\n",
        f"**Animal subset:** `{animal_subset}`\n\n",
        "> **Key insight:** Threshold choice is an operational decision, not a model quality metric. "
        "The threshold should be chosen based on the shelter's objective — "
        "maximising adoption identification vs. minimising false alarms.\n\n",
        "## Threshold Comparison\n\n",
        df.to_markdown(index=False),
        "\n\n",
        "## Threshold Interpretations\n\n",
        "| Threshold | Best for | Trade-off |\n",
        "|-----------|----------|----------|\n",
        "| `default_0.50` | Balanced general use | May miss high-recall use cases |\n",
        "| `max_f1` | Maximising F1 score | Balances precision and recall automatically |\n",
        "| `high_recall_ge85` | Flagging all adoption-likely animals | Accepts more false positives |\n",
        "| `balanced_precision_recall` | Equal weight on precision and recall | Not always useful operationally |\n\n",
        "## Thesis Statement\n\n",
        "The classifier is evaluated primarily as a **ranking / decision-support tool**, not a binary decision maker. "
        "The default threshold (0.50) is used for F1 and confusion matrix reporting. "
        "If the shelter wished to use this model operationally, threshold selection would depend on the cost "
        "of false positives (resources allocated to animals unlikely to be adopted) vs. "
        "false negatives (adoption-ready animals not flagged for promotion).\n",
    ]
    (summary / "threshold_selection.md").write_text("".join(lines), encoding="utf-8")
    print(f"[4.3] Wrote threshold_selection.md")
