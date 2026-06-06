"""Model evaluation helpers."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true, y_score, bins: int = 10) -> float:
    """Return fixed-bin expected calibration error for binary probabilities."""
    y_true_array = np.asarray(y_true, dtype=float)
    y_score_array = np.asarray(y_score, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (y_score_array >= lower) & (y_score_array <= upper)
        else:
            mask = (y_score_array >= lower) & (y_score_array < upper)
        if not mask.any():
            continue
        bin_weight = float(mask.mean())
        observed = float(y_true_array[mask].mean())
        predicted = float(y_score_array[mask].mean())
        ece += bin_weight * abs(observed - predicted)
    return float(ece)


def classification_metrics(y_true, y_pred, y_score=None) -> dict[str, float | int | None]:
    """Return standard binary classification metrics."""
    metrics: dict[str, float | int | None] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": None,
        "pr_auc": None,
        "brier_score": None,
        "expected_calibration_error": None,
    }

    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
        metrics["brier_score"] = brier_score_loss(y_true, y_score)
        metrics["expected_calibration_error"] = expected_calibration_error(y_true, y_score)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return standard regression metrics for LOS/time prediction."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mse)),
        "median_absolute_error": median_absolute_error(y_true, y_pred),
    }
