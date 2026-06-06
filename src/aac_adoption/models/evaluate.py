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
    r2_score,
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


from aac_adoption.config import RANDOM_STATE

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: callable,
    y_score: np.ndarray | None = None,
    n_bootstraps: int = 1000,
    random_state: int = RANDOM_STATE,
    animal_ids: np.ndarray | None = None,
) -> tuple[float, float]:
    """Calculate 95% confidence interval for a metric using bootstrapping.
    
    When animal_ids is provided, performs cluster-aware bootstrap by resampling
    animals with replacement and including all observations from selected animals.
    Falls back to row-level bootstrap when animal_ids is None or empty.
    """
    rng = np.random.default_rng(random_state)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_score_arr = np.asarray(y_score) if y_score is not None else None
    
    if animal_ids is not None and len(animal_ids) > 0:
        animal_ids_arr = np.asarray(animal_ids)
        unique_animals = np.unique(animal_ids_arr)
        animal_to_indices = {aid: np.where(animal_ids_arr == aid)[0] for aid in unique_animals}
        
        for _ in range(n_bootstraps):
            sampled_animals = rng.choice(unique_animals, size=len(unique_animals), replace=True)
            sample_indices = np.concatenate([animal_to_indices[aid] for aid in sampled_animals])
            sample_indices = np.unique(sample_indices)
            
            if len(np.unique(y_true_arr[sample_indices])) < 2:
                continue
            if y_score_arr is not None:
                score = metric_func(y_true_arr[sample_indices], y_score_arr[sample_indices])
            else:
                score = metric_func(y_true_arr[sample_indices], y_pred_arr[sample_indices])
            scores.append(score)
    else:
        indices = np.arange(len(y_true))
        scores = []
        
        for _ in range(n_bootstraps):
            idx = rng.choice(indices, size=len(indices), replace=True)
            if len(np.unique(y_true_arr[idx])) < 2:
                continue
            if y_score_arr is not None:
                score = metric_func(y_true_arr[idx], y_score_arr[idx])
            else:
                score = metric_func(y_true_arr[idx], y_pred_arr[idx])
            scores.append(score)
        
        if not scores:
            return (np.nan, np.nan)
        return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))
    
    if not scores:
        return (np.nan, np.nan)
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def classification_metrics(y_true, y_pred, y_score=None, compute_ci=False) -> dict[str, float | int | None]:
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

        if compute_ci:
            ci_roc = bootstrap_ci(y_true, y_pred, roc_auc_score, y_score=y_score)
            ci_pr = bootstrap_ci(y_true, y_pred, average_precision_score, y_score=y_score)
            metrics["roc_auc_lower"] = ci_roc[0]
            metrics["roc_auc_upper"] = ci_roc[1]
            metrics["pr_auc_lower"] = ci_pr[0]
            metrics["pr_auc_upper"] = ci_pr[1]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics


def regression_metrics(y_true, y_pred, compute_ci=False) -> dict[str, float]:
    """Return standard regression metrics for LOS/time prediction."""
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mse)),
        "median_absolute_error": median_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
    
    if compute_ci:
        ci_mae = bootstrap_ci(y_true, y_pred, mean_absolute_error)
        metrics["mae_lower"] = ci_mae[0]
        metrics["mae_upper"] = ci_mae[1]
        
    return metrics
