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


from aac_adoption.models.bootstrap import bootstrap_ci


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


HORIZON_DAYS = [7, 30, 60, 90]


def subgroup_analysis(y_true, y_pred, y_score, subgroup_column, subgroup_names=None, animal_ids=None):
    """Compute metrics separately for each subgroup (e.g., dogs/cats).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        y_score: Probability scores
        subgroup_column: Array or Series of subgroup labels
        subgroup_names: Optional list of subgroup names for ordered output
        animal_ids: Optional array of cluster identifiers for bootstrap CI
    
    Returns:
        DataFrame with metrics by subgroup
    """
    subgroup_arr = np.asarray(subgroup_column)
    unique_subgroups = np.unique(subgroup_arr)
    
    if subgroup_names is None:
        subgroup_names = sorted(unique_subgroups)
    
    results = []
    for subgroup in subgroup_names:
        mask = subgroup_arr == subgroup
        y_true_sub = np.asarray(y_true)[mask]
        y_score_sub = np.asarray(y_score)[mask]
        
        if len(y_true_sub) < 2 or len(np.unique(y_true_sub)) < 2:
            continue
        
        if animal_ids is not None:
            animal_ids_sub = np.asarray(animal_ids)[mask]
        else:
            animal_ids_sub = None
            
        metrics = classification_metrics_with_ci(
            y_true_sub, 
            y_pred[mask], 
            y_score_sub, 
            n_bootstraps=1000,
            animal_ids=animal_ids_sub
        )
        metrics["subgroup"] = subgroup
        metrics["n_samples"] = int(mask.sum())
        results.append(metrics)
    
    return pd.DataFrame(results)


def classification_metrics_with_ci(y_true, y_pred, y_score, n_bootstraps: int = 1000, animal_ids=None) -> dict[str, float]:
    """Compute classification metrics with bootstrap 95% CI.
    
    Returns PR-AUC, ROC-AUC, Brier score, ECE, and their 95% bootstrap CIs.
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    y_pred_arr = np.asarray(y_pred)
    
    base_metrics = {
        "pr_auc": float(average_precision_score(y_true_arr, y_score_arr)),
        "roc_auc": float(roc_auc_score(y_true_arr, y_score_arr)),
        "brier_score": float(brier_score_loss(y_true_arr, y_score_arr)),
        "expected_calibration_error": float(expected_calibration_error(y_true_arr, y_score_arr)),
    }
    
    ci_pr = bootstrap_ci(y_true_arr, y_pred_arr, average_precision_score, y_score=y_score_arr, n_bootstraps=n_bootstraps, animal_ids=animal_ids)
    ci_roc = bootstrap_ci(y_true_arr, y_pred_arr, roc_auc_score, y_score=y_score_arr, n_bootstraps=n_bootstraps, animal_ids=animal_ids)
    ci_brier = bootstrap_ci(y_true_arr, y_pred_arr, brier_score_loss, y_score=y_score_arr, n_bootstraps=n_bootstraps, animal_ids=animal_ids)
    ci_ece = bootstrap_ci(y_true_arr, y_pred_arr, expected_calibration_error, y_score=y_score_arr, n_bootstraps=n_bootstraps, animal_ids=animal_ids)
    
    return {
        "pr_auc": base_metrics["pr_auc"],
        "pr_auc_lower": ci_pr[0],
        "pr_auc_upper": ci_pr[1],
        "roc_auc": base_metrics["roc_auc"],
        "roc_auc_lower": ci_roc[0],
        "roc_auc_upper": ci_roc[1],
        "brier_score": base_metrics["brier_score"],
        "brier_lower": ci_brier[0],
        "brier_upper": ci_brier[1],
        "expected_calibration_error": base_metrics["expected_calibration_error"],
        "ece_lower": ci_ece[0],
        "ece_upper": ci_ece[1],
    }
