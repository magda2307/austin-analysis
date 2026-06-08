"""Bootstrap confidence interval computation."""

from typing import Optional
import numpy as np

from aac_adoption.config import RANDOM_STATE
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)



def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: callable,
    y_score: Optional[np.ndarray] = None,
    n_bootstraps: int = 1000,
    random_state: int = RANDOM_STATE,
    animal_ids: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Calculate 95% confidence interval for a metric using bootstrapping.
    
    When animal_ids is provided, performs cluster-aware bootstrap by resampling
    animals with replacement and including all observations from selected animals.
    Falls back to row-level bootstrap when animal_ids is None or empty.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        metric_func: Function to compute metric (y_true, y_pred) or (y_true, y_score)
        y_score: Optional probability scores for metric computation
        n_bootstraps: Number of bootstrap iterations
        random_state: Random seed for reproducibility
        animal_ids: Cluster identifiers for cluster-aware bootstrap
        
    Returns:
        Tuple of (lower_bound, upper_bound) for 95% CI
    """
    rng = np.random.default_rng(random_state)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_score_arr = np.asarray(y_score) if y_score is not None else None
    
    if animal_ids is not None and len(animal_ids) > 0:
        animal_ids_arr = np.asarray(animal_ids)
        if not (len(y_true_arr) == len(y_pred_arr) == len(animal_ids_arr)):
            raise ValueError("Input lengths must match")
        unique_animals = np.unique(animal_ids_arr)
        animal_to_indices = {aid: np.where(animal_ids_arr == aid)[0] for aid in unique_animals}
        scores = []
        
        for _ in range(n_bootstraps):
            sampled_animals = rng.choice(unique_animals, size=len(unique_animals), replace=True)
            sample_indices = np.concatenate([animal_to_indices[aid] for aid in sampled_animals])
            

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


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: callable,
    animal_ids: np.ndarray,
    n_bootstraps: int = 1000,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """Cluster-aware bootstrap CI by animal_id.
    
    Wrapper around bootstrap_ci for explicit cluster-aware usage.
    """
    return bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=metric_func,
        n_bootstraps=n_bootstraps,
        random_state=random_state,
        animal_ids=np.asarray(animal_ids),
    )


def row_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: callable,
    n_bootstraps: int = 1000,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """Row-level bootstrap CI (fallback when cluster data unavailable).
    
    Note: This assumes rows are independent, which may not hold for 
    multiple episodes per animal.
    """
    return bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=metric_func,
        n_bootstraps=n_bootstraps,
        random_state=random_state,
        animal_ids=None,
    )
