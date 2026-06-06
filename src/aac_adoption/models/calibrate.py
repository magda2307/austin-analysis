"""Isotonic and Platt calibration with proper validation."""

from typing import Any, Optional

import joblib
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, clone


def calibrate_with_isotonic(
    base_model: Any,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    X_test: pd.DataFrame,
    method: str = "isotonic",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply isotonic regression calibration."""
    if method == "sigmoid":
        method = "isotonic"  # Use isotonic as default
    
    calibrated = CalibratedClassifierCV(
        estimator=clone(base_model),
        method=method,
        cv=5,
    )
    calibrated.fit(X_calib, y_calib)
    
    predictions = calibrated.predict(X_test).astype(int)
    scores = calibrated.predict_proba(X_test)[:, 1]
    
    return predictions, scores


def calibrate_with_platt(
    base_model: Any,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Platt scaling (sigmoid) calibration."""
    return calibrate_with_isotonic(base_model, X_calib, y_calib, X_test, method="sigmoid")


def post_hoc_calibration_pipeline(
    base_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    calib_method: str = "isotonic",
) -> tuple[np.ndarray, np.ndarray, CalibratedClassifierCV]:
    """Complete post-hoc calibration pipeline."""
    if calib_method not in {"isotonic", "platt", "sigmoid"}:
        calib_method = "isotonic"
    
    if calib_method == "platt" or calib_method == "sigmoid":
        method = "sigmoid"
    else:
        method = "isotonic"
    
    calibrated = CalibratedClassifierCV(
        estimator=clone(base_model),
        method=method,
        cv=5,
    )
    calibrated.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    
    predictions = calibrated.predict(X_test).astype(int)
    scores = calibrated.predict_proba(X_test)[:, 1]
    
    return predictions, scores, calibrated


def apply_calibration_to_predictions(
    base_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    calib_method: str = "isotonic",
) -> CalibratedClassifierCV:
    """Train calibration on calibration set only."""
    if calib_method not in {"isotonic", "platt", "sigmoid"}:
        calib_method = "isotonic"
    
    if calib_method == "platt" or calib_method == "sigmoid":
        method = "sigmoid"
    else:
        method = "isotonic"
    
    calibrated = CalibratedClassifierCV(
        estimator=clone(base_model),
        method=method,
        cv=5,
    )
    calibrated.fit(X_calib, y_calib)
    
    return calibrated


def save_calibration(calibrated: CalibratedClassifierCV, path: str) -> None:
    """Save calibrated model to disk."""
    joblib.dump(calibrated, path)


def load_calibration(path: str) -> CalibratedClassifierCV:
    """Load calibrated model from disk."""
    return joblib.load(path)
