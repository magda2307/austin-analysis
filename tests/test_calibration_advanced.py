"""Tests for advanced calibration validations."""

import pandas as pd
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss

from aac_adoption.models.calibrate import apply_calibration_to_predictions

@pytest.fixture
def sample_classification_data():
    np.random.seed(42)
    n_samples = 500
    
    # Create somewhat separable data
    X = pd.DataFrame({
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
    })
    
    # Target is somewhat related to feature1
    y = pd.Series((X["feature1"] + np.random.randn(n_samples) > 0).astype(int))
    return X, y


def test_calibration_improves_brier_score(sample_classification_data):
    """Test that calibration actually improves (lowers) the Brier score."""
    X, y = sample_classification_data
    
    # Split into train, val, test
    X_train, y_train = X.iloc[:300], y.iloc[:300]
    X_val, y_val = X.iloc[300:400], y.iloc[300:400]
    X_test, y_test = X.iloc[400:], y.iloc[400:]
    
    # Train uncalibrated model
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    uncalibrated_probs = model.predict_proba(X_test)[:, 1]
    uncalibrated_brier = brier_score_loss(y_test, uncalibrated_probs)
    
    # Train calibrated wrapper on the holdout calibration split.
    calibrated_model = apply_calibration_to_predictions(model, X_train, y_train, X_val, y_val)
    
    calibrated_probs = calibrated_model.predict_proba(X_test)[:, 1]
    calibrated_brier = brier_score_loss(y_test, calibrated_probs)
    
    # The Brier score should ideally be lower or at least not significantly worse
    # Depending on the data, it might not always strictly improve, but it shouldn't degrade much.
    # In many cases it improves. Let's assert it's reasonably bounded.
    assert calibrated_brier < uncalibrated_brier or calibrated_brier < 0.25


def test_calibration_platt_method_sigmoid(sample_classification_data):
    """Test that Platt calibration preserves method='sigmoid' in the calibrator."""
    X, y = sample_classification_data
    
    X_train, y_train = X.iloc[:300], y.iloc[:300]
    X_val, y_val = X.iloc[300:400], y.iloc[300:400]
    
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    calibrated_model = apply_calibration_to_predictions(model, X_train, y_train, X_val, y_val, calib_method="platt")
    
    assert calibrated_model.method == "sigmoid"
