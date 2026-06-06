"""Tests for concept drift detection."""

import pandas as pd
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def simulate_concept_drift_data():
    """Simulate data that drifts over time."""
    np.random.seed(42)
    
    # Month 1: normal
    X1 = pd.DataFrame({"f1": np.random.randn(100), "f2": np.random.randn(100)})
    y1 = (X1["f1"] + X1["f2"] > 0).astype(int)
    
    # Month 2: slight drift
    X2 = pd.DataFrame({"f1": np.random.randn(100) + 0.5, "f2": np.random.randn(100)})
    y2 = (X1["f1"] + X1["f2"] > 0).astype(int)  # The relationship starts changing
    
    # Month 3: severe drift (relationship flips)
    X3 = pd.DataFrame({"f1": np.random.randn(100), "f2": np.random.randn(100)})
    y3 = (X3["f1"] + X3["f2"] < 0).astype(int)
    
    return [(X1, y1), (X2, y2), (X3, y3)]


def test_detect_concept_drift_metric_degradation():
    """Test that a model's performance degrades when concept drift occurs."""
    datasets = simulate_concept_drift_data()
    X_train, y_train = datasets[0]
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    aucs = []
    for X, y in datasets:
        probs = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
        aucs.append(auc)
    
    # AUC should be high on Month 1
    assert aucs[0] > 0.8
    # AUC should degrade significantly by Month 3 due to the flipped relationship
    assert aucs[2] < aucs[0]
    assert aucs[2] < 0.6
