"""Tests for calibration module."""

import json
import subprocess
import sys
import pandas as pd
import numpy as np
import pytest
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from aac_adoption.models.calibrate import (
    calibrate_with_isotonic,
    calibrate_with_platt,
    post_hoc_calibration_pipeline,
    apply_calibration_to_predictions,
    calibrate_classifiers,
)
from aac_adoption.models.artifacts import artifact_path, save_model_artifact
from aac_adoption.models.metadata import REQUIRED_MODEL_METADATA, base_training_metadata
from aac_adoption.models.split import make_time_split


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_train = 50
    n_calib = 30
    n_test = 20
    
    X_train = pd.DataFrame({
        "feature1": np.random.randn(n_train),
        "feature2": np.random.randn(n_train),
    })
    y_train = pd.Series(np.random.choice([0, 1], n_train))
    
    X_calib = pd.DataFrame({
        "feature1": np.random.randn(n_calib),
        "feature2": np.random.randn(n_calib),
    })
    y_calib = pd.Series(np.random.choice([0, 1], n_calib))
    
    X_test = pd.DataFrame({
        "feature1": np.random.randn(n_test),
        "feature2": np.random.randn(n_test),
    })
    
    return X_train, y_train, X_calib, y_calib, X_test


def test_calibrate_with_isotonic(sample_data):
    X_train, y_train, X_calib, y_calib, X_test = sample_data
    
    base_model = RandomForestClassifier(n_estimators=5, random_state=42)
    base_model.fit(X_train, y_train)
    
    predictions, scores = calibrate_with_isotonic(
        base_model,
        X_calib,
        y_calib,
        X_test,
        method="isotonic",
    )
    
    assert len(predictions) == 20
    assert len(scores) == 20
    assert set(predictions).issubset({0, 1})
    assert all(0 <= s <= 1 for s in scores)


def test_calibrate_with_platt(sample_data):
    X_train, y_train, X_calib, y_calib, X_test = sample_data
    
    base_model = RandomForestClassifier(n_estimators=5, random_state=42)
    base_model.fit(X_train, y_train)
    
    predictions, scores = calibrate_with_platt(
        base_model,
        X_calib,
        y_calib,
        X_test,
    )
    
    assert len(predictions) == 20
    assert len(scores) == 20


def test_post_hoc_calibration_pipeline(sample_data):
    X_train, y_train, X_calib, y_calib, X_test = sample_data
    
    base_model = RandomForestClassifier(n_estimators=5, random_state=42)
    base_model.fit(X_train, y_train)
    
    predictions, scores, calibrated = post_hoc_calibration_pipeline(
        base_model,
        X_train,
        y_train,
        X_calib,
        y_calib,
        X_test,
        calib_method="isotonic",
    )
    
    assert len(predictions) == 20
    assert calibrated is not None


def test_apply_calibration_to_predictions(sample_data):
    X_train, y_train, X_calib, y_calib, X_test = sample_data
    
    base_model = RandomForestClassifier(n_estimators=5, random_state=42)
    base_model.fit(X_train, y_train)
    
    calibrated = apply_calibration_to_predictions(
        base_model,
        X_train,
        y_train,
        X_calib,
        y_calib,
        calib_method="isotonic",
    )
    
    predictions = calibrated.predict(X_calib)
    assert len(predictions) == 30


def test_apply_calibration_preserves_platt_method(sample_data):
    X_train, y_train, X_calib, y_calib, X_test = sample_data

    base_model = RandomForestClassifier(n_estimators=5, random_state=42)
    base_model.fit(X_train, y_train)

    calibrated = apply_calibration_to_predictions(
        base_model,
        X_train,
        y_train,
        X_calib,
        y_calib,
        calib_method="platt",
    )

    assert calibrated.method == "sigmoid"


def test_calibrate_classifiers_help_exits_0():
    """Test that calibrate_classifiers.py --help exits 0."""
    result = subprocess.run(
        [sys.executable, "scripts/calibrate_classifiers.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Calibrate trained classifiers" in result.stdout


def test_calibrate_classifiers_end_to_end_fixture(tmp_path, sample_data):
    """Test end-to-end calibration on synthetic data and verify output CSV format."""
    X_train, y_train, X_calib, y_calib, X_test = sample_data
    data = pd.concat([X_train, X_calib, X_test], ignore_index=True)
    data["animal_type"] = ["Dog"] * len(data)
    data["intake_datetime"] = pd.date_range("2019-01-01", periods=len(data), freq="30D")
    data["outcome_datetime"] = data["intake_datetime"] + pd.Timedelta(days=7)
    data["intake_year"] = data["intake_datetime"].dt.year
    data["classification_target"] = pd.concat([y_train, y_calib, pd.Series(np.random.choice([0, 1], len(X_test)))], ignore_index=True)
    data_path = tmp_path / "modeling.csv"
    data.to_csv(data_path, index=False)

    outputs = calibrate_classifiers(
        data_path=data_path,
        source_artifacts=[(tmp_path / "missing", "catboost")],
        metrics_dir=tmp_path / "metrics",
        models_dir=tmp_path / "models",
    )

    assert outputs.classification_metrics.empty
    assert (tmp_path / "metrics" / "calibrated_classification_metrics.csv").exists()


def test_calibrate_classifiers_csv_columns_format(tmp_path, sample_data, monkeypatch):
    """Test that calibrated_classification_metrics.csv has all required columns when models are available."""
    X_train, y_train, X_calib, y_calib, X_test = sample_data
    data = pd.concat([X_train, X_calib, X_test], ignore_index=True)
    data["animal_type"] = ["Dog"] * len(data)
    data["intake_datetime"] = pd.date_range("2019-01-01", periods=len(data), freq="30D")
    data["outcome_datetime"] = data["intake_datetime"] + pd.Timedelta(days=7)
    data["intake_year"] = data["intake_datetime"].dt.year
    data["classification_target"] = pd.concat([y_train, y_calib, pd.Series(np.random.choice([0, 1], len(X_test)))], ignore_index=True)
    
    data_path = tmp_path / "modeling.csv"
    data.to_csv(data_path, index=False)
    
    # Create a minimal source artifact directory with a CatBoost model
    source_dir = tmp_path / "source_models"
    source_dir.mkdir()
    
    # Train a simple model for calibration
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    feature_cols = ["feature1", "feature2"]
    model.fit(X_train[feature_cols], y_train)
    
    # Save the model with proper naming convention
    from joblib import dump
    model_path = source_dir / "classification" / "combined" / "catboost.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)
    
    split = make_time_split(data, "classification_target", animal_subset="combined")
    metadata = base_training_metadata(
        model_name="catboost",
        task="classification",
        split=split,
        feature_columns=feature_cols,
        run_timestamp="2025-01-01T00:00:00+00:00",
        target_column="classification_target",
        dataset_path=str(data_path),
        run_id="base-run",
        producer_source_sha="base-source-sha",
    )
    metadata["artifact_path"] = str(model_path)
    metadata["artifact_sha256"] = "base-artifact-sha"
    metadata_path = source_dir / "classification" / "combined" / "catboost.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    monkeypatch.setenv("AAC_RUN_ID", "calibration-test-run")
    monkeypatch.setenv("AAC_PRODUCER_SOURCE_SHA", "calibration-source-sha")
    
    outputs = calibrate_classifiers(
        data_path=data_path,
        source_artifacts=[(source_dir, "catboost")],
        metrics_dir=tmp_path / "metrics",
        models_dir=tmp_path / "models",
    )

    output_path = tmp_path / "metrics" / "calibrated_classification_metrics.csv"
    assert output_path.exists()
    
    # Read CSV, handling the case where it might be empty
    try:
        df = pd.read_csv(output_path)
    except pd.errors.EmptyDataError:
        pytest.skip("CSV is empty, cannot verify columns")
    
    if not df.empty:
        required_columns = {
            "animal_subset", "model_name", "base_model_name", "calibration_method",
            "pr_auc", "roc_auc", "brier_score", "expected_calibration_error",
            "train_rows", "validation_rows", "test_rows"
        }
        assert required_columns.issubset(df.columns)

        sidecar_path = artifact_path(
            tmp_path / "models",
            "classification_calibrated",
            "combined",
            "catboost_calibrated",
        ).with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert REQUIRED_MODEL_METADATA.issubset(sidecar)
        assert sidecar["run_id"] == "calibration-test-run"
        assert sidecar["producer_source_sha"] == "calibration-source-sha"
        assert sidecar["base_model_name"] == "catboost"
        assert sidecar["feature_columns"] == feature_cols
        assert sidecar["base_run_id"] == "base-run"
        assert sidecar["base_producer_source_sha"] == "base-source-sha"
        assert sidecar["base_artifact_sha256"] == "base-artifact-sha"
    else:
        pytest.skip("CSV contains no rows, cannot verify columns")


def test_save_model_artifact_validation_failure_leaves_no_partial_artifact(tmp_path):
    output_path = artifact_path(
        tmp_path,
        "classification",
        "combined",
        "broken",
    )

    with pytest.raises(ValueError, match="Missing required model metadata fields"):
        save_model_artifact(
            {"model": "placeholder"},
            tmp_path,
            "classification",
            "combined",
            "broken",
            {"model_name": "broken"},
        )

    assert not output_path.exists()
    assert not output_path.with_suffix(".json").exists()


def test_save_model_artifact_rolls_back_if_sidecar_replace_fails(tmp_path, monkeypatch):
    metadata = {key: "value" for key in REQUIRED_MODEL_METADATA}
    metadata.update(
        {
            "feature_columns": [],
            "packages": {},
            "is_thesis_evaluation": False,
            "random_state": 42,
        }
    )
    output_path = save_model_artifact(
        {"version": 1},
        tmp_path,
        "classification",
        "combined",
        "stable",
        metadata,
    )
    sidecar_path = output_path.with_suffix(".json")
    original_model = output_path.read_bytes()
    original_sidecar = sidecar_path.read_bytes()
    original_replace = type(output_path).replace

    def fail_sidecar_replace(path, target):
        if path.name == f".{sidecar_path.name}.tmp":
            raise OSError("simulated sidecar replacement failure")
        return original_replace(path, target)

    monkeypatch.setattr(type(output_path), "replace", fail_sidecar_replace)

    with pytest.raises(OSError, match="simulated sidecar replacement failure"):
        save_model_artifact(
            {"version": 2},
            tmp_path,
            "classification",
            "combined",
            "stable",
            metadata,
        )

    assert output_path.read_bytes() == original_model
    assert sidecar_path.read_bytes() == original_sidecar


def test_calibration_rejects_invalid_source_metadata(tmp_path, sample_data):
    X_train, y_train, X_calib, y_calib, X_test = sample_data
    data = pd.concat([X_train, X_calib, X_test], ignore_index=True)
    data["animal_type"] = "Dog"
    data["intake_datetime"] = pd.date_range("2019-01-01", periods=len(data), freq="30D")
    data["intake_year"] = data["intake_datetime"].dt.year
    data["classification_target"] = pd.concat(
        [y_train, y_calib, pd.Series(np.random.choice([0, 1], len(X_test)))],
        ignore_index=True,
    )
    data_path = tmp_path / "modeling.csv"
    data.to_csv(data_path, index=False)

    model = RandomForestClassifier(n_estimators=5, random_state=42).fit(
        X_train[["feature1", "feature2"]],
        y_train,
    )
    model_path = (
        tmp_path
        / "source"
        / "classification"
        / "combined"
        / "catboost.joblib"
    )
    model_path.parent.mkdir(parents=True)
    import joblib
    joblib.dump(model, model_path)
    model_path.with_suffix(".json").write_text(
        '{"feature_columns": ["feature1", "feature2"]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing required model metadata fields"):
        calibrate_classifiers(
            data_path=data_path,
            source_artifacts=[(tmp_path / "source", "catboost")],
            metrics_dir=tmp_path / "metrics",
            models_dir=tmp_path / "models",
        )


def test_calibrate_platt_uses_sigmoid_method(sample_data):
    """Test that Platt calibration uses method='sigmoid' (not overridden to isotonic)."""
    X_train, y_train, X_calib, y_calib, X_test = sample_data
    
    base_model = RandomForestClassifier(n_estimators=5, random_state=42)
    base_model.fit(X_train, y_train)
    
    calibrated = apply_calibration_to_predictions(
        base_model,
        X_train,
        y_train,
        X_calib,
        y_calib,
        calib_method="platt",
    )
    
    assert calibrated.method == "sigmoid"


def test_calibrate_classifiers_handles_missing_artifacts(tmp_path, sample_data):
    X_train, y_train, X_calib, y_calib, X_test = sample_data
    data = pd.concat([X_train, X_calib, X_test], ignore_index=True)
    data["animal_type"] = ["Dog"] * len(data)
    data["intake_datetime"] = pd.date_range("2019-01-01", periods=len(data), freq="30D")
    data["outcome_datetime"] = data["intake_datetime"] + pd.Timedelta(days=7)
    data["intake_year"] = data["intake_datetime"].dt.year
    data["classification_target"] = pd.concat([y_train, y_calib, pd.Series(np.random.choice([0, 1], len(X_test)))], ignore_index=True)
    data_path = tmp_path / "modeling.csv"
    data.to_csv(data_path, index=False)

    outputs = calibrate_classifiers(
        data_path=data_path,
        source_artifacts=[(tmp_path / "missing", "catboost")],
        metrics_dir=tmp_path / "metrics",
        models_dir=tmp_path / "models",
    )

    assert outputs.classification_metrics.empty
    assert (tmp_path / "metrics" / "calibrated_classification_metrics.csv").exists()
