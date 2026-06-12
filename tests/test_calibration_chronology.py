import pandas as pd
import numpy as np
import json
import pytest
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier

from aac_adoption.models.calibrate import _calibration_frame, calibrate_classifiers
from aac_adoption.models.split import DatasetSplit, make_time_split
from aac_adoption.models.metadata import base_training_metadata

def test_calibration_frame_excludes_selection_period():
    calibration = pd.DataFrame(
        {
            "intake_year": [2022, 2022],
            "classification_target": [0, 1],
        }
    )
    selection = pd.DataFrame(
        {
            "intake_year": [2023, 2023],
            "classification_target": [1, 0],
        }
    )
    split = DatasetSplit(
        full_data=pd.concat([calibration, selection], ignore_index=True),
        train=pd.DataFrame(),
        calibration=calibration,
        selection=selection,
        test=pd.DataFrame(),
        strategy="time",
        train_period="2013-2021",
        calibration_period="2022",
        selection_period="2023",
        test_period="2024-2025",
        animal_subset="combined",
        is_thesis_evaluation=True,
    )

    result = _calibration_frame(split)

    assert set(result["intake_year"]) == {2022}
    pd.testing.assert_frame_equal(result, calibration)


def _setup_calibration_test(base_path, data, monkeypatch):
    base_path.mkdir(parents=True, exist_ok=True)
    data_path = base_path / "modeling.csv"
    data.to_csv(data_path, index=False)
    
    source_dir = base_path / "source_models"
    source_dir.mkdir(exist_ok=True)
    
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    feature_cols = ["feature1", "feature2"]
    
    train_data = data[data["intake_datetime"].dt.year <= 2021]
    model.fit(train_data[feature_cols], train_data["classification_target"])
    
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
    
    metadata_path = model_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
        
    monkeypatch.setenv("AAC_RUN_ID", "calibration-test-run")
    monkeypatch.setenv("AAC_PRODUCER_SOURCE_SHA", "calibration-source-sha")
    
    return data_path, source_dir


def test_calibration_artifact_is_invariant_to_selection_labels(tmp_path, monkeypatch):
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2021-01-01", "2024-12-31", periods=n)
    data1 = pd.DataFrame({
        "intake_datetime": dates,
        "outcome_datetime": dates + pd.Timedelta(days=7),
        "intake_year": dates.year,
        "animal_type": ["Dog"] * n,
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "classification_target": np.random.choice([0, 1], n),
    })
    
    run1_path = tmp_path / "run1"
    data_path1, source_dir1 = _setup_calibration_test(run1_path, data1, monkeypatch)
    calibrate_classifiers(
        data_path=data_path1,
        source_artifacts=[(source_dir1, "catboost")],
        metrics_dir=run1_path / "metrics",
        models_dir=run1_path / "models",
    )
    
    data2 = data1.copy()
    mask_2023 = data2["intake_datetime"].dt.year == 2023
    data2.loc[mask_2023, "classification_target"] = 1 - data2.loc[mask_2023, "classification_target"]
    
    run2_path = tmp_path / "run2"
    data_path2, source_dir2 = _setup_calibration_test(run2_path, data2, monkeypatch)
    calibrate_classifiers(
        data_path=data_path2,
        source_artifacts=[(source_dir2, "catboost")],
        metrics_dir=run2_path / "metrics",
        models_dir=run2_path / "models",
    )
    
    model1 = load(run1_path / "models" / "classification_calibrated" / "combined" / "catboost_calibrated.joblib")
    model2 = load(run2_path / "models" / "classification_calibrated" / "combined" / "catboost_calibrated.joblib")
    
    test_data = data1[data1["intake_datetime"].dt.year >= 2024]
    preds1 = model1.predict_proba(test_data[["feature1", "feature2"]])
    preds2 = model2.predict_proba(test_data[["feature1", "feature2"]])
    np.testing.assert_allclose(preds1, preds2)


def test_calibrate_classifiers_emits_both_selection_and_test_rows(tmp_path, monkeypatch):
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2021-01-01", "2024-12-31", periods=n)
    data = pd.DataFrame({
        "intake_datetime": dates,
        "outcome_datetime": dates + pd.Timedelta(days=7),
        "intake_year": dates.year,
        "animal_type": ["Dog"] * n,
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "classification_target": np.random.choice([0, 1], n),
    })
    
    data_path, source_dir = _setup_calibration_test(tmp_path, data, monkeypatch)
    outputs = calibrate_classifiers(
        data_path=data_path,
        source_artifacts=[(source_dir, "catboost")],
        metrics_dir=tmp_path / "metrics",
        models_dir=tmp_path / "models",
    )
    
    df = outputs.classification_metrics
    assert not df.empty
    
    assert "metric_split" in df.columns
    splits = set(df["metric_split"])
    assert "selection" in splits
    assert "test" in splits
    
    sel_rows = df[df["metric_split"] == "selection"]
    test_rows = df[df["metric_split"] == "test"]
    
    assert len(sel_rows) == 1
    assert len(test_rows) == 1
    assert sel_rows.iloc[0]["selection_eligible"] == 1
    assert test_rows.iloc[0]["selection_eligible"] == 0
