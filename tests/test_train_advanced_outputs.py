import inspect

import pandas as pd

from aac_adoption.models.train_advanced import (
    train_advanced_classification,
    train_advanced_regression,
    train_all_advanced,
)


def test_advanced_trainers_accept_tuned_catboost_regularization():
    for trainer in [train_advanced_classification, train_advanced_regression]:
        parameters = inspect.signature(trainer).parameters
        assert "l2_leaf_reg" in parameters
        assert "subsample" in parameters


def _small_modeling_dataset() -> pd.DataFrame:
    rows = []
    for year in range(2018, 2026):
        for animal_type in ["Dog", "Cat"]:
            for i in range(6):
                adopted = int((year + i + (animal_type == "Dog")) % 2 == 0)
                rows.append(
                    {
                        "animal_id": f"{animal_type[0]}{year}{i}",
                        "animal_type": animal_type,
                        "intake_type": "Stray" if i % 2 == 0 else "Owner Surrender",
                        "intake_condition": "Normal" if i < 3 else "Sick",
                        "sex_upon_intake": "Spayed Female" if i % 2 == 0 else "Intact Male",
                        "age_upon_intake": f"{i + 1} years",
                        "age_days": 120 + i * 250,
                        "age_months": (120 + i * 250) / 30.4375,
                        "age_years": (120 + i * 250) / 365.25,
                        "age_group": "baby" if i == 0 else "adult",
                        "breed": "Domestic Shorthair Mix" if animal_type == "Cat" else "Labrador Retriever Mix",
                        "primary_breed": "domestic_shorthair" if animal_type == "Cat" else "labrador_retriever",
                        "is_mixed_breed": True,
                        "simplified_breed_group": "domestic_cat" if animal_type == "Cat" else "retriever_type",
                        "color": "Black/White" if i % 2 == 0 else "Brown",
                        "primary_color": "black" if i % 2 == 0 else "brown",
                        "simplified_color_group": "black_or_dark" if i % 2 == 0 else "brown_tan",
                        "is_black_or_dark": i % 2 == 0,
                        "has_name": i % 2 == 1,
                        "is_named": i % 2 == 1,
                        "intake_year": year,
                        "intake_month": i + 1,
                        "intake_quarter": 1,
                        "intake_season": "winter",
                        "covid_period": "pre_covid" if year < 2020 else "post_covid",
                        "classification_target": adopted,
                        "regression_target_days": float(5 + i + (year - 2018)),
                    }
                )
    return pd.DataFrame(rows)


def test_train_all_advanced_writes_metrics_and_artifacts(tmp_path):
    data_path = tmp_path / "modeling_dataset.csv"
    metrics_dir = tmp_path / "metrics"
    models_dir = tmp_path / "models"
    _small_modeling_dataset().to_csv(data_path, index=False)

    outputs = train_all_advanced(
        data_path=data_path,
        metrics_dir=metrics_dir,
        models_dir=models_dir,
        iterations=5,
        learning_rate=0.1,
        depth=2,
        early_stopping_rounds=2,
    )

    required = {"model_name", "task", "animal_subset", "feature_columns", "categorical_features", "artifact_path"}
    assert required.issubset(outputs.classification_metrics.columns)
    assert required.issubset(outputs.regression_metrics.columns)
    assert set(outputs.classification_metrics["animal_subset"]) == {"combined", "dogs", "cats"}
    assert (metrics_dir / "advanced_classification_metrics.csv").exists()
    assert (metrics_dir / "advanced_regression_metrics.csv").exists()
    assert list(models_dir.rglob("*.joblib"))



import pytest
pytestmark = pytest.mark.slow
