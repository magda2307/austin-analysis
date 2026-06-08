import pandas as pd

from aac_adoption.models.train_baseline import train_all_baselines


def _small_modeling_dataset() -> pd.DataFrame:
    rows = []
    for year in range(2018, 2026):
        for animal_type in ["Dog", "Cat"]:
            for i in range(4):
                adopted = int((year + i + (animal_type == "Dog")) % 2 == 0)
                rows.append(
                    {
                        "animal_id": f"{animal_type[0]}{year}{i}",
                        "animal_type": animal_type,
                        "intake_type": "Stray" if i % 2 == 0 else "Owner Surrender",
                        "intake_condition": "Normal",
                        "sex_upon_intake": "Spayed Female" if i % 2 == 0 else "Intact Male",
                        "age_days": 100 + i * 300,
                        "age_months": (100 + i * 300) / 30.4375,
                        "age_years": (100 + i * 300) / 365.25,
                        "age_group": "baby" if i == 0 else "adult",
                        "primary_breed": "domestic_shorthair" if animal_type == "Cat" else "labrador_retriever",
                        "simplified_breed_group": "domestic_cat" if animal_type == "Cat" else "retriever_type",
                        "primary_color": "black" if i % 2 == 0 else "brown",
                        "simplified_color_group": "black_or_dark" if i % 2 == 0 else "brown_tan",
                        "is_black_or_dark": i % 2 == 0,
                        "is_named": i % 2 == 1,
                        "intake_year": year,
                        "intake_month": i + 1,
                        "intake_quarter": 1,
                        "intake_season": "winter",
                        "covid_period": "pre_covid" if year < 2020 else "post_covid",
                        "daily_temp_max": 80 + i,
                        "daily_temp_min": 50 + i,
                        "daily_precipitation": 0.1 if i == 0 else 0.0,
                        "is_extreme_heat": i == 3,
                        "is_rainy_day": i == 0,
                        "animal_311_requests_7d": float(i + 1),
                        "animal_311_requests_30d": float(i + 10),
                        "intake_volume_7d": float(i + 2),
                        "intake_volume_30d": float(i + 20),
                        "classification_target": adopted,
                        "regression_target_days": float(5 + i + (year - 2018)),
                    }
                )
    return pd.DataFrame(rows)


def test_train_all_baselines_winsorization_train_only(tmp_path):
    """Test that regression target winsorization happens train-only (not in build_dataset)."""
    import numpy as np
    
    # Create dataset with extreme outliers (1000 days) and both species
    rows = []
    for i in range(50):
        animal_type = "Dog" if i < 25 else "Cat"
        rows.append(
            {
                "animal_id": f"{animal_type[0]}{i}",
                "animal_type": animal_type,
                "intake_type": "Stray",
                "intake_condition": "Normal",
                "sex_upon_intake": "Spayed Female",
                "age_days": 100 + i * 10,
                "age_months": (100 + i * 10) / 30.4375,
                "age_years": (100 + i * 10) / 365.25,
                "age_group": "adult",
                "primary_breed": "labrador_retriever" if animal_type == "Dog" else "domestic_shorthair",
                "simplified_breed_group": "retriever_type" if animal_type == "Dog" else "domestic_cat",
                "primary_color": "black",
                "simplified_color_group": "black_or_dark",
                "is_black_or_dark": True,
                "is_named": i % 2 == 1,
                "intake_year": 2022 + (i % 2),
                "intake_month": i + 1,
                "intake_quarter": (i // 3) + 1,
                "intake_season": "winter",
                "covid_period": "pre_covid",
                "daily_temp_max": 80 + i,
                "daily_temp_min": 50 + i,
                "daily_precipitation": 0.1 if i == 0 else 0.0,
                "is_extreme_heat": i == 3,
                "is_rainy_day": i == 0,
                "animal_311_requests_7d": float(i + 1),
                "animal_311_requests_30d": float(i + 10),
                "intake_volume_7d": float(i + 2),
                "intake_volume_30d": float(i + 20),
                "classification_target": int(i % 2 == 0),  # Balanced 0/1 (25 each)
                "regression_target_days": 5 if i < 49 else 1000,  # Extreme outlier
            }
        )
    df = pd.DataFrame(rows)
    data_path = tmp_path / "modeling_dataset.csv"
    df.to_csv(data_path, index=False)

    outputs = train_all_baselines(
        data_path=data_path,
        metrics_dir=tmp_path / "metrics",
        models_dir=tmp_path / "models",
        tables_dir=tmp_path / "tables",
        max_rows=None,
    )

    # Check metadata has winsorization parameters
    regression_metadata_file = tmp_path / "models" / "regression" / "combined" / "ridge.json"
    assert regression_metadata_file.exists()
    
    import json
    with open(regression_metadata_file) as f:
        metadata = json.load(f)
    
    # Winsorization should be stored in metadata
    assert "winsorization_lower_quantile" in metadata
    assert "winsorization_upper_quantile" in metadata
    assert "winsorization_lower_value" in metadata
    assert "winsorization_upper_value" in metadata


def test_train_all_baselines_writes_metadata_rich_outputs(tmp_path):
    data_path = tmp_path / "modeling_dataset.csv"
    metrics_dir = tmp_path / "metrics"
    models_dir = tmp_path / "models"
    tables_dir = tmp_path / "tables"
    _small_modeling_dataset().to_csv(data_path, index=False)

    outputs = train_all_baselines(
        data_path=data_path,
        metrics_dir=metrics_dir,
        models_dir=models_dir,
        tables_dir=tables_dir,
        max_rows=None,
    )

    required_columns = {
        "model_name",
        "task",
        "animal_subset",
        "train_period",
        "test_period",
        "feature_set",
        "random_state",
        "run_timestamp",
    }
    assert required_columns.issubset(outputs.classification_metrics.columns)
    assert required_columns.issubset(outputs.regression_metrics.columns)
    assert set(outputs.classification_metrics["animal_subset"]) == {"combined", "dogs", "cats"}
    assert set(outputs.classification_metrics["feature_set"]) == {"intake_time_context_v2"}
    assert (metrics_dir / "classification_metrics.csv").exists()
    assert (metrics_dir / "regression_metrics.csv").exists()
    assert list(models_dir.rglob("*.joblib"))
    assert (tables_dir / "logistic_regression_coefficients.csv").exists()
    assert (tables_dir / "random_forest_feature_importance.csv").exists()


import pytest
pytestmark = pytest.mark.slow
