import pandas as pd

from aac_adoption.models.train_boosting import train_all_boosting


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
                        "classification_target": adopted,
                        "regression_target_days": float(5 + i + (year - 2018)),
                    }
                )
    return pd.DataFrame(rows)


def test_train_all_boosting_writes_metrics_artifacts_and_permutation_tables(tmp_path):
    data_path = tmp_path / "modeling_dataset.csv"
    metrics_dir = tmp_path / "metrics"
    models_dir = tmp_path / "models"
    tables_dir = tmp_path / "tables"
    _small_modeling_dataset().to_csv(data_path, index=False)

    outputs = train_all_boosting(
        data_path=data_path,
        metrics_dir=metrics_dir,
        models_dir=models_dir,
        tables_dir=tables_dir,
        max_rows=None,
        permutation_repeats=1,
    )

    required = {
        "model_name",
        "task",
        "animal_subset",
        "train_period",
        "test_period",
        "feature_set",
        "random_state",
        "run_timestamp",
    }
    assert required.issubset(outputs.classification_metrics.columns)
    assert required.issubset(outputs.regression_metrics.columns)
    assert set(outputs.classification_metrics["animal_subset"]) == {"combined", "dogs", "cats"}
    assert (metrics_dir / "boosting_classification_metrics.csv").exists()
    assert (metrics_dir / "boosting_regression_metrics.csv").exists()
    assert list(models_dir.rglob("*.joblib"))
    assert (tables_dir / "permutation_importance_classification.csv").exists()
    assert (tables_dir / "permutation_importance_regression.csv").exists()


def test_train_all_boosting_permutation_tables_evaluation_period(tmp_path):
    data_path = tmp_path / "modeling_dataset.csv"
    metrics_dir = tmp_path / "metrics"
    models_dir = tmp_path / "models"
    tables_dir = tmp_path / "tables"
    _small_modeling_dataset().to_csv(data_path, index=False)

    train_all_boosting(
        data_path=data_path,
        metrics_dir=metrics_dir,
        models_dir=models_dir,
        tables_dir=tables_dir,
        max_rows=None,
        permutation_repeats=1,
    )

    perm_reg = pd.read_csv(tables_dir / "permutation_importance_regression.csv")
    assert "evaluation_period" in perm_reg.columns
    assert "importance_split" in perm_reg.columns
    assert set(perm_reg["evaluation_period"].unique()) == {"validation"}
    assert set(perm_reg["importance_split"].unique()) == {"validation"}
    assert perm_reg["evaluation_period"].equals(perm_reg["importance_split"])
    assert not perm_reg["evaluation_period"].isna().any()

    perm_clf = pd.read_csv(tables_dir / "permutation_importance_classification.csv")
    assert "evaluation_period" in perm_clf.columns
    assert "importance_split" in perm_clf.columns
    assert set(perm_clf["evaluation_period"].unique()) == {"validation"}
    assert set(perm_clf["importance_split"].unique()) == {"validation"}
    assert perm_clf["evaluation_period"].equals(perm_clf["importance_split"])
    assert not perm_clf["evaluation_period"].isna().any()



import pytest
pytestmark = pytest.mark.slow
