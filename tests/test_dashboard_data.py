import pandas as pd
import pytest

from aac_adoption.dashboard.data import (
    best_model_rows,
    build_prediction_record,
    build_profile_prediction_record,
    profile_global_shap_reasons,
    similar_historical_cases,
    visibility_need_from_prediction,
)


def test_best_model_rows_selects_expected_metrics():
    classification = pd.DataFrame(
        [
            {"animal_subset": "combined", "model_name": "logistic", "roc_auc": 0.7},
            {"animal_subset": "combined", "model_name": "boosting", "roc_auc": 0.8},
        ]
    )
    regression = pd.DataFrame(
        [
            {"animal_subset": "combined", "model_name": "ridge", "mae": 30.0},
            {"animal_subset": "combined", "model_name": "boosting", "mae": 20.0},
        ]
    )

    result = best_model_rows(classification, regression)

    assert len(result) == 2
    assert set(result["model_name"]) == {"boosting"}
    assert set(result["primary_metric"]) == {"roc_auc", "mae"}


def test_build_prediction_record_creates_model_features():
    record = build_prediction_record(
        animal_type="Dog",
        intake_type="Stray",
        intake_condition="Normal",
        sex_upon_intake="Intact Male",
        age_days=365.25 * 2,
        breed="Labrador Retriever Mix",
        color="Black/White",
        has_name=True,
        intake_date=pd.Timestamp("2024-06-01"),
    )

    assert record.loc[0, "age_group"] == "young"
    assert record.loc[0, "simplified_breed_group"] == "retriever_type"
    assert record.loc[0, "simplified_color_group"] == "black_or_dark"
    assert record.loc[0, "covid_period"] == "post_covid"
    assert record.loc[0, "intake_season"] == "summer"


def test_build_profile_prediction_record_uses_representative_values():
    profile = pd.Series(
        {
            "animal_type": "Dog",
            "age_group": "senior",
            "intake_type": "Owner Surrender",
            "intake_condition": "Normal",
            "sex_upon_intake": "Neutered Male",
            "simplified_breed_group": "pit_bull_type",
            "simplified_color_group": "brown_tan",
            "is_named": True,
        }
    )

    record = build_profile_prediction_record(profile)

    assert record.loc[0, "age_group"] == "senior"
    assert record.loc[0, "simplified_breed_group"] == "pit_bull_type"
    assert record.loc[0, "simplified_color_group"] == "brown_tan"
    assert bool(record.loc[0, "is_named"]) is True


def test_similar_historical_cases_returns_outcome_mix(tmp_path):
    data = pd.DataFrame(
        {
            "animal_type": ["Dog", "Dog", "Dog"],
            "age_group": ["senior", "senior", "senior"],
            "intake_type": ["Stray", "Stray", "Stray"],
            "intake_condition": ["Normal", "Normal", "Normal"],
            "simplified_breed_group": ["pit_bull_type", "pit_bull_type", "pit_bull_type"],
            "simplified_color_group": ["brown_tan", "brown_tan", "brown_tan"],
            "sex_upon_intake": ["Neutered Male", "Neutered Male", "Neutered Male"],
            "is_named": [True, True, True],
            "classification_target": [1, 0, 0],
            "days_to_outcome": [10.0, 20.0, 30.0],
            "outcome_type": ["Adoption", "Transfer", "Euthanasia"],
        }
    )
    path = tmp_path / "modeling.csv"
    data.to_csv(path, index=False)
    record = build_prediction_record(
        animal_type="Dog",
        intake_type="Stray",
        intake_condition="Normal",
        sex_upon_intake="Neutered Male",
        age_days=365.25 * 10,
        breed="Pit Bull Mix",
        color="Brown/White",
        has_name=True,
        intake_date=pd.Timestamp("2024-06-01"),
    )

    similar = similar_historical_cases(path, record)

    assert similar.loc[0, "similar_records"] == 3
    assert similar.loc[0, "historical_adoption_rate_pct"] == pytest.approx(100 / 3)
    assert similar.loc[0, "median_days_to_outcome"] == 20.0
    assert similar.loc[0, "adoption_rate_pct"] == pytest.approx(100 / 3)
    assert similar.loc[0, "transfer_rate_pct"] == pytest.approx(100 / 3)
    assert similar.loc[0, "euthanasia_rate_pct"] == pytest.approx(100 / 3)
    assert similar.loc[0, "matching_level"] == "exact visible profile"


def test_profile_global_shap_reasons_maps_profile_values():
    profile = pd.Series(
        {
            "age_group": "baby",
            "intake_type": "Stray",
            "simplified_breed_group": "domestic_cat",
            "is_named": False,
        }
    )
    shap_global = pd.DataFrame(
        {
            "feature": ["age_upon_intake", "intake_type", "surprise"],
            "mean_abs_shap": [0.5, 0.4, 0.9],
        }
    )

    reasons = profile_global_shap_reasons(profile, shap_global, top_n=2)

    assert list(reasons["feature"]) == ["age_upon_intake", "intake_type"]
    assert "profile_value" in reasons.columns


def test_visibility_need_from_prediction_labels_quadrants():
    assert visibility_need_from_prediction(0.7, 5) == "quick placement likely"
    assert visibility_need_from_prediction(0.7, 20) == "needs visibility"
    assert visibility_need_from_prediction(0.2, 20) == "long-stay risk"
