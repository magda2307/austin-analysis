import pandas as pd

from aac_adoption.dashboard.data import best_model_rows, build_prediction_record


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
