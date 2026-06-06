"""Tests for horizon-based adoption targets."""

import pandas as pd
import numpy as np

from aac_adoption.data.build_dataset import build_modeling_dataset
from scripts.generate_data_audit import build_horizon_followup_audit

def test_horizon_targets():
    """Verify adopted_in_7d, 30d, 60d, 90d are correctly created."""
    intakes = pd.DataFrame({
        "animal_id": ["A1", "A2", "A3"],
        "name": ["Fido", "Whiskers", "Rex"],
        "animal_type": ["Dog", "Cat", "Dog"],
        "intake_datetime": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-01"]),
        "intake_type": ["Stray", "Owner Surrender", "Stray"],
        "intake_condition": ["Normal", "Normal", "Injured"],
        "sex_upon_intake": ["Intact Male", "Intact Female", "Neutered Male"],
        "age_upon_intake": ["2 years", "3 months", "4 years"],
        "breed": ["Labrador", "Domestic Shorthair", "Pit Bull"],
        "color": ["Black", "Orange", "White"],
        "found_location": ["Austin (TX)", "Travis (TX)", "Unknown"],
    })

    outcomes = pd.DataFrame({
        "animal_id": ["A1", "A2", "A3"],
        "animal_type": ["Dog", "Cat", "Dog"],
        "outcome_datetime": pd.to_datetime(["2020-01-05", "2020-01-15", "2020-05-01"]),
        "outcome_type": ["Adoption", "Adoption", "Transfer"],
        "outcome_subtype": ["Foster", "Offsite", "Partner"],
        "sex_upon_outcome": ["Neutered Male", "Spayed Female", "Neutered Male"],
        "age_upon_outcome": ["2 years", "3 months", "4 years"],
    })

    # A1: Adopted in 4 days -> True for all horizons
    # A2: Adopted in 14 days -> False for 7d, True for 30, 60, 90
    # A3: Transferred in 121 days -> False for all

    result = build_modeling_dataset(intakes, outcomes).dataset

    assert "adopted_in_7d" in result.columns
    assert "adopted_in_30d" in result.columns
    assert "adopted_in_60d" in result.columns
    assert "adopted_in_90d" in result.columns

    a1 = result[result["animal_id"] == "A1"].iloc[0]
    assert a1["adopted_in_7d"] == True
    assert a1["adopted_in_30d"] == True
    assert a1["adopted_in_90d"] == True

    a2 = result[result["animal_id"] == "A2"].iloc[0]
    assert a2["adopted_in_7d"] == False
    assert a2["adopted_in_30d"] == True
    assert a2["adopted_in_90d"] == True

    a3 = result[result["animal_id"] == "A3"].iloc[0]
    assert a3["adopted_in_7d"] == False
    assert a3["adopted_in_30d"] == False
    assert a3["adopted_in_90d"] == False


def test_horizon_targets_preserve_followup_and_censor_late_tail():
    intakes = pd.DataFrame({
        "animal_id": ["A1", "A2"],
        "name": ["Fast", "Late"],
        "animal_type": ["Dog", "Dog"],
        "intake_datetime": pd.to_datetime(["2020-01-01", "2020-03-25"]),
        "intake_type": ["Stray", "Stray"],
        "intake_condition": ["Normal", "Normal"],
        "sex_upon_intake": ["Intact Male", "Intact Male"],
        "age_upon_intake": ["2 years", "2 years"],
        "breed": ["Labrador", "Labrador"],
        "color": ["Black", "Brown"],
        "found_location": ["Austin (TX)", "Austin (TX)"],
    })
    outcomes = pd.DataFrame({
        "animal_id": ["A1", "A2"],
        "animal_type": ["Dog", "Dog"],
        "outcome_datetime": pd.to_datetime(["2020-01-05", "2020-03-28"]),
        "outcome_type": ["Adoption", "Transfer"],
        "outcome_subtype": ["Foster", "Partner"],
        "sex_upon_outcome": ["Neutered Male", "Neutered Male"],
        "age_upon_outcome": ["2 years", "2 years"],
    })

    result = build_modeling_dataset(
        intakes,
        outcomes,
        extract_end_date=pd.Timestamp("2020-03-31"),
    ).dataset

    assert "followup_days_available" in result.columns
    late = result[result["animal_id"] == "A2"].iloc[0]
    assert np.isnan(late["adopted_in_7d"])
    assert np.isnan(late["adopted_in_30d"])
    assert np.isnan(late["adopted_in_90d"])

    fast = result[result["animal_id"] == "A1"].iloc[0]
    assert fast["adopted_in_90d"] == True


def test_horizon_followup_audit_counts_included_and_censored(tmp_path):
    df = pd.DataFrame({
        "animal_id": ["A1", "A2", "A3"],
        "intake_datetime": pd.to_datetime(["2020-01-01", "2020-03-25", "2020-03-29"]),
        "outcome_datetime": pd.to_datetime(["2020-01-05", "2020-03-28", "2020-03-30"]),
        "adopted": [True, False, True],
        "days_to_outcome": [4.0, 3.0, 1.0],
        "followup_days_available": [90.0, 6.0, 2.0],
        "adopted_in_7d": [1.0, np.nan, 1.0],
        "adopted_in_30d": [1.0, np.nan, 1.0],
        "adopted_in_60d": [1.0, np.nan, 1.0],
        "adopted_in_90d": [1.0, np.nan, 1.0],
    })
    path = tmp_path / "modeling_dataset.csv"
    df.to_csv(path, index=False)

    audit = build_horizon_followup_audit(str(path))
    h7 = audit[audit["horizon_days"] == 7].iloc[0]
    assert h7["included_rows"] == 2
    assert h7["censored_rows"] == 1
    assert h7["excluded_rows"] == 1
    assert h7["fast_adopted_short_followup_rows"] == 1
