"""Tests for intake/outcome matching with re-intake detection."""

import pandas as pd
import pytest

from aac_adoption.data.match_records import (
    match_intakes_to_future_outcomes,
    verify_reintake_patterns,
)


def test_match_records_detects_reintakes():
    intakes = pd.DataFrame({
        "animal_id": ["A1", "A1", "A2"],
        "intake_datetime": pd.to_datetime([
            "2021-01-01 10:00:00",
            "2021-02-01 10:00:00",
            "2021-01-15 10:00:00",
        ]),
        "animal_type": ["Dog", "Dog", "Cat"],
    })
    
    outcomes = pd.DataFrame({
        "animal_id": ["A1", "A1", "A2"],
        "outcome_datetime": pd.to_datetime([
            "2021-01-10 10:00:00",
            "2021-02-10 10:00:00",
            "2021-01-20 10:00:00",
        ]),
        "outcome_type": ["Adoption", "Adoption", "Transfer"],
    })
    
    episodes = verify_reintake_patterns(intakes, outcomes)
    
    assert len(episodes) == 3
    assert int(episodes[episodes["animal_id"] == "A1"]["episode_number"].iloc[1]) == 2


def test_match_records_reintake_features():
    intakes = pd.DataFrame({
        "animal_id": ["A1", "A1"],
        "intake_datetime": pd.to_datetime([
            "2021-01-01 10:00:00",
            "2021-02-01 10:00:00",
        ]),
        "animal_type": ["Dog", "Dog"],
    })
    
    outcomes = pd.DataFrame({
        "animal_id": ["A1", "A1"],
        "outcome_datetime": pd.to_datetime([
            "2021-01-10 10:00:00",
            "2021-02-10 10:00:00",
        ]),
        "outcome_type": ["Return to Owner", "Adoption"],
    })
    
    result, unmatched = match_intakes_to_future_outcomes(intakes, outcomes)
    
    assert len(result) == 2
    assert "is_reintake" in result.columns
    assert "episode_number" in result.columns
    assert bool(result["is_reintake"].iloc[1]) is True
    assert result["episode_number"].iloc[1] == 2


def test_match_records_no_reintake():
    intakes = pd.DataFrame({
        "animal_id": ["A1", "A2"],
        "intake_datetime": pd.to_datetime([
            "2021-01-01 10:00:00",
            "2021-01-15 10:00:00",
        ]),
        "animal_type": ["Dog", "Cat"],
    })
    
    outcomes = pd.DataFrame({
        "animal_id": ["A1", "A2"],
        "outcome_datetime": pd.to_datetime([
            "2021-01-10 10:00:00",
            "2021-01-20 10:00:00",
        ]),
        "outcome_type": ["Adoption", "Adoption"],
    })
    
    result, unmatched = match_intakes_to_future_outcomes(intakes, outcomes)
    
    assert len(result) == 2
    assert result["is_reintake"].sum() == 0
    assert result["episode_number"].tolist() == [1, 1]


def test_match_records_unmatched_intakes():
    intakes = pd.DataFrame({
        "animal_id": ["A1", "A2", "A3"],
        "intake_datetime": pd.to_datetime([
            "2021-01-01 10:00:00",
            "2021-01-15 10:00:00",
            "2021-03-01 10:00:00",
        ]),
        "animal_type": ["Dog", "Cat", "Dog"],
    })
    
    outcomes = pd.DataFrame({
        "animal_id": ["A1", "A2"],
        "outcome_datetime": pd.to_datetime([
            "2021-01-10 10:00:00",
            "2021-01-20 10:00:00",
        ]),
        "outcome_type": ["Adoption", "Adoption"],
    })
    
    result, unmatched = match_intakes_to_future_outcomes(intakes, outcomes)
    
    assert unmatched == 1
    assert len(result) == 2


def test_match_records_reintake_with_days_since():
    intakes = pd.DataFrame({
        "animal_id": ["A1", "A1"],
        "intake_datetime": pd.to_datetime([
            "2021-01-01 10:00:00",
            "2021-02-01 10:00:00",
        ]),
        "animal_type": ["Dog", "Dog"],
    })
    
    outcomes = pd.DataFrame({
        "animal_id": ["A1", "A1"],
        "outcome_datetime": pd.to_datetime([
            "2021-01-10 10:00:00",
            "2021-02-10 10:00:00",
        ]),
        "outcome_type": ["Return to Owner", "Adoption"],
    })
    
    episodes = verify_reintake_patterns(intakes, outcomes)
    reintake_info = episodes[episodes["is_reintake"]].iloc[0]
    
    assert reintake_info["days_since_last_stay"] == 22
