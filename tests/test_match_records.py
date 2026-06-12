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
    
    match_result = match_intakes_to_future_outcomes(
        intakes, 
        outcomes,
        extract_end_date=pd.Timestamp("2021-03-31 10:00:00")
    )
    result = match_result.matched_episodes
    
    assert len(result) == 2
    assert match_result.unmatched_intakes == 0
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
    
    match_result = match_intakes_to_future_outcomes(
        intakes, 
        outcomes,
        extract_end_date=pd.Timestamp("2021-03-31 10:00:00")
    )
    result = match_result.matched_episodes
    
    assert len(result) == 2
    assert match_result.unmatched_intakes == 0
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
    
    match_result = match_intakes_to_future_outcomes(
        intakes,
        outcomes,
        extract_end_date=pd.Timestamp("2021-03-31 10:00:00"),
    )
    
    assert match_result.unmatched_intakes == 1
    assert len(match_result.matched_episodes) == 2
    assert match_result.unresolved_intakes["animal_id"].tolist() == ["A3"]
    assert match_result.unresolved_intakes["followup_days_available"].tolist() == [30]


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


def test_outcome_at_next_intake_boundary_is_not_assigned_to_prior_episode():
    intakes = pd.DataFrame(
        {
            "animal_id": ["A1", "A1"],
            "intake_datetime": pd.to_datetime(["2024-01-01", "2024-01-05"]),
            "animal_type": ["Dog", "Dog"],
        }
    )
    outcomes = pd.DataFrame(
        {
            "animal_id": ["A1"],
            "outcome_datetime": pd.to_datetime(["2024-01-05"]),
            "outcome_type": ["Adoption"],
        }
    )

    result = match_intakes_to_future_outcomes(
        intakes,
        outcomes,
        extract_end_date=pd.Timestamp("2024-01-10"),
    )

    assert result.matched_episodes["intake_datetime"].tolist() == [pd.Timestamp("2024-01-05")]
    assert result.unresolved_intakes["intake_datetime"].tolist() == [pd.Timestamp("2024-01-01")]
    assert len(result.matched_episodes["outcome_datetime"].unique()) == 1


def test_matcher_rejects_extract_end_before_unresolved_intake():
    intakes = pd.DataFrame(
        {
            "animal_id": ["A1"],
            "intake_datetime": pd.to_datetime(["2024-01-10"]),
            "animal_type": ["Dog"],
        }
    )
    outcomes = pd.DataFrame(columns=["animal_id", "outcome_datetime", "outcome_type"])

    with pytest.raises(ValueError, match="extract_end_date"):
        match_intakes_to_future_outcomes(
            intakes,
            outcomes,
            extract_end_date=pd.Timestamp("2024-01-05"),
        )


def test_outcome_after_extract_end_remains_unresolved():
    intakes = pd.DataFrame(
        {
            "animal_id": ["A1"],
            "intake_datetime": pd.to_datetime(["2024-01-01"]),
            "animal_type": ["Dog"],
        }
    )
    outcomes = pd.DataFrame(
        {
            "animal_id": ["A1"],
            "outcome_datetime": pd.to_datetime(["2024-02-01"]),
            "outcome_type": ["Adoption"],
        }
    )

    result = match_intakes_to_future_outcomes(
        intakes,
        outcomes,
        extract_end_date=pd.Timestamp("2024-01-15"),
    )

    assert result.matched_episodes.empty
    assert result.unresolved_intakes["observation_end"].tolist() == [pd.Timestamp("2024-01-15")]
    assert result.unresolved_intakes["followup_days_available"].tolist() == [14]


def test_unresolved_followup_is_truncated_at_next_intake():
    intakes = pd.DataFrame(
        {
            "animal_id": ["A1", "A1"],
            "intake_datetime": pd.to_datetime(["2024-01-01", "2024-01-10"]),
            "animal_type": ["Dog", "Dog"],
        }
    )
    outcomes = pd.DataFrame(
        {
            "animal_id": ["A1"],
            "outcome_datetime": pd.to_datetime(["2024-01-12"]),
            "outcome_type": ["Adoption"],
        }
    )

    result = match_intakes_to_future_outcomes(
        intakes,
        outcomes,
        extract_end_date=pd.Timestamp("2024-02-01"),
    )

    unresolved = result.unresolved_intakes.iloc[0]
    assert unresolved["intake_datetime"] == pd.Timestamp("2024-01-01")
    assert unresolved["observation_end"] == pd.Timestamp("2024-01-10")
    assert unresolved["followup_days_available"] == 9


def test_duplicate_prior_outcomes_do_not_inflate_episode_number():
    intakes = pd.DataFrame(
        {
            "animal_id": ["A1", "A1"],
            "intake_datetime": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "animal_type": ["Dog", "Dog"],
        }
    )
    outcomes = pd.DataFrame(
        {
            "animal_id": ["A1", "A1", "A1"],
            "outcome_datetime": pd.to_datetime(["2024-01-05", "2024-01-05", "2024-02-05"]),
            "outcome_type": ["Transfer", "Transfer", "Adoption"],
        }
    )

    episodes = verify_reintake_patterns(intakes, outcomes)

    assert episodes["episode_number"].tolist() == [1, 2]
