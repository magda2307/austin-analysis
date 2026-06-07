"""Build modeling dataset from AAC intake and outcome records."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aac_adoption.data.clean_data import clean_intakes, clean_outcomes
from aac_adoption.data.context_data import CONTEXT_FEATURES, add_context_features_from_dir
from aac_adoption.data.load_data import load_intakes, load_outcomes
from aac_adoption.data.match_records import match_intakes_to_future_outcomes
from aac_adoption.features.feature_engineering import add_intake_features
from aac_adoption.features.feature_sets import (
    TARGET_COLUMNS,
    available_intake_features,
    validate_no_leakage,
)


REQUIRED_MODELING_COLUMNS = {
    "animal_id",
    "animal_type",
    "intake_datetime",
    "outcome_datetime",
    "outcome_type",
    "days_to_outcome",
    "adopted",
    "classification_target",
    "regression_target_days",
    "is_censored",
    "censoring_reason",
    "event_type",
    "followup_days_censored",
}

INTAKE_COLUMNS_TO_KEEP = [
    "animal_id",
    "name",
    "animal_type",
    "intake_datetime",
    "intake_type",
    "intake_condition",
    "sex_upon_intake",
    "age_upon_intake",
    "breed",
    "color",
    "found_location",
]

OUTCOME_COLUMNS_TO_KEEP = [
    "animal_id",
    "outcome_datetime",
    "outcome_type",
    "outcome_subtype",
    "sex_upon_outcome",
    "age_upon_outcome",
]


@dataclass(frozen=True)
class DatasetBuildResult:
    """Processed dataset plus simple build diagnostics."""

    dataset: pd.DataFrame
    matched_rows: int
    unmatched_intakes: int


def _keep_existing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[column for column in columns if column in df.columns]].copy()


def _fill_missing_and_select(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Safely select columns, filling any missing expected columns with nulls or safe defaults."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            # Set safe defaults/nulls for missing expected columns
            if col in ["is_censored"]:
                df[col] = False
            elif col in ["censoring_reason"]:
                df[col] = ""
            elif col in ["event_type"]:
                if "outcome_type" in df.columns:
                    df[col] = df["outcome_type"].fillna("").astype(str).str.strip().str.lower()
                else:
                    df[col] = ""
            elif col in ["followup_days_censored"]:
                df[col] = df.get("days_to_outcome", np.nan)
            elif col in ["outcome_subtype", "sex_upon_outcome", "age_upon_outcome"]:
                df[col] = pd.Series(pd.NA, index=df.index, dtype="object")
            elif col in ["adopted", "is_adopted"]:
                df[col] = False
            elif col in ["target_adopted", "classification_target"]:
                df[col] = 0
            elif col in [
                "is_austin_found_location",
                "is_outside_jurisdiction",
                "is_intersection_location",
                "is_address_like_location",
                "is_airport_location",
                "is_black_or_dark",
                "is_mixed_breed",
                "has_name",
                "is_named"
            ]:
                df[col] = False
            else:
                df[col] = np.nan
    return df[columns]


def build_modeling_dataset(intakes: pd.DataFrame, outcomes: pd.DataFrame, extract_end_date: pd.Timestamp | None = None) -> DatasetBuildResult:
    """Clean, join, feature-engineer, and validate AAC modeling dataset."""
    clean_intake_df = _keep_existing_columns(clean_intakes(intakes), INTAKE_COLUMNS_TO_KEEP)
    clean_outcome_df = _keep_existing_columns(clean_outcomes(outcomes), OUTCOME_COLUMNS_TO_KEEP)

    matched, unmatched_intakes = match_intakes_to_future_outcomes(clean_intake_df, clean_outcome_df)
    if matched.empty:
        raise ValueError("No valid intake/outcome matches found")

    dataset = add_intake_features(matched)
    dataset["days_to_outcome"] = (
        dataset["outcome_datetime"] - dataset["intake_datetime"]
    ).dt.total_seconds() / 86400

    outcome_type_normalized = dataset["outcome_type"].fillna("").astype(str).str.strip().str.lower()
    dataset["adopted"] = outcome_type_normalized.eq("adoption")
    dataset["is_adopted"] = dataset["adopted"]
    dataset["target_adopted"] = dataset["adopted"].astype(int)
    dataset["classification_target"] = dataset["adopted"].astype(int)
    dataset["regression_target_days"] = dataset["days_to_outcome"]
    
    dataset["is_censored"] = False
    dataset["censoring_reason"] = ""
    dataset["event_type"] = outcome_type_normalized
    dataset["followup_days_censored"] = dataset["days_to_outcome"]
    dataset["length_of_stay"] = dataset["days_to_outcome"]
    dataset["days_to_adoption"] = np.where(
        dataset["adopted"],
        dataset["days_to_outcome"],
        np.nan,
    )
    
    # Horizon-based targets and right-censoring safeguards
    max_date = extract_end_date if extract_end_date is not None else max(dataset["intake_datetime"].max(), dataset["outcome_datetime"].max())
    dataset["followup_days_available"] = (max_date - dataset["intake_datetime"]).dt.total_seconds() / 86400

    for horizon in [7, 30, 60, 90]:
        has_full_followup = dataset["followup_days_available"] >= horizon
        adopted_in_horizon = dataset["adopted"] & (dataset["days_to_outcome"] <= horizon)
        
        dataset[f"adopted_in_{horizon}d"] = np.where(
            has_full_followup | adopted_in_horizon,
            np.where(adopted_in_horizon, 1.0, 0.0),
            np.nan
        )
    
    required_columns = [
        "animal_id",
        "animal_type",
        "intake_datetime",
        "outcome_datetime",
        "outcome_type",
        "days_to_outcome",
        "adopted",
        "classification_target",
        "regression_target_days",
    ]
    missing = set(required_columns) - set(dataset.columns)
    if missing:
        raise ValueError(f"Missing required columns after dataset build: {missing}")
    
    # Normalize aliases before final selection (Rule 7)
    if "is_named" not in dataset.columns:
        if "name" in dataset.columns:
            dataset["is_named"] = dataset["name"].fillna("").astype(str).str.strip().ne("")
        else:
            dataset["is_named"] = False
            
    if "has_name" not in dataset.columns:
        dataset["has_name"] = dataset["is_named"]

    if "age_days" not in dataset.columns:
        if "age_in_days" in dataset.columns:
            dataset["age_days"] = dataset["age_in_days"]
        elif "age_upon_intake" in dataset.columns:
            from aac_adoption.features.feature_engineering import parse_age_to_days
            dataset["age_days"] = dataset["age_upon_intake"].map(parse_age_to_days)
        else:
            dataset["age_days"] = np.nan

    if "age_in_days" not in dataset.columns:
        dataset["age_in_days"] = dataset["age_days"]

    if "age_months" not in dataset.columns:
        if "age_in_months" in dataset.columns:
            dataset["age_months"] = dataset["age_in_months"]
        else:
            dataset["age_months"] = dataset["age_days"] / 30.0

    if "age_years" not in dataset.columns:
        if "age_in_years" in dataset.columns:
            dataset["age_years"] = dataset["age_in_years"]
        else:
            dataset["age_years"] = dataset["age_days"] / 365.0

    if "age_in_months" not in dataset.columns:
        dataset["age_in_months"] = dataset["age_months"]

    if "age_in_years" not in dataset.columns:
        dataset["age_in_years"] = dataset["age_years"]

    # Optional outcome metadata (Rule 8)
    for col in ["outcome_subtype", "sex_upon_outcome", "age_upon_outcome"]:
        if col not in dataset.columns:
            dataset[col] = pd.Series(pd.NA, index=dataset.index, dtype="object")

    final_columns = [
        "animal_id",
        "animal_type",
        "intake_datetime",
        "outcome_datetime",
        "intake_type",
        "intake_condition",
        "outcome_type",
        "outcome_subtype",
        "sex_upon_intake",
        "sex_upon_outcome",
        "age_upon_intake",
        "age_upon_outcome",
        "breed",
        "color",
        "found_location_kind",
        "found_location_area",
        "is_austin_found_location",
        "is_outside_jurisdiction",
        "is_intersection_location",
        "is_address_like_location",
        "is_airport_location",
        "has_name",
        "is_named",
        "age_in_days",
        "age_in_months",
        "age_in_years",
        "age_days",
        "age_months",
        "age_years",
        "age_group",
        "intake_year",
        "intake_month",
        "intake_quarter",
        "intake_season",
        "covid_period",
        "color_group",
        "primary_color",
        "simplified_color_group",
        "is_black_or_dark",
        "primary_breed",
        "is_mixed_breed",
        "simplified_breed_group",
        "days_to_outcome",
        "length_of_stay",
        "adopted",
        "is_adopted",
        "target_adopted",
        "classification_target",
        "regression_target_days",
        "days_to_adoption",
        "followup_days_available",
        "adopted_in_7d",
        "adopted_in_30d",
        "adopted_in_60d",
        "adopted_in_90d",
        "is_censored",
        "censoring_reason",
        "event_type",
        "followup_days_censored"
    ]
    dataset = _fill_missing_and_select(dataset, final_columns)
    validate_modeling_dataset(dataset)

    return DatasetBuildResult(
        dataset=dataset,
        matched_rows=len(dataset),
        unmatched_intakes=unmatched_intakes,
    )


def validate_modeling_dataset(dataset: pd.DataFrame) -> None:
    """Run basic integrity checks for the first ML dataset."""
    missing = sorted(REQUIRED_MODELING_COLUMNS - set(dataset.columns))
    if missing:
        raise ValueError(f"modeling dataset missing required columns: {missing}")

    animal_types = set(dataset["animal_type"].astype(str).str.lower().unique())
    unsupported = animal_types - {"dog", "cat"}
    if unsupported:
        raise ValueError(f"unsupported animal types found: {sorted(unsupported)}")

    if dataset["days_to_outcome"].lt(0).any():
        raise ValueError("negative days_to_outcome values found")

    if not dataset["classification_target"].isin([0, 1]).all():
        raise ValueError("classification_target must contain only 0/1")

    target_mismatch = dataset["classification_target"].ne(dataset["adopted"].astype(int)).any()
    if target_mismatch:
        raise ValueError("classification_target does not match adopted flag")

    required_censoring_cols = ["is_censored", "censoring_reason", "event_type", "followup_days_censored"]
    missing_censoring = set(required_censoring_cols) - set(dataset.columns)
    if missing_censoring:
        raise ValueError(f"Missing censoring columns: {missing_censoring}")


def build_modeling_dataset_from_files(
    intakes_path: str | Path,
    outcomes_path: str | Path,
    output_path: str | Path,
    context_data_dir: str | Path | None = None,
    max_intake_volume_threshold: float | None = 100.0,
) -> DatasetBuildResult:
    """Load raw CSVs, build modeling dataset, and write processed CSV."""
    intakes = load_intakes(intakes_path)
    outcomes = load_outcomes(outcomes_path)
    extract_dates = []
    if "intake_datetime" in intakes.columns:
        extract_dates.append(pd.to_datetime(intakes["intake_datetime"], errors="coerce", utc=True).dt.tz_localize(None).max())
    if "outcome_datetime" in outcomes.columns:
        extract_dates.append(pd.to_datetime(outcomes["outcome_datetime"], errors="coerce", utc=True).dt.tz_localize(None).max())
    extract_dates = [date for date in extract_dates if pd.notna(date)]
    extract_end_date = max(extract_dates) if extract_dates else None

    result = build_modeling_dataset(intakes, outcomes, extract_end_date=extract_end_date)
    dataset = result.dataset
    if max_intake_volume_threshold is not None:
        if "intake_volume_7d" in dataset.columns:
            dataset = dataset[dataset["intake_volume_7d"] <= max_intake_volume_threshold]
            result = DatasetBuildResult(
                dataset=dataset,
                matched_rows=len(dataset),
                unmatched_intakes=result.unmatched_intakes,
            )
    if context_data_dir is not None:
        dataset = add_context_features_from_dir(dataset, context_data_dir)
        result = DatasetBuildResult(
            dataset=dataset,
            matched_rows=result.matched_rows,
            unmatched_intakes=result.unmatched_intakes,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.dataset.to_csv(output, index=False)

    feature_columns = available_intake_features(result.dataset.columns)
    validate_no_leakage(feature_columns)
    target_columns = [column for column in TARGET_COLUMNS if column in result.dataset.columns]

    (output.parent / "feature_columns.json").write_text(
        json.dumps(feature_columns, indent=2),
        encoding="utf-8",
    )
    (output.parent / "target_columns.json").write_text(
        json.dumps(target_columns, indent=2),
        encoding="utf-8",
    )
    if context_data_dir is not None:
        context_columns = [column for column in CONTEXT_FEATURES if column in result.dataset.columns]
        (output.parent / "context_feature_columns.json").write_text(
            json.dumps(context_columns, indent=2),
            encoding="utf-8",
        )
    return result
