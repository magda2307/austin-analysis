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
class HorizonDatasetBuildResult:
    """Dataset with horizon-based targets for all intakes (matched + unresolved)."""

    dataset: pd.DataFrame
    observation_end: pd.Timestamp
    horizon_days: tuple[int, ...]


@dataclass(frozen=True)
class DatasetBuildResult:
    """Processed dataset plus simple build diagnostics."""

    dataset: pd.DataFrame
    unresolved_intakes: pd.DataFrame
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
            if col in ["outcome_subtype", "sex_upon_outcome", "age_upon_outcome"]:
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

    match_result = match_intakes_to_future_outcomes(clean_intake_df, clean_outcome_df, extract_end_date=extract_end_date)
    matched = match_result.matched_episodes
    unresolved_intakes = match_result.unresolved_intakes
    unmatched_intakes = match_result.unmatched_intakes
    
    assert len(matched) + unmatched_intakes == len(clean_intake_df), "Intake count conservation failed"
    
    if matched.empty:
        raise ValueError("No valid intake/outcome matches found")

    dataset = add_intake_features(matched)
    if not unresolved_intakes.empty:
        unresolved_intakes = add_intake_features(unresolved_intakes)
        
        pass

    dataset["days_to_outcome"] = (
        dataset["outcome_datetime"] - dataset["intake_datetime"]
    ).dt.total_seconds() / 86400

    outcome_type_normalized = dataset["outcome_type"].fillna("").astype(str).str.strip().str.lower()
    dataset["adopted"] = outcome_type_normalized.eq("adoption")
    dataset["is_adopted"] = dataset["adopted"]
    dataset["target_adopted"] = dataset["adopted"].astype(int)
    dataset["classification_target"] = dataset["adopted"].astype(int)
    dataset["regression_target_days"] = dataset["days_to_outcome"]
    
    dataset["length_of_stay"] = dataset["days_to_outcome"]
    dataset["days_to_adoption"] = np.where(
        dataset["adopted"],
        dataset["days_to_outcome"],
        np.nan,
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
    for df in [dataset] + ([unresolved_intakes] if not unresolved_intakes.empty else []):
        if "is_named" not in df.columns:
            if "name" in df.columns:
                df["is_named"] = df["name"].fillna("").astype(str).str.strip().ne("")
            else:
                df["is_named"] = False
                
        if "has_name" not in df.columns:
            df["has_name"] = df["is_named"]

        if "age_days" not in df.columns:
            if "age_in_days" in df.columns:
                df["age_days"] = df["age_in_days"]
            elif "age_upon_intake" in df.columns:
                from aac_adoption.features.feature_engineering import parse_age_to_days
                df["age_days"] = df["age_upon_intake"].map(parse_age_to_days)
            else:
                df["age_days"] = np.nan

        if "age_in_days" not in df.columns:
            df["age_in_days"] = df["age_days"]

        if "age_months" not in df.columns:
            if "age_in_months" in df.columns:
                df["age_months"] = df["age_in_months"]
            else:
                df["age_months"] = df["age_days"] / 30.0

        if "age_years" not in df.columns:
            if "age_in_years" in df.columns:
                df["age_years"] = df["age_in_years"]
            else:
                df["age_years"] = df["age_days"] / 365.0

        if "age_in_months" not in df.columns:
            df["age_in_months"] = df["age_months"]

        if "age_in_years" not in df.columns:
            df["age_in_years"] = df["age_years"]

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
        "days_to_adoption"
    ]
    dataset = _fill_missing_and_select(dataset, final_columns)
    validate_modeling_dataset(dataset)

    return DatasetBuildResult(
        dataset=dataset,
        unresolved_intakes=unresolved_intakes,
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


def build_horizon_dataset(
    matched_dataset: pd.DataFrame,
    unresolved_intakes: pd.DataFrame,
    extract_end_date: pd.Timestamp,
    horizons: tuple[int, ...] = (7, 30, 60, 90)
) -> HorizonDatasetBuildResult:
    """Build a separate all-intake horizon cohort."""
    combined = pd.concat([matched_dataset, unresolved_intakes], ignore_index=True)
    combined["intake_datetime"] = pd.to_datetime(combined["intake_datetime"], errors="coerce")
    if combined["intake_datetime"].isna().any():
        raise ValueError("horizon cohort contains invalid intake_datetime values")
    if combined["intake_datetime"].gt(extract_end_date).any():
        raise ValueError("horizon cohort contains intakes after extract_end_date")

    combined = combined.sort_values(["animal_id", "intake_datetime"], kind="stable")
    
    combined["next_intake_datetime"] = combined.groupby("animal_id")["intake_datetime"].shift(-1)
    end_date_series = pd.Series(extract_end_date, index=combined.index)
    combined["observation_end_actual"] = pd.concat(
        [combined["next_intake_datetime"], end_date_series],
        axis=1,
    ).min(axis=1)
    
    combined["followup_days_available"] = (combined["observation_end_actual"] - combined["intake_datetime"]).dt.total_seconds() / 86400
    if combined["followup_days_available"].lt(0).any():
        raise ValueError("horizon cohort contains negative observable follow-up")

    for h in horizons:
        is_matched = combined["outcome_datetime"].notna()
        adopted = combined["adopted"].fillna(False)
        outcome_within_h = is_matched & (combined["days_to_outcome"] <= h)
        outcome_before_boundary = (
            combined["next_intake_datetime"].isna()
            | combined["outcome_datetime"].lt(combined["next_intake_datetime"])
        )
        invalid_cross_boundary = is_matched & ~outcome_before_boundary
        if invalid_cross_boundary.any():
            import warnings
            warnings.warn(
                f"Horizon h={h}: {invalid_cross_boundary.sum()} matched episodes have outcome crossing "
                f"a later intake boundary and will be excluded from horizon labels.",
                UserWarning,
                stacklevel=2,
            )
        # Exclude cross-boundary episodes from horizon labels (left as NaN = no label)
        valid_matched = is_matched & outcome_before_boundary

        cond1 = valid_matched & adopted & outcome_within_h
        cond2 = valid_matched & (~adopted) & outcome_within_h
        cond3 = valid_matched & (combined["days_to_outcome"] > h)
        cond4 = (~is_matched) & (combined["followup_days_available"] >= h)

        combined[f"adopted_in_{h}d"] = np.nan
        combined.loc[cond1, f"adopted_in_{h}d"] = 1.0
        combined.loc[cond2 | cond3 | cond4, f"adopted_in_{h}d"] = 0.0

    return HorizonDatasetBuildResult(
        dataset=combined,
        observation_end=extract_end_date,
        horizon_days=horizons
    )


def build_modeling_dataset_from_files(
    intakes_path: str | Path,
    outcomes_path: str | Path,
    output_path: str | Path,
    context_data_dir: str | Path | None = None,
    max_intake_volume_threshold: float | None = None,
    unresolved_out_path: str | Path | None = None,
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

    context_enabled = False
    if context_data_dir is not None:
        ctx_path = Path(context_data_dir)
        if (ctx_path / "austin_weather_daily.csv").exists() and (ctx_path / "austin_311_animal_requests.csv").exists():
            context_enabled = True
            print("Context files found. Context features enabled.")
            dataset = add_context_features_from_dir(dataset, raw_intakes=intakes, context_data_dir=context_data_dir)
            unresolved = result.unresolved_intakes
            if not unresolved.empty:
                unresolved = add_context_features_from_dir(unresolved, raw_intakes=intakes, context_data_dir=context_data_dir)
            result = DatasetBuildResult(
                dataset=dataset,
                unresolved_intakes=unresolved,
                matched_rows=len(dataset),
                unmatched_intakes=result.unmatched_intakes,
            )
        else:
            print("Context files absent. Context features disabled.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    horizon_source_dataset = result.dataset
    horizon_source_unresolved = result.unresolved_intakes

    if extract_end_date is not None:
        horizon_result = build_horizon_dataset(
            horizon_source_dataset,
            horizon_source_unresolved,
            extract_end_date,
        )
        horizon_output = output.parent / "horizon_modeling_dataset.csv"
        horizon_result.dataset.to_csv(horizon_output, index=False)

        horizon_feature_columns = available_intake_features(horizon_result.dataset.columns)
        validate_no_leakage(horizon_feature_columns)
        horizon_target_columns = [
            f"adopted_in_{horizon}d"
            for horizon in horizon_result.horizon_days
        ]

        (output.parent / "horizon_feature_columns.json").write_text(
            json.dumps(horizon_feature_columns, indent=2),
            encoding="utf-8",
        )
        (output.parent / "horizon_target_columns.json").write_text(
            json.dumps(horizon_target_columns, indent=2),
            encoding="utf-8",
        )

    if max_intake_volume_threshold is not None:
        if "intake_volume_7d" in result.dataset.columns:
            dataset = result.dataset
            rows_before = len(dataset)
            dataset = dataset[dataset["intake_volume_7d"] <= max_intake_volume_threshold]
            rows_after = len(dataset)
            rows_removed = rows_before - rows_after
            result = DatasetBuildResult(
                dataset=dataset,
                unresolved_intakes=result.unresolved_intakes,
                matched_rows=rows_after,
                unmatched_intakes=result.unmatched_intakes,
            )
            audit_metadata = {
                "threshold_value": max_intake_volume_threshold,
                "rows_before": rows_before,
                "rows_removed": rows_removed,
                "rows_after": rows_after,
            }
            (output.parent / "volume_threshold_audit.json").write_text(
                json.dumps(audit_metadata, indent=2),
                encoding="utf-8",
            )

    result.dataset.to_csv(output, index=False)
    
    if unresolved_out_path is not None:
        unresolved_output = Path(unresolved_out_path)
    else:
        unresolved_output = output.parent / "unresolved_intakes.csv"
    unresolved_output.parent.mkdir(parents=True, exist_ok=True)
    result.unresolved_intakes.to_csv(unresolved_output, index=False)

    feature_columns = available_intake_features(result.dataset.columns)
    validate_no_leakage(feature_columns)
    target_columns = [column for column in TARGET_COLUMNS if column in result.dataset.columns]

    if context_enabled:
        context_columns = [column for column in CONTEXT_FEATURES if column in result.dataset.columns]
        # Write separate context-only file for backward compatibility
        (output.parent / "context_feature_columns.json").write_text(
            json.dumps(context_columns, indent=2),
            encoding="utf-8",
        )
        # Merge: feature_columns.json now contains base + context so any consumer
        # reading only this file gets the complete picture for a context-enabled run.
        # Preserve order: base features first, then context features appended.
        seen = set(feature_columns)
        combined_feature_columns = feature_columns + [c for c in context_columns if c not in seen]
        validate_no_leakage(combined_feature_columns)
        (output.parent / "feature_columns.json").write_text(
            json.dumps(combined_feature_columns, indent=2),
            encoding="utf-8",
        )
        (output.parent / "context_metadata.json").write_text(
            json.dumps({
                "context_weather_lag_days": 1,
                "context_enabled": True,
                "context_feature_count": len(context_columns),
                "context_features": context_columns,
            }, indent=2),
            encoding="utf-8",
        )
    else:
        # No context: write base-only feature_columns.json
        (output.parent / "feature_columns.json").write_text(
            json.dumps(feature_columns, indent=2),
            encoding="utf-8",
        )
        (output.parent / "context_metadata.json").write_text(
            json.dumps({"context_enabled": False}),
            encoding="utf-8",
        )

    (output.parent / "target_columns.json").write_text(
        json.dumps(target_columns, indent=2),
        encoding="utf-8",
    )
    return result
