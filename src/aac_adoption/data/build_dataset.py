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
from aac_adoption.features.feature_engineering import add_intake_features, winsorize_outliers
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
class DatasetBuildResult:
    """Processed dataset plus simple build diagnostics."""

    dataset: pd.DataFrame
    matched_rows: int
    unmatched_intakes: int


def _keep_existing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[column for column in columns if column in df.columns]].copy()


def build_modeling_dataset(intakes: pd.DataFrame, outcomes: pd.DataFrame) -> DatasetBuildResult:
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
    dataset["length_of_stay"] = dataset["days_to_outcome"]
    dataset["days_to_adoption"] = np.where(
        dataset["adopted"],
        dataset["days_to_outcome"],
        np.nan,
    )
    
    # Horizon-based targets
    for horizon in [7, 30, 60, 90]:
        dataset[f"adopted_in_{horizon}d"] = dataset["adopted"] & (dataset["days_to_outcome"] <= horizon)
    # Only winsorize if there are extreme outliers (>99th percentile > 99th percentile of non-extreme)
    if len(dataset) > 100:
        q99 = dataset["length_of_stay"].quantile(0.99)
        q01 = dataset["length_of_stay"].quantile(0.01)
        if (dataset["length_of_stay"] > q99).sum() > 0:
            dataset["length_of_stay"] = winsorize_outliers(dataset["length_of_stay"], 0.01, 0.99)
            dataset["days_to_outcome"] = dataset["length_of_stay"]
            dataset["regression_target_days"] = dataset["length_of_stay"]

    ordered_columns = [
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
        "adopted_in_7d",
        "adopted_in_30d",
        "adopted_in_60d",
        "adopted_in_90d",
    ]
    dataset = dataset[[column for column in ordered_columns if column in dataset.columns]]
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


def build_modeling_dataset_from_files(
    intakes_path: str | Path,
    outcomes_path: str | Path,
    output_path: str | Path,
    context_data_dir: str | Path | None = None,
    max_intake_volume_threshold: float | None = 100.0,
) -> DatasetBuildResult:
    """Load raw CSVs, build modeling dataset, and write processed CSV."""
    result = build_modeling_dataset(load_intakes(intakes_path), load_outcomes(outcomes_path))
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
