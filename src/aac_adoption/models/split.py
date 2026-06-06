"""Train/validation/test splitting for thesis experiments."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from aac_adoption.config import RANDOM_STATE


TRAIN_PERIOD = "2013-2021"
VALIDATION_PERIOD = "2022-2023"
TEST_PERIOD = "2024-2025"


@dataclass(frozen=True)
class DatasetSplit:
    """Container for split frames and reporting metadata."""

    full_data: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    strategy: str
    train_period: str
    validation_period: str
    test_period: str
    animal_subset: str


def _filter_subset(df: pd.DataFrame, animal_subset: str | None) -> tuple[pd.DataFrame, str]:
    subset = (animal_subset or "combined").lower()
    if subset == "combined":
        return df.copy(), "combined"
    if subset in {"dog", "dogs"}:
        return df.loc[df["animal_type"].astype(str).str.lower().eq("dog")].copy(), "dogs"
    if subset in {"cat", "cats"}:
        return df.loc[df["animal_type"].astype(str).str.lower().eq("cat")].copy(), "cats"
    raise ValueError("animal_subset must be one of: combined, dogs, cats")


def _has_time_split(df: pd.DataFrame) -> bool:
    years = set(df["intake_year"].dropna().astype(int))
    return bool(years & set(range(2013, 2022))) and bool(years & {2024, 2025})


def make_time_split(
    df: pd.DataFrame,
    target_column: str,
    animal_subset: str | None = None,
    recency_weighting: bool = True,
) -> DatasetSplit:
    """Create thesis default split with recency weighting and censoring safeguards."""
    if target_column not in df.columns:
        raise ValueError(f"target column missing: {target_column}")
    if "intake_year" not in df.columns:
        raise ValueError("intake_year column is required for thesis split")

    subset_df, subset_name = _filter_subset(df, animal_subset)
    subset_df = subset_df.dropna(subset=[target_column]).copy()
    if subset_df.empty:
        raise ValueError(f"no rows available for subset={subset_name}, target={target_column}")
    
    if _has_time_split(subset_df):
        train = subset_df.loc[subset_df["intake_year"].between(2013, 2021)].copy()
        validation = subset_df.loc[subset_df["intake_year"].between(2022, 2023)].copy()
        test = subset_df.loc[subset_df["intake_year"].between(2024, 2025)].copy()
        
        if recency_weighting and "intake_datetime" in train.columns and not train.empty:
            train = train.copy()
            train["sample_weight"] = train["intake_datetime"].apply(
                lambda x: 1.0 + 0.5 * (x.year - 2013) / (2021 - 2013)
            )
        
        if not train.empty and not test.empty:
            return DatasetSplit(
                full_data=subset_df,
                train=train,
                validation=validation,
                test=test,
                strategy="time",
                train_period=TRAIN_PERIOD,
                validation_period=VALIDATION_PERIOD,
                test_period=TEST_PERIOD,
                animal_subset=subset_name,
            )

    stratify = None
    if target_column in {"classification_target", "target_adopted"}:
        counts = subset_df[target_column].value_counts()
        if len(counts) == 2 and counts.min() >= 2:
            stratify = subset_df[target_column]

    train, test = train_test_split(
        subset_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    train, validation = train_test_split(
        train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train[target_column] if stratify is not None and train[target_column].value_counts().min() >= 2 else None,
    )
    return DatasetSplit(
        full_data=subset_df,
        train=train,
        validation=validation,
        test=test,
        strategy="random",
        train_period="random_train",
        validation_period="random_validation",
        test_period="random_test",
        animal_subset=subset_name,
    )

