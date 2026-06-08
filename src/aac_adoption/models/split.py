from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from aac_adoption.config import RANDOM_STATE

TRAIN_PERIOD = "2013-2021"
CALIBRATION_PERIOD = "2022"
SELECTION_PERIOD = "2023"
TEST_PERIOD = "2024-2025"

@dataclass(frozen=True)
class DatasetSplit:
    """Container for split frames and reporting metadata."""

    full_data: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    strategy: str
    train_period: str
    test_period: str
    animal_subset: str

    calibration: pd.DataFrame = None
    selection: pd.DataFrame = None
    calibration_period: str = None
    selection_period: str = None
    validation: pd.DataFrame = None
    validation_period: str = None
    is_thesis_evaluation: bool = False

    def __post_init__(self):
        if self.validation is None:
            if self.calibration is not None and self.selection is not None:
                if self.calibration.empty and self.selection.empty:
                    v = pd.DataFrame(columns=self.full_data.columns)
                elif self.calibration.empty:
                    v = self.selection
                elif self.selection.empty:
                    v = self.calibration
                else:
                    v = pd.concat([self.calibration, self.selection])
                object.__setattr__(self, 'validation', v)
        if self.validation_period is None:
            if self.calibration_period is not None and self.selection_period is not None:
                if self.calibration_period == self.selection_period:
                    object.__setattr__(self, 'validation_period', self.calibration_period)
                else:
                    object.__setattr__(self, 'validation_period', f"{self.calibration_period}-{self.selection_period}")



def _filter_subset(df: pd.DataFrame, animal_subset: str | None) -> tuple[pd.DataFrame, str]:
    subset = (animal_subset or "combined").lower()
    if subset == "combined":
        return df.copy(), "combined"
    if subset in {"dog", "dogs"}:
        return df.loc[df["animal_type"].astype(str).str.lower().eq("dog")].copy(), "dogs"
    if subset in {"cat", "cats"}:
        return df.loc[df["animal_type"].astype(str).str.lower().eq("cat")].copy(), "cats"
    raise ValueError("animal_subset must be one of: combined, dogs, cats")

def make_time_split(
    df: pd.DataFrame,
    target_column: str,
    animal_subset: str | None = None,
    recency_weighting: bool = True,
    *,
    allow_random_fallback: bool = False,
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
    
    train = subset_df.loc[subset_df["intake_year"].between(2013, 2021)].copy()
    calibration = subset_df.loc[subset_df["intake_year"] == 2022].copy()
    selection = subset_df.loc[subset_df["intake_year"] == 2023].copy()
    test = subset_df.loc[subset_df["intake_year"].between(2024, 2025)].copy()

    missing = []
    if train.empty: missing.append("train period 2013-2021")
    if calibration.empty: missing.append("calibration period 2022")
    if selection.empty: missing.append("selection period 2023")
    if test.empty: missing.append("test period 2024-2025")

    if not missing:
        if recency_weighting:
            train["sample_weight"] = train["intake_year"].apply(
                lambda y: 1.0 + 0.5 * (y - 2013) / (2021 - 2013) if pd.notnull(y) else 1.0
            ).clip(lower=1.0, upper=1.5)
        
        return DatasetSplit(
            full_data=subset_df,
            train=train,
            calibration=calibration,
            selection=selection,
            test=test,
            strategy="time",
            train_period=TRAIN_PERIOD,
            calibration_period=CALIBRATION_PERIOD,
            selection_period=SELECTION_PERIOD,
            test_period=TEST_PERIOD,
            animal_subset=subset_name,
            is_thesis_evaluation=True,
        )

    if not allow_random_fallback:
        raise ValueError(f"Thesis chronological split unavailable: missing {missing[0]}")

    stratify = None
    if target_column in {"classification_target", "target_adopted"}:
        counts = subset_df[target_column].value_counts()
        if len(counts) == 2 and counts.min() >= 2:
            stratify = subset_df[target_column]

    train_df, test_df = train_test_split(
        subset_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    
    # Let's say random split gives 20% validation, which is then halved into calibration and selection
    train_df, validation_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_df[target_column] if stratify is not None and train_df[target_column].value_counts().min() >= 2 else None,
    )

    mid = len(validation_df) // 2
    calibration_df = validation_df.iloc[:mid].copy()
    selection_df = validation_df.iloc[mid:].copy()

    return DatasetSplit(
        full_data=subset_df,
        train=train_df,
        calibration=calibration_df,
        selection=selection_df,
        test=test_df,
        strategy="random_development_only",
        train_period="random_train",
        calibration_period="random_calibration",
        selection_period="random_selection",
        test_period="random_test",
        animal_subset=subset_name,
        is_thesis_evaluation=False,
    )
