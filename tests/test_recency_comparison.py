import pytest
import pandas as pd
from datetime import datetime
from aac_adoption.models.split import make_time_split

def test_recency_weight_chronological_constraint():
    # Create a dummy dataframe spanning 2013 to 2024
    df = pd.DataFrame({
        "intake_datetime": [
            datetime(2013, 1, 1),
            datetime(2017, 1, 1),
            datetime(2021, 1, 1),
            datetime(2022, 1, 1),
            datetime(2024, 1, 1)
        ],
        "intake_year": [2013, 2017, 2021, 2022, 2024],
        "classification_target": [0, 1, 0, 1, 0],
        "animal_type": ["Dog", "Dog", "Dog", "Dog", "Dog"]
    })
    
    split = make_time_split(df, target_column="classification_target", recency_weighting=True)
    train_df = split.train
    
    assert "sample_weight" in train_df.columns
    
    weight_2013 = train_df.loc[train_df["intake_year"] == 2013, "sample_weight"].iloc[0]
    weight_2017 = train_df.loc[train_df["intake_year"] == 2017, "sample_weight"].iloc[0]
    weight_2021 = train_df.loc[train_df["intake_year"] == 2021, "sample_weight"].iloc[0]
    
    assert weight_2021 > weight_2013, f"Weight for 2021 ({weight_2021}) should be > 2013 ({weight_2013})"
    assert weight_2021 > weight_2017, f"Weight for 2021 ({weight_2021}) should be > 2017 ({weight_2017})"
    assert weight_2017 > weight_2013, f"Weight for 2017 ({weight_2017}) should be > 2013 ({weight_2013})"
    
    assert weight_2013 == 1.0
    assert weight_2021 == 1.5
