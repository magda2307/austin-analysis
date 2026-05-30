import pandas as pd

from aac_adoption.visualization.plots import create_eda_outputs


def test_create_eda_outputs_writes_priority_tables(tmp_path):
    data_path = tmp_path / "modeling_dataset.csv"
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    pd.DataFrame(
        {
            "intake_year": [2019, 2020, 2024],
            "animal_type": ["Dog", "Cat", "Dog"],
            "adopted": [True, False, True],
            "days_to_outcome": [3.0, 10.0, 5.0],
            "age_group": ["baby", "adult", "senior"],
            "intake_type": ["Stray", "Owner Surrender", "Stray"],
            "intake_condition": ["Normal", "Normal", "Injured"],
            "simplified_breed_group": ["retriever_type", "domestic_cat", "retriever_type"],
            "simplified_color_group": ["black_or_dark", "brown_tan", "white_light"],
            "covid_period": ["pre_covid", "covid", "post_covid"],
            "intake_season": ["winter", "spring", "summer"],
            "is_black_or_dark": [True, False, False],
        }
    ).to_csv(data_path, index=False)

    create_eda_outputs(data_path, tables_dir, figures_dir)

    expected = {
        "adoption_los_by_intake_type.csv",
        "adoption_los_by_intake_condition.csv",
        "adoption_los_by_simplified_breed_group.csv",
        "adoption_los_by_simplified_color_group.csv",
        "adoption_los_by_age_group.csv",
        "adoption_los_by_covid_period.csv",
        "adoption_los_by_intake_season.csv",
        "adoption_los_by_is_black_or_dark.csv",
    }
    assert expected.issubset({path.name for path in tables_dir.glob("*.csv")})
