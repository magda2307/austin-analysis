import pandas as pd

from aac_adoption.data.clean_data import parse_datetime_columns


def test_parse_datetime_columns_removes_timezone_without_shifting_clock_time():
    df = pd.DataFrame({"outcome_datetime": ["2013-12-02T00:00:00-05:00"]})

    result = parse_datetime_columns(df, ["outcome_datetime"])

    assert str(result["outcome_datetime"].dtype) == "datetime64[ns]"
    assert result.loc[0, "outcome_datetime"] == pd.Timestamp("2013-12-02 00:00:00")


def test_parse_datetime_columns_handles_mixed_aac_formats():
    df = pd.DataFrame(
        {
            "outcome_datetime": [
                "2013-12-02T00:00:00-05:00",
                "2013-10-01T09:31:00",
                "10/01/2013 07:51:00 AM",
            ]
        }
    )

    result = parse_datetime_columns(df, ["outcome_datetime"])

    assert result["outcome_datetime"].isna().sum() == 0
    assert result.loc[1, "outcome_datetime"] == pd.Timestamp("2013-10-01 09:31:00")
