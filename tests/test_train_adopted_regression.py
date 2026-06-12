import pandas as pd

from aac_adoption.models.split import DatasetSplit
from aac_adoption.models.train_adopted_regression import _fit_and_save_adopted


class _FakeModel:
    def fit(self, **kwargs):
        return self


def test_adopted_regression_metadata_includes_dataset_contract(tmp_path, monkeypatch):
    data_path = tmp_path / "modeling_dataset.csv"
    data_path.write_text("feature,days_to_adoption\n1,3\n", encoding="utf-8")
    frame = pd.DataFrame({"feature": [1.0], "days_to_adoption": [3.0]})
    split = DatasetSplit(
        full_data=frame,
        train=frame,
        test=frame.iloc[0:0],
        strategy="time",
        train_period="2013-2021",
        test_period="2024-2025",
        animal_subset="combined",
    )
    captured = {}

    def fake_metadata(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        "aac_adoption.models.train_adopted_regression.base_training_metadata",
        fake_metadata,
    )
    monkeypatch.setattr(
        "aac_adoption.models.train_adopted_regression.save_model_artifact",
        lambda *args, **kwargs: tmp_path / "model.joblib",
    )

    _fit_and_save_adopted(
        model=_FakeModel(),
        task="regression_adopted",
        split=split,
        feature_columns=["feature"],
        target_column="days_to_adoption",
        dataset_path=str(data_path),
        models_dir=tmp_path,
        run_timestamp="2026-06-12T00:00:00+00:00",
        params={},
    )

    assert captured["target_column"] == "days_to_adoption"
    assert captured["dataset_path"] == str(data_path)
