from pathlib import Path

from aac_adoption.models.artifacts import artifact_path, safe_name


def test_safe_name_normalizes_model_identifiers():
    assert safe_name("Random Forest / Dogs Only") == "random_forest_dogs_only"


def test_artifact_path_uses_task_subset_and_model(tmp_path: Path):
    path = artifact_path(
        base_dir=tmp_path,
        task="classification",
        animal_subset="combined",
        model_name="logistic_regression",
    )

    assert path == tmp_path / "classification" / "combined" / "logistic_regression.joblib"

