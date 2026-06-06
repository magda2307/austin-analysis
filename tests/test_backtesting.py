import pytest
import pandas as pd
from pathlib import Path
import subprocess

def test_backtesting_script_output_schema(tmp_path, monkeypatch):
    # Mock the data path to a tiny fixture
    fixture_path = tmp_path / "modeling_dataset.csv"
    
    # Create tiny fixture spanning 3 years
    df = pd.DataFrame({
        "animal_id": [1, 2, 3, 4, 5, 6],
        "intake_year": [2018, 2018, 2019, 2019, 2020, 2020],
        "classification_target": [1, 0, 1, 0, 1, 0],
        "regression_target_days": [10.0, 5.0, 10.0, 5.0, 10.0, 5.0],
        "animal_type": ["Dog", "Cat", "Dog", "Cat", "Dog", "Cat"],
        "intake_age_days": [100, 200, 100, 200, 100, 200],
    })
    df.to_csv(fixture_path, index=False)
    
    # Patch Path to use tmp_path
    original_exists = Path.exists
    def mock_exists(self):
        if str(self) == "data/processed/modeling_dataset.csv":
            return True
        return original_exists(self)
        
    original_read_csv = pd.read_csv
    def mock_read_csv(filepath_or_buffer, *args, **kwargs):
        if str(filepath_or_buffer) == "data/processed/modeling_dataset.csv":
            return original_read_csv(fixture_path, *args, **kwargs)
        return original_read_csv(filepath_or_buffer, *args, **kwargs)
        
    monkeypatch.setattr(Path, "exists", mock_exists)
    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    
    out_dir = tmp_path / "reports/tables"
    out_dir.mkdir(parents=True)
    
    # Hook the output path
    def mock_to_csv(self, filepath_or_buffer, *args, **kwargs):
        if str(filepath_or_buffer) == "reports/tables/yearly_backtesting.csv":
            return original_to_csv(self, out_dir / "yearly_backtesting.csv", *args, **kwargs)
        return original_to_csv(self, filepath_or_buffer, *args, **kwargs)
        
    original_to_csv = pd.DataFrame.to_csv
    monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)

    import sys
    sys.path.insert(0, str(Path.cwd()))
    import scripts.evaluate_backtesting
    scripts.evaluate_backtesting.main()
    
    out_file = out_dir / "yearly_backtesting.csv"
    assert out_file.exists()
    
    result = pd.read_csv(out_file)
    assert "train_years" in result.columns
    assert "test_year" in result.columns
    assert "subset" in result.columns
    assert "model" in result.columns
    assert "pr_auc" in result.columns
    assert "roc_auc" in result.columns
    assert "brier" in result.columns
    assert "ece" in result.columns
    
    assert len(result) == 2 # 2018 -> 2019, 2018-2019 -> 2020
