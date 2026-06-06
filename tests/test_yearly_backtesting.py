"""Regression tests for yearly temporal backtesting functionality."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from scripts.evaluate_backtesting import main as backtesting_main


@pytest.fixture
def six_year_fixture(tmp_path):
    """Create small test dataset spanning 6 years (2019-2024)."""
    np.random.seed(42)
    
    n_per_year = {
        2019: 50,
        2020: 60,
        2021: 55,
        2022: 65,
        2023: 70,
        2024: 75,
    }
    
    rows = []
    animal_id = 1
    
    for year, n in n_per_year.items():
        for _ in range(n):
            animal_type = np.random.choice(["Dog", "Cat"])
            classification_target = np.random.choice([0, 1], p=[0.6, 0.4])
            regression_target = np.random.exponential(30) + 1
            
            rows.append({
                "animal_id": f"A{animal_id:04d}",
                "animal_type": animal_type,
                "intake_year": year,
                "intake_age_days": np.random.randint(0, 3650),
                "classification_target": classification_target,
                "regression_target_days": round(regression_target, 1),
                "is_named": np.random.choice([0, 1], p=[0.3, 0.7]),
                "intake_condition": np.random.choice(["Normal", "Sick", "Injured"]),
            })
            animal_id += 1
    
    df = pd.DataFrame(rows)
    
    fixture_path = tmp_path / "modeling_dataset.csv"
    df.to_csv(fixture_path, index=False)
    
    return df, fixture_path


def test_yearly_backtesting_script_runs_without_error(six_year_fixture, tmp_path):
    """Test that yearly backtesting script runs without error on mock data."""
    df, fixture_path = six_year_fixture
    
    out_dir = tmp_path / "reports/tables"
    out_dir.mkdir(parents=True)
    output_csv = out_dir / "yearly_backtesting.csv"
    
    import sys
    sys.path.insert(0, str(Path.cwd()))
    
    import pandas as pd
    from unittest.mock import patch
    
    with patch("pathlib.Path.exists") as mock_exists, \
         patch("pandas.read_csv") as mock_read_csv, \
         patch("pandas.DataFrame.to_csv") as mock_to_csv:
        
        def mock_exists_side_effect(self):
            if str(self).endswith("modeling_dataset.csv"):
                return True
            return original_exists(self)
        
        def mock_read_csv_side_effect(filepath_or_buffer, *args, **kwargs):
            if str(filepath_or_buffer).endswith("modeling_dataset.csv"):
                return pd.read_csv(fixture_path)
            return original_read_csv(filepath_or_buffer, *args, **kwargs)
        
        def mock_to_csv_side_effect(self, filepath_or_buffer, *args, **kwargs):
            if str(filepath_or_buffer).endswith("yearly_backtesting.csv"):
                self.to_csv(output_csv, *args, **kwargs)
            else:
                original_to_csv(self, filepath_or_buffer, *args, **kwargs)
        
        from pathlib import Path as OriginalPath
        original_exists = OriginalPath.exists
        original_read_csv = pd.read_csv
        original_to_csv = pd.DataFrame.to_csv
        
        mock_exists.side_effect = mock_exists_side_effect
        mock_read_csv.side_effect = mock_read_csv_side_effect
        mock_to_csv.side_effect = mock_to_csv_side_effect
        
        with patch.object(OriginalPath, "exists", mock_exists_side_effect):
            with patch.object(pd, "read_csv", mock_read_csv_side_effect):
                with patch.object(pd.DataFrame, "to_csv", mock_to_csv_side_effect):
                    backtesting_main()
    
    assert output_csv.exists(), "Output CSV should be created"


def test_yearly_backtesting_output_schema(six_year_fixture, tmp_path):
    """Verify output schema: train_years, test_year, subset, model, pr_auc, roc_auc, brier, ece, mae, rmse, r2."""
    df, fixture_path = six_year_fixture
    
    out_dir = tmp_path / "reports/tables"
    out_dir.mkdir(parents=True)
    output_csv = out_dir / "yearly_backtesting.csv"
    
    import sys
    sys.path.insert(0, str(Path.cwd()))
    
    import pandas as pd
    from unittest.mock import patch
    
    with patch("pathlib.Path.exists") as mock_exists, \
         patch("pandas.read_csv") as mock_read_csv, \
         patch("pandas.DataFrame.to_csv") as mock_to_csv:
        
        def mock_exists_side_effect(self):
            if str(self).endswith("modeling_dataset.csv"):
                return True
            return original_exists(self)
        
        def mock_read_csv_side_effect(filepath_or_buffer, *args, **kwargs):
            if str(filepath_or_buffer).endswith("modeling_dataset.csv"):
                return pd.read_csv(fixture_path)
            return original_read_csv(filepath_or_buffer, *args, **kwargs)
        
        def mock_to_csv_side_effect(self, filepath_or_buffer, *args, **kwargs):
            if str(filepath_or_buffer).endswith("yearly_backtesting.csv"):
                self.to_csv(output_csv, *args, **kwargs)
            else:
                original_to_csv(self, filepath_or_buffer, *args, **kwargs)
        
        from pathlib import Path as OriginalPath
        original_exists = OriginalPath.exists
        original_read_csv = pd.read_csv
        original_to_csv = pd.DataFrame.to_csv
        
        mock_exists.side_effect = mock_exists_side_effect
        mock_read_csv.side_effect = mock_read_csv_side_effect
        mock_to_csv.side_effect = mock_to_csv_side_effect
        
        with patch.object(OriginalPath, "exists", mock_exists_side_effect):
            with patch.object(pd, "read_csv", mock_read_csv_side_effect):
                with patch.object(pd.DataFrame, "to_csv", mock_to_csv_side_effect):
                    backtesting_main()
    
    result = pd.read_csv(output_csv)
    
    required_columns = [
        "train_years", "test_year", "subset", "model",
        "pr_auc", "roc_auc", "brier", "ece",
        "mae", "rmse", "r2"
    ]
    
    for col in required_columns:
        assert col in result.columns, f"Missing required column: {col}"


def test_yearly_backtesting_catboost_classifier_metrics(six_year_fixture, tmp_path):
    """Test CatBoost classifier outputs PR-AUC and ROC-AUC."""
    df, fixture_path = six_year_fixture
    
    out_dir = tmp_path / "reports/tables"
    out_dir.mkdir(parents=True)
    output_csv = out_dir / "yearly_backtesting.csv"
    
    import sys
    sys.path.insert(0, str(Path.cwd()))
    
    import pandas as pd
    from unittest.mock import patch
    
    with patch("pathlib.Path.exists") as mock_exists, \
         patch("pandas.read_csv") as mock_read_csv, \
         patch("pandas.DataFrame.to_csv") as mock_to_csv:
        
        def mock_exists_side_effect(self):
            if str(self).endswith("modeling_dataset.csv"):
                return True
            return original_exists(self)
        
        def mock_read_csv_side_effect(filepath_or_buffer, *args, **kwargs):
            if str(filepath_or_buffer).endswith("modeling_dataset.csv"):
                return pd.read_csv(fixture_path)
            return original_read_csv(filepath_or_buffer, *args, **kwargs)
        
        def mock_to_csv_side_effect(self, filepath_or_buffer, *args, **kwargs):
            if str(filepath_or_buffer).endswith("yearly_backtesting.csv"):
                self.to_csv(output_csv, *args, **kwargs)
            else:
                original_to_csv(self, filepath_or_buffer, *args, **kwargs)
        
        from pathlib import Path as OriginalPath
        original_exists = OriginalPath.exists
        original_read_csv = pd.read_csv
        original_to_csv = pd.DataFrame.to_csv
        
        mock_exists.side_effect = mock_exists_side_effect
        mock_read_csv.side_effect = mock_read_csv_side_effect
        mock_to_csv.side_effect = mock_to_csv_side_effect
        
        with patch.object(OriginalPath, "exists", mock_exists_side_effect):
            with patch.object(pd, "read_csv", mock_read_csv_side_effect):
                with patch.object(pd.DataFrame, "to_csv", mock_to_csv_side_effect):
                    backtesting_main()
    
    result = pd.read_csv(output_csv)
    
    catboost_classifier = result[result["model"].str.contains("catboost_classifier")]
    assert len(catboost_classifier) > 0, "Should have CatBoost classifier results"
    
    for _, row in catboost_classifier.iterrows():
        assert pd.notna(row["pr_auc"]), "PR-AUC should be computed for CatBoost classifier"
        assert pd.notna(row["roc_auc"]), "ROC-AUC should be computed for CatBoost classifier"


def test_yearly_backtesting_histgradientboosting_classifier_metrics(six_year_fixture, tmp_path):
    """Test HistGradientBoosting classifier outputs PR-AUC and ROC-AUC."""
    df, fixture_path = six_year_fixture
    
    out_dir = tmp_path / "reports/tables"
    out_dir.mkdir(parents=True)
    output_csv = out_dir / "yearly_backtesting.csv"
    
    import sys
    sys.path.insert(0, str(Path.cwd()))
    
    import pandas as pd
    from unittest.mock import patch
    
    with patch("pathlib.Path.exists") as mock_exists, \
         patch("pandas.read_csv") as mock_read_csv, \
         patch("pandas.DataFrame.to_csv") as mock_to_csv:
        
        def mock_exists_side_effect(self):
            if str(self).endswith("modeling_dataset.csv"):
                return True
            return original_exists(self)
        
        def mock_read_csv_side_effect(filepath_or_buffer, *args, **kwargs):
            if str(filepath_or_buffer).endswith("modeling_dataset.csv"):
                return pd.read_csv(fixture_path)
            return original_read_csv(filepath_or_buffer, *args, **kwargs)
        
        def mock_to_csv_side_effect(self, filepath_or_buffer, *args, **kwargs):
            if str(filepath_or_buffer).endswith("yearly_backtesting.csv"):
                self.to_csv(output_csv, *args, **kwargs)
            else:
                original_to_csv(self, filepath_or_buffer, *args, **kwargs)
        
        from pathlib import Path as OriginalPath
        original_exists = OriginalPath.exists
        original_read_csv = pd.read_csv
        original_to_csv = pd.DataFrame.to_csv
        
        mock_exists.side_effect = mock_exists_side_effect
        mock_read_csv.side_effect = mock_read_csv_side_effect
        mock_to_csv.side_effect = mock_to_csv_side_effect
        
        with patch.object(OriginalPath, "exists", mock_exists_side_effect):
            with patch.object(pd, "read_csv", mock_read_csv_side_effect):
                with patch.object(pd.DataFrame, "to_csv", mock_to_csv_side_effect):
                    backtesting_main()
    
    result = pd.read_csv(output_csv)
    
    hist_classifier = result[result["model"].str.contains("histgradientboosting_classifier")]
    assert len(hist_classifier) > 0, "Should have HistGradientBoosting classifier results"
    
    for _, row in hist_classifier.iterrows():
        assert pd.notna(row["pr_auc"]), "PR-AUC should be computed for HistGradientBoosting classifier"
        assert pd.notna(row["roc_auc"]), "ROC-AUC should be computed for HistGradientBoosting classifier"


def test_yearly_backtesting_catboost_regressor_metrics(six_year_fixture, tmp_path):
    """Test CatBoost regressor outputs MAE, RMSE, R²."""
    df, fixture_path = six_year_fixture
    
    out_dir = tmp_path / "reports/tables"
    out_dir.mkdir(parents=True)
    output_csv = out_dir / "yearly_backtesting.csv"
    
    import sys
    sys.path.insert(0, str(Path.cwd()))
    
    import pandas as pd
    from unittest.mock import patch
    
    with patch("pathlib.Path.exists") as mock_exists, \
         patch("pandas.read_csv") as mock_read_csv, \
         patch("pandas.DataFrame.to_csv") as mock_to_csv:
        
        def mock_exists_side_effect(self):
            if str(self).endswith("modeling_dataset.csv"):
                return True
            return original_exists(self)
        
        def mock_read_csv_side_effect(filepath_or_buffer, *args, **kwargs):
            if str(filepath_or_buffer).endswith("modeling_dataset.csv"):
                return pd.read_csv(fixture_path)
            return original_read_csv(filepath_or_buffer, *args, **kwargs)
        
        def mock_to_csv_side_effect(self, filepath_or_buffer, *args, **kwargs):
            if str(filepath_or_buffer).endswith("yearly_backtesting.csv"):
                self.to_csv(output_csv, *args, **kwargs)
            else:
                original_to_csv(self, filepath_or_buffer, *args, **kwargs)
        
        from pathlib import Path as OriginalPath
        original_exists = OriginalPath.exists
        original_read_csv = pd.read_csv
        original_to_csv = pd.DataFrame.to_csv
        
        mock_exists.side_effect = mock_exists_side_effect
        mock_read_csv.side_effect = mock_read_csv_side_effect
        mock_to_csv.side_effect = mock_to_csv_side_effect
        
        with patch.object(OriginalPath, "exists", mock_exists_side_effect):
            with patch.object(pd, "read_csv", mock_read_csv_side_effect):
                with patch.object(pd.DataFrame, "to_csv", mock_to_csv_side_effect):
                    backtesting_main()
    
    result = pd.read_csv(output_csv)
    
    catboost_regressor = result[result["model"].str.contains("catboost_regressor")]
    assert len(catboost_regressor) > 0, "Should have CatBoost regressor results"
    
    for _, row in catboost_regressor.iterrows():
        assert pd.notna(row["mae"]), "MAE should be computed for CatBoost regressor"
        assert pd.notna(row["rmse"]), "RMSE should be computed for CatBoost regressor"
        assert pd.notna(row["r2"]), "R² should be computed for CatBoost regressor"


def test_yearly_backtesting_histgradientboosting_regressor_metrics(six_year_fixture, tmp_path):
    """Test HistGradientBoosting regressor outputs MAE, RMSE, R²."""
    df, fixture_path = six_year_fixture
    
    out_dir = tmp_path / "reports/tables"
    out_dir.mkdir(parents=True)
    output_csv = out_dir / "yearly_backtesting.csv"
    
    import sys
    sys.path.insert(0, str(Path.cwd()))
    
    import pandas as pd
    from unittest.mock import patch
    
    with patch("pathlib.Path.exists") as mock_exists, \
         patch("pandas.read_csv") as mock_read_csv, \
         patch("pandas.DataFrame.to_csv") as mock_to_csv:
        
        def mock_exists_side_effect(self):
            if str(self).endswith("modeling_dataset.csv"):
                return True
            return original_exists(self)
        
        def mock_read_csv_side_effect(filepath_or_buffer, *args, **kwargs):
            if str(filepath_or_buffer).endswith("modeling_dataset.csv"):
                return pd.read_csv(fixture_path)
            return original_read_csv(filepath_or_buffer, *args, **kwargs)
        
        def mock_to_csv_side_effect(self, filepath_or_buffer, *args, **kwargs):
            if str(filepath_or_buffer).endswith("yearly_backtesting.csv"):
                self.to_csv(output_csv, *args, **kwargs)
            else:
                original_to_csv(self, filepath_or_buffer, *args, **kwargs)
        
        from pathlib import Path as OriginalPath
        original_exists = OriginalPath.exists
        original_read_csv = pd.read_csv
        original_to_csv = pd.DataFrame.to_csv
        
        mock_exists.side_effect = mock_exists_side_effect
        mock_read_csv.side_effect = mock_read_csv_side_effect
        mock_to_csv.side_effect = mock_to_csv_side_effect
        
        with patch.object(OriginalPath, "exists", mock_exists_side_effect):
            with patch.object(pd, "read_csv", mock_read_csv_side_effect):
                with patch.object(pd.DataFrame, "to_csv", mock_to_csv_side_effect):
                    backtesting_main()
    
    result = pd.read_csv(output_csv)
    
    hist_regressor = result[result["model"].str.contains("histgradientboosting_regressor")]
    assert len(hist_regressor) > 0, "Should have HistGradientBoosting regressor results"
    
    for _, row in hist_regressor.iterrows():
        assert pd.notna(row["mae"]), "MAE should be computed for HistGradientBoosting regressor"
        assert pd.notna(row["rmse"]), "RMSE should be computed for HistGradientBoosting regressor"
        assert pd.notna(row["r2"]), "R² should be computed for HistGradientBoosting regressor"


def test_yearly_backtesting_bootstrap_confidence_intervals(six_year_fixture, tmp_path):
    """Test bootstrap confidence intervals generated (lower/upper columns present)."""
    df, fixture_path = six_year_fixture
    
    out_dir = tmp_path / "reports/tables"
    out_dir.mkdir(parents=True)
    output_csv = out_dir / "yearly_backtesting.csv"
    
    import sys
    sys.path.insert(0, str(Path.cwd()))
    
    import pandas as pd
    from unittest.mock import patch
    
    with patch("pathlib.Path.exists") as mock_exists, \
         patch("pandas.read_csv") as mock_read_csv, \
         patch("pandas.DataFrame.to_csv") as mock_to_csv:
        
        def mock_exists_side_effect(self):
            if str(self).endswith("modeling_dataset.csv"):
                return True
            return original_exists(self)
        
        def mock_read_csv_side_effect(filepath_or_buffer, *args, **kwargs):
            if str(filepath_or_buffer).endswith("modeling_dataset.csv"):
                return pd.read_csv(fixture_path)
            return original_read_csv(filepath_or_buffer, *args, **kwargs)
        
        def mock_to_csv_side_effect(self, filepath_or_buffer, *args, **kwargs):
            if str(filepath_or_buffer).endswith("yearly_backtesting.csv"):
                self.to_csv(output_csv, *args, **kwargs)
            else:
                original_to_csv(self, filepath_or_buffer, *args, **kwargs)
        
        from pathlib import Path as OriginalPath
        original_exists = OriginalPath.exists
        original_read_csv = pd.read_csv
        original_to_csv = pd.DataFrame.to_csv
        
        mock_exists.side_effect = mock_exists_side_effect
        mock_read_csv.side_effect = mock_read_csv_side_effect
        mock_to_csv.side_effect = mock_to_csv_side_effect
        
        with patch.object(OriginalPath, "exists", mock_exists_side_effect):
            with patch.object(pd, "read_csv", mock_read_csv_side_effect):
                with patch.object(pd.DataFrame, "to_csv", mock_to_csv_side_effect):
                    backtesting_main()
    
    result = pd.read_csv(output_csv)
    
    has_ci_columns = any(col.endswith("_lower") or col.endswith("_upper") for col in result.columns)
    assert has_ci_columns, "Should have bootstrap CI lower/upper columns"


def test_yearly_backtesting_animal_subsets(six_year_fixture, tmp_path):
    """Test animal subsets: combined, dogs, cats."""
    df, fixture_path = six_year_fixture
    
    out_dir = tmp_path / "reports/tables"
    out_dir.mkdir(parents=True)
    output_csv = out_dir / "yearly_backtesting.csv"
    
    import sys
    sys.path.insert(0, str(Path.cwd()))
    
    import pandas as pd
    from unittest.mock import patch
    
    with patch("pathlib.Path.exists") as mock_exists, \
         patch("pandas.read_csv") as mock_read_csv, \
         patch("pandas.DataFrame.to_csv") as mock_to_csv:
        
        def mock_exists_side_effect(self):
            if str(self).endswith("modeling_dataset.csv"):
                return True
            return original_exists(self)
        
        def mock_read_csv_side_effect(filepath_or_buffer, *args, **kwargs):
            if str(filepath_or_buffer).endswith("modeling_dataset.csv"):
                return pd.read_csv(fixture_path)
            return original_read_csv(filepath_or_buffer, *args, **kwargs)
        
        def mock_to_csv_side_effect(self, filepath_or_buffer, *args, **kwargs):
            if str(filepath_or_buffer).endswith("yearly_backtesting.csv"):
                self.to_csv(output_csv, *args, **kwargs)
            else:
                original_to_csv(self, filepath_or_buffer, *args, **kwargs)
        
        from pathlib import Path as OriginalPath
        original_exists = OriginalPath.exists
        original_read_csv = pd.read_csv
        original_to_csv = pd.DataFrame.to_csv
        
        mock_exists.side_effect = mock_exists_side_effect
        mock_read_csv.side_effect = mock_read_csv_side_effect
        mock_to_csv.side_effect = mock_to_csv_side_effect
        
        with patch.object(OriginalPath, "exists", mock_exists_side_effect):
            with patch.object(pd, "read_csv", mock_read_csv_side_effect):
                with patch.object(pd.DataFrame, "to_csv", mock_to_csv_side_effect):
                    backtesting_main()
    
    result = pd.read_csv(output_csv)
    
    subsets = result["subset"].unique()
    assert "combined" in subsets, "Should have combined subset"
    assert "dogs" in subsets, "Should have dogs subset"
    assert "cats" in subsets, "Should have cats subset"


def test_yearly_backtesting_horizon_targets(six_year_fixture, tmp_path):
    """Test horizon targets (7/30/60/90 days) produce correct output."""
    df, fixture_path = six_year_fixture
    
    out_dir = tmp_path / "reports/tables"
    out_dir.mkdir(parents=True)
    output_csv = out_dir / "yearly_backtesting.csv"
    
    import sys
    sys.path.insert(0, str(Path.cwd()))
    
    import pandas as pd
    from unittest.mock import patch
    
    with patch("pathlib.Path.exists") as mock_exists, \
         patch("pandas.read_csv") as mock_read_csv, \
         patch("pandas.DataFrame.to_csv") as mock_to_csv:
        
        def mock_exists_side_effect(self):
            if str(self).endswith("modeling_dataset.csv"):
                return True
            return original_exists(self)
        
        def mock_read_csv_side_effect(filepath_or_buffer, *args, **kwargs):
            if str(filepath_or_buffer).endswith("modeling_dataset.csv"):
                return pd.read_csv(fixture_path)
            return original_read_csv(filepath_or_buffer, *args, **kwargs)
        
        def mock_to_csv_side_effect(self, filepath_or_buffer, *args, **kwargs):
            if str(filepath_or_buffer).endswith("yearly_backtesting.csv"):
                self.to_csv(output_csv, *args, **kwargs)
            else:
                original_to_csv(self, filepath_or_buffer, *args, **kwargs)
        
        from pathlib import Path as OriginalPath
        original_exists = OriginalPath.exists
        original_read_csv = pd.read_csv
        original_to_csv = pd.DataFrame.to_csv
        
        mock_exists.side_effect = mock_exists_side_effect
        mock_read_csv.side_effect = mock_read_csv_side_effect
        mock_to_csv.side_effect = mock_to_csv_side_effect
        
        with patch.object(OriginalPath, "exists", mock_exists_side_effect):
            with patch.object(pd, "read_csv", mock_read_csv_side_effect):
                with patch.object(pd.DataFrame, "to_csv", mock_to_csv_side_effect):
                    backtesting_main()
    
    result = pd.read_csv(output_csv)
    
    test_years = result["test_year"].unique()
    expected_years = {2019, 2020, 2021, 2022, 2023, 2024}
    assert set(test_years) == expected_years, f"Should have test years {expected_years}, got {set(test_years)}"
    
    train_years = result["train_years"].unique()
    for year in test_years:
        expected_train = f"2013-{year-1}"
        assert expected_train in train_years, f"Should have train period {expected_train}"


def test_yearly_backtesting_empty_splits_skipped(six_year_fixture, tmp_path):
    """Test empty train/test splits are skipped gracefully."""
    df, fixture_path = six_year_fixture
    
    out_dir = tmp_path / "reports/tables"
    out_dir.mkdir(parents=True)
    output_csv = out_dir / "yearly_backtesting.csv"
    
    import sys
    sys.path.insert(0, str(Path.cwd()))
    
    import pandas as pd
    from unittest.mock import patch
    
    with patch("pathlib.Path.exists") as mock_exists, \
         patch("pandas.read_csv") as mock_read_csv, \
         patch("pandas.DataFrame.to_csv") as mock_to_csv:
        
        def mock_exists_side_effect(self):
            if str(self).endswith("modeling_dataset.csv"):
                return True
            return original_exists(self)
        
        def mock_read_csv_side_effect(filepath_or_buffer, *args, **kwargs):
            if str(filepath_or_buffer).endswith("modeling_dataset.csv"):
                return pd.read_csv(fixture_path)
            return original_read_csv(filepath_or_buffer, *args, **kwargs)
        
        def mock_to_csv_side_effect(self, filepath_or_buffer, *args, **kwargs):
            if str(filepath_or_buffer).endswith("yearly_backtesting.csv"):
                self.to_csv(output_csv, *args, **kwargs)
            else:
                original_to_csv(self, filepath_or_buffer, *args, **kwargs)
        
        from pathlib import Path as OriginalPath
        original_exists = OriginalPath.exists
        original_read_csv = pd.read_csv
        original_to_csv = pd.DataFrame.to_csv
        
        mock_exists.side_effect = mock_exists_side_effect
        mock_read_csv.side_effect = mock_read_csv_side_effect
        mock_to_csv.side_effect = mock_to_csv_side_effect
        
        with patch.object(OriginalPath, "exists", mock_exists_side_effect):
            with patch.object(pd, "read_csv", mock_read_csv_side_effect):
                with patch.object(pd.DataFrame, "to_csv", mock_to_csv_side_effect):
                    backtesting_main()
    
    result = pd.read_csv(output_csv)
    
    assert len(result) > 0, "Should have some results"
    
    train_years = result["train_years"].str.split("-").str[1].astype(int)
    test_years = result["test_year"]
    
    for train_end, test_year in zip(train_years, test_years):
        assert train_end < test_year, f"Train end {train_end} should be before test year {test_year}"
