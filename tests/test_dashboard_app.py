import pytest
from streamlit.testing.v1 import AppTest
from unittest.mock import patch
import pandas as pd
import json

from aac_adoption.dashboard.data import DASHBOARD_TABLE_SCHEMAS

# Tests check if streamlit_app.py loads without crashing (zero uncaught exceptions).
# Since it depends on many files, we mock cached_tables and cached_diagnostics where needed,
# or mock the file system if that's easier. But wait, `streamlit_app.py` reads via load_table.

@patch("aac_adoption.dashboard.data.Path.exists")
def test_dashboard_app_current_artifacts_empty(mock_exists):
    mock_exists.return_value = False
    at = AppTest.from_file("streamlit_app.py").run()
    assert not at.exception

def test_dashboard_app_current_artifacts_with_data(tmp_path):
    # Create valid dummy tables for all schemas
    tables_dir = tmp_path / "reports/tables"
    tables_dir.mkdir(parents=True)
    models_dir = tmp_path / "models/advanced"
    models_dir.mkdir(parents=True)
    
    # We patch config paths to our tmp directory
    with patch("aac_adoption.config.PROJECT_ROOT", tmp_path):
        
        from aac_adoption.dashboard.data import TABLE_FILES
        for key, schema in DASHBOARD_TABLE_SCHEMAS.items():
            filename = TABLE_FILES.get(key)
            if not filename:
                continue
            
            # create valid frame
            row = {}
            for col, dtype in schema.items():
                if dtype == "bool":
                    row[col] = True
                elif dtype == "float":
                    row[col] = 1.0
                else:
                    row[col] = "test"
            df = pd.DataFrame([row])
            df.to_csv(tables_dir / filename, index=False)
            
        at = AppTest.from_file("streamlit_app.py").run()
        assert not at.exception

def test_dashboard_app_missing_columns(tmp_path):
    tables_dir = tmp_path / "reports/tables"
    tables_dir.mkdir(parents=True)
    
    with patch("aac_adoption.config.PROJECT_ROOT", tmp_path):
        from aac_adoption.dashboard.data import TABLE_FILES
        # Create a table missing a required column
        df = pd.DataFrame([{"model_name": "catboost"}]) # missing roc_auc, pr_auc
        df.to_csv(tables_dir / TABLE_FILES["classification"], index=False)
        
        at = AppTest.from_file("streamlit_app.py").run()
        assert not at.exception # schema check should fail fast and return empty, app should handle empty gracefully

def test_dashboard_app_string_booleans(tmp_path):
    tables_dir = tmp_path / "reports/tables"
    tables_dir.mkdir(parents=True)
    
    with patch("aac_adoption.config.PROJECT_ROOT", tmp_path):
        from aac_adoption.dashboard.data import TABLE_FILES
        # string booleans should be parsed correctly
        df = pd.DataFrame([{"selected": "True", "task": "classification", "animal_subset": "all", "model_name": "catboost"}])
        df.to_csv(tables_dir / TABLE_FILES["final_model_selection"], index=False)
        
        at = AppTest.from_file("streamlit_app.py").run()
        assert not at.exception

@patch("aac_adoption.dashboard.data.load_model")
def test_dashboard_app_missing_model_corrupt_metadata(mock_load_model, tmp_path):
    # App handles prediction exceptions gracefully
    tables_dir = tmp_path / "reports/tables"
    tables_dir.mkdir(parents=True)
    
    with patch("aac_adoption.config.PROJECT_ROOT", tmp_path):
        from aac_adoption.dashboard.data import TABLE_FILES
        # Ensure we try to load animal archetypes to trigger predict_from_record
        df = pd.DataFrame([{"profile_label": "Test Animal", "is_named": "True"}])
        df.to_csv(tables_dir / TABLE_FILES["animal_archetypes"], index=False)
        
        mock_load_model.side_effect = FileNotFoundError("Missing")
        
        at = AppTest.from_file("streamlit_app.py").run()
        assert not at.exception
