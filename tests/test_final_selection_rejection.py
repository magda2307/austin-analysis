import pandas as pd
from aac_adoption.analysis.model_selection import create_final_model_selection

def test_final_selection_rejects_missing_provenance(tmp_path):
    tables_dir = tmp_path / "tables"
    summary_dir = tmp_path / "summary"
    tables_dir.mkdir()
    
    pd.DataFrame(
        [
            {"model_name": "no_artifact", "animal_subset": "combined", "roc_auc": 0.90, "pr_auc": 0.70, "f1": 0.6, "split_strategy": "time", "metric_split": "selection", "expected_calibration_error": 0.1},
            {"model_name": "wrong_split", "animal_subset": "combined", "roc_auc": 0.90, "pr_auc": 0.70, "f1": 0.6, "artifact_path": "path", "split_strategy": "random", "metric_split": "selection", "expected_calibration_error": 0.1},
            {"model_name": "wrong_metric_split", "animal_subset": "combined", "roc_auc": 0.90, "pr_auc": 0.70, "f1": 0.6, "artifact_path": "path", "split_strategy": "time", "metric_split": "test", "expected_calibration_error": 0.1},
            {"model_name": "no_calibration", "animal_subset": "combined", "roc_auc": 0.90, "pr_auc": 0.70, "f1": 0.6, "artifact_path": "path", "split_strategy": "time", "metric_split": "selection"},
        ]
    ).to_csv(tables_dir / "model_comparison_classification.csv", index=False)
    
    create_final_model_selection(tables_dir, summary_dir)
    table = pd.read_csv(tables_dir / "final_model_selection.csv")
    
    assert table.empty or (table["task"] == "classification").sum() == 0
