import pytest
import pandas as pd
import sys
from pathlib import Path

ROOT = Path("c:/Users/paula/Documents/mgr pjatk")
sys.path.insert(0, str(ROOT / "src"))

from aac_adoption.features.feature_sets import PROHIBITED_MODEL_COLUMNS, validate_no_leakage
from aac_adoption.data.leakage_audit import audit_leakage_columns, DataLeakageError

def test_all_prohibited_columns_rejected():
    future_cols = ["future_adoption", "next_month_status", "_next_event"]
    cols = list(PROHIBITED_MODEL_COLUMNS) + future_cols
    
    # 1. Test audit_leakage_columns
    df = pd.DataFrame(columns=cols)
    with pytest.raises(DataLeakageError, match="Unsafe leakage columns detected") as exc:
        audit_leakage_columns(df)
        
    err_msg = str(exc.value)
    for col in cols:
        assert f"'{col}'" in err_msg or col in err_msg, f"Column {col} not marked unsafe by audit_leakage_columns"
        
    # 2. Test validate_no_leakage
    with pytest.raises(ValueError, match="Leakage columns cannot be model features") as exc:
        validate_no_leakage(cols)
        
    err_msg2 = str(exc.value)
    for col in cols:
        assert f"'{col}'" in err_msg2 or col in err_msg2, f"Column {col} not rejected by validate_no_leakage"

if __name__ == "__main__":
    test_all_prohibited_columns_rejected()
    print("Passed!")
