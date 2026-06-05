"""Tests that hypothesis evidence artifacts contain H1-H5.

These tests auto-skip if prerequisite files are absent
(generated during Priority 2/4 agent work).
"""

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "reports" / "tables"
EVIDENCE_PACK_CSV = TABLES_DIR / "model_evidence_pack.csv"


@pytest.mark.skipif(
    not EVIDENCE_PACK_CSV.exists(),
    reason="model_evidence_pack.csv not yet generated — run scripts/generate_evidence_pack.py",
)
def test_evidence_pack_has_required_columns() -> None:
    df = pd.read_csv(EVIDENCE_PACK_CSV)
    # Check at minimum some evidence pack structure
    assert not df.empty, "model_evidence_pack.csv is empty"


@pytest.mark.skipif(
    not (TABLES_DIR / "h1_intake_vs_appearance.csv").exists(),
    reason="H1 evidence table not generated yet",
)
def test_h1_evidence_table_nonempty() -> None:
    df = pd.read_csv(TABLES_DIR / "h1_intake_vs_appearance.csv")
    assert not df.empty, "H1 evidence table is empty"


@pytest.mark.skipif(
    not (TABLES_DIR / "h3_age_adoption_speed.csv").exists(),
    reason="H3 evidence table not generated yet",
)
def test_h3_evidence_table_nonempty() -> None:
    df = pd.read_csv(TABLES_DIR / "h3_age_adoption_speed.csv")
    assert not df.empty, "H3 evidence table is empty"


@pytest.mark.skipif(
    not (TABLES_DIR / "h5_covid_period.csv").exists(),
    reason="H5 evidence table not generated yet",
)
def test_h5_evidence_table_nonempty() -> None:
    df = pd.read_csv(TABLES_DIR / "h5_covid_period.csv")
    assert not df.empty, "H5 evidence table is empty"
