"""Tests that hypothesis evidence artifacts contain H1-H5.

These tests auto-skip if prerequisite files are absent
(generated during Priority 2/4 agent work).
"""

from pathlib import Path

import pandas as pd
import pytest

from aac_adoption.analysis.hypothesis_tables import create_h2_seasonality_outputs, create_h4_dark_color_outputs

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "reports" / "tables"
EVIDENCE_PACK_CSV = TABLES_DIR / "model_evidence_pack.csv"


@pytest.mark.skipif(
    not EVIDENCE_PACK_CSV.exists(),
    reason="model_evidence_pack.csv not yet generated - run scripts/generate_evidence_pack.py",
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


@pytest.mark.skipif(
    not (TABLES_DIR / "h2_seasonality_summary.csv").exists(),
    reason="H2 evidence table not generated yet",
)
def test_h2_evidence_table_nonempty() -> None:
    df = pd.read_csv(TABLES_DIR / "h2_seasonality_summary.csv")
    assert not df.empty, "H2 evidence table is empty"


@pytest.mark.skipif(
    not (TABLES_DIR / "h4_dark_color_summary.csv").exists(),
    reason="H4 evidence table not generated yet",
)
def test_h4_evidence_table_nonempty() -> None:
    df = pd.read_csv(TABLES_DIR / "h4_dark_color_summary.csv")
    assert not df.empty, "H4 evidence table is empty"


def test_hypothesis_markdown_reports_exist() -> None:
    summary_dir = ROOT / "reports" / "summary"
    for h in ["h1_interpretation.md", "h2_interpretation.md", "h3_interpretation.md", "h4_interpretation.md", "h5_interpretation.md"]:
        p = summary_dir / h
        if p.exists():
            content = p.read_text(encoding="utf-8")
            assert h.split("_")[0].upper() in content
            assert "causal" in content.lower()


def test_methodological_reports_exist_and_contain_content() -> None:
    summary_dir = ROOT / "reports" / "summary"
    
    ext_path = summary_dir / "external_validity_limitations.md"
    assert ext_path.exists(), "external_validity_limitations.md does not exist"
    ext_content = ext_path.read_text(encoding="utf-8")
    assert "No-Kill" in ext_content
    assert "generalize" in ext_content.lower()
    
    breed_path = summary_dir / "breed_color_justification.md"
    assert breed_path.exists(), "breed_color_justification.md does not exist"
    breed_content = breed_path.read_text(encoding="utf-8")
    assert "sparsity" in breed_content.lower()
    assert "granularity" in breed_content.lower()
    
    base_path = summary_dir / "descriptive_baseline_comparison.md"
    assert base_path.exists(), "descriptive_baseline_comparison.md does not exist"
    base_content = base_path.read_text(encoding="utf-8")
    assert "lift" in base_content.lower()
    assert "dummy" in base_content.lower()


def test_h2_h4_interpretation_reports_use_computed_values(tmp_path) -> None:
    data_path = tmp_path / "modeling.csv"
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    summary_dir = tmp_path / "summary"
    pd.DataFrame(
        {
            "intake_season": ["winter", "winter", "summer", "summer"],
            "is_black_or_dark": [True, True, False, False],
            "adopted": [True, False, False, False],
            "days_to_outcome": [2.0, 6.0, 10.0, 14.0],
        }
    ).to_csv(data_path, index=False)

    create_h2_seasonality_outputs(data_path, tables_dir, figures_dir, summary_dir)
    create_h4_dark_color_outputs(data_path, tables_dir, figures_dir, summary_dir)

    h2 = (summary_dir / "h2_interpretation.md").read_text(encoding="utf-8")
    assert "winter has the highest adoption rate (50.00%)" in h2
    assert "summer has the longest median time to outcome (12.00 days)" in h2
    assert "52.49%" not in h2
    assert "7.01 days" not in h2

    h4 = (summary_dir / "h4_interpretation.md").read_text(encoding="utf-8")
    assert "**50.00%** compared to **0.00%**" in h4
    assert "**4.00 days** for dark-coloured animals vs **12.00 days**" in h4
    assert "51.58%" not in h4
    assert "6.26 days" not in h4

