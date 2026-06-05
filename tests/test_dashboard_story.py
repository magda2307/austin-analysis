from pathlib import Path

from aac_adoption.dashboard.story import approach_comparison_rows, decision_sankey, story_cards, workflow_dot


def test_story_helpers_return_expected_content():
    approaches = approach_comparison_rows()
    assert {"layer", "technology", "answers", "strength", "dashboard_use"}.issubset(approaches.columns)
    assert "CatBoost" in " ".join(approaches["technology"])
    assert "Raw AAC" in workflow_dot()
    assert len(story_cards()) >= 4
    assert decision_sankey().data


def test_streamlit_report_allowlist_is_thesis_only():
    source = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text(encoding="utf-8")
    block = source.split("THESIS_REPORT_OPTIONS = {", 1)[1].split("}\n\nLANGUAGES", 1)[0]

    assert "docs/target_definitions.md" in block
    assert "reports/summary/h2_interpretation.md" in block
    assert "reports/summary/h4_interpretation.md" in block
    assert "data/raw/" not in block
    assert "data/processed/" not in block
    assert "models/" not in block
    assert "What-if" not in block
