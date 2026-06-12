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


def test_model_sensitivity_checks_prediction_result_before_rendering():
    source = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text(encoding="utf-8")
    block = source.split('with tabs[9]:', 1)[1].split('with tabs[10]:', 1)[0]

    assert "if not prediction.ok:" in block
    assert "prediction.error_message" in block
    button_block = block.split('if st.button(t("Run prediction")', 1)[1].split(
        'if st.session_state.get("prediction_hash")',
        1,
    )[0]
    assert button_block.index('pop("prediction_result", None)') < button_block.index(
        "predict_from_record("
    )
