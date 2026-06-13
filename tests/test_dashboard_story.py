import ast
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


def _polish_dashboard_copy() -> dict[str, str]:
    source_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "PL" for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("streamlit_app.py must define the PL translation dictionary")


def test_polish_dashboard_copy_avoids_unnecessary_academic_jargon():
    copy = " ".join(_polish_dashboard_copy().values()).casefold()
    forbidden_fragments = (
        "kohort",
        "inkorpor",
        "deinkorpor",
        "kwantyl temporal",
        "wektor zapytań",
        "kauzal",
        "paradygmat optymalizacyjny",
        "dyskrepanc",
        "dychotomia dla reżimu",
    )
    assert not [fragment for fragment in forbidden_fragments if fragment in copy]


def test_polish_dashboard_copy_preserves_key_methodological_meaning():
    copy = _polish_dashboard_copy()
    assert copy["Predicted days to outcome"] == "Przewidywana liczba dni do dowolnego wyniku"
    assert copy["Cohort size"] == "Liczba zwierząt w grupie"
    finding = copy[
        "**Finding:** Older animals face significant penalties in adoption likelihood. "
        "Wait times to any outcome are complex, as seniors may leave the shelter faster "
        "due to higher rates of non-adoption outcomes."
    ]
    assert "nieadopcyjnych wyników" in finding
    assert "eutanazj" not in finding.casefold()


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
