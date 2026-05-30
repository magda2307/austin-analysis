from aac_adoption.dashboard.story import approach_comparison_rows, decision_sankey, story_cards, workflow_dot


def test_story_helpers_return_expected_content():
    approaches = approach_comparison_rows()
    assert {"layer", "technology", "answers", "strength", "dashboard_use"}.issubset(approaches.columns)
    assert "CatBoost" in " ".join(approaches["technology"])
    assert "Raw AAC" in workflow_dot()
    assert len(story_cards()) >= 4
    assert decision_sankey().data
