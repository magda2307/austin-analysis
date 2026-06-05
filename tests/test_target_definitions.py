"""Tests for target definition documentation."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TARGET_DOC = ROOT / "docs" / "target_definitions.md"


def test_target_definitions_doc_exists() -> None:
    assert TARGET_DOC.exists(), f"Missing: {TARGET_DOC}"


def test_target_definitions_contains_binary_outcome() -> None:
    content = TARGET_DOC.read_text(encoding="utf-8")
    assert "binary" in content.lower() or "adopted" in content.lower(), (
        "target_definitions.md should define a binary adoption outcome"
    )


def test_target_definitions_contains_los() -> None:
    content = TARGET_DOC.read_text(encoding="utf-8")
    assert any(term in content.lower() for term in ["los", "days", "length of stay"]), (
        "target_definitions.md should define a length-of-stay / days target"
    )


def test_target_definitions_mentions_leakage() -> None:
    content = TARGET_DOC.read_text(encoding="utf-8")
    assert "leakage" in content.lower() or "intake" in content.lower(), (
        "target_definitions.md should discuss intake-time features and leakage"
    )
