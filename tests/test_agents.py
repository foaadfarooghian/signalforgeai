from __future__ import annotations

import pytest

from tensorfoundry.evaluation.harness import _extract_result_text, get_agent_runner


def test_extract_result_text_decision() -> None:
    result_obj = {"memo": {"next_steps": ["Step 1", "Step 2"]}}
    assert _extract_result_text("decision_agent", result_obj) == "Step 1 Step 2"


def test_extract_result_text_research() -> None:
    result_obj = {"summary": "Summary text"}
    assert _extract_result_text("research_agent", result_obj) == "Summary text"


def test_extract_result_text_refactor() -> None:
    result_obj = {"patch_result": {"diff_summary": "Diff applied", "reason": "ok"}}
    assert _extract_result_text("refactor_agent", result_obj) == "Diff applied ok"


def test_get_agent_runner_unknown_raises() -> None:
    with pytest.raises(ValueError):
        get_agent_runner("unknown_agent")
