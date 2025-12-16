from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorfoundry.logging.events import make_event, new_span_id, new_trace_id
from tensorfoundry.logging.validate import validate_trace_file


def _write_jsonl(path: Path, objs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def test_valid_trace_passes(tmp_path: Path) -> None:
    trace_id = new_trace_id()
    root_span = new_span_id()

    e1 = make_event(
        trace_id=trace_id,
        span_id=root_span,
        agent_name="research_agent",
        agent_version="0.1.0",
        stage="system",
        event_type="task_received",
        payload={"task": "hello"},
    ).to_dict()

    e2 = make_event(
        trace_id=trace_id,
        span_id=new_span_id(),
        parent_span_id=root_span,
        agent_name="research_agent",
        agent_version="0.1.0",
        stage="planner",
        event_type="plan_created",
        payload={"plan": ["a", "b"]},
    ).to_dict()

    e3 = make_event(
        trace_id=trace_id,
        span_id=new_span_id(),
        parent_span_id=root_span,
        agent_name="research_agent",
        agent_version="0.1.0",
        stage="system",
        event_type="task_completed",
        outcome={"status": "success", "confidence": 0.8},
        payload={"result_summary": "done"},
    ).to_dict()

    fp = tmp_path / "trace.jsonl"
    _write_jsonl(fp, [e1, e2, e3])

    issues = validate_trace_file(fp)
    assert issues == []


def test_missing_terminal_event_fails(tmp_path: Path) -> None:
    trace_id = new_trace_id()

    e1 = make_event(
        trace_id=trace_id,
        span_id=new_span_id(),
        agent_name="research_agent",
        agent_version="0.1.0",
        stage="system",
        event_type="task_received",
        payload={"task": "hello"},
    ).to_dict()

    fp = tmp_path / "trace.jsonl"
    _write_jsonl(fp, [e1])

    issues = validate_trace_file(fp)
    codes = {i.code for i in issues}
    assert "missing_terminal_event" in codes


def test_invalid_stage_fails(tmp_path: Path) -> None:
    trace_id = new_trace_id()

    ev = make_event(
        trace_id=trace_id,
        span_id=new_span_id(),
        agent_name="research_agent",
        agent_version="0.1.0",
        stage="not_a_stage",
        event_type="task_received",
        payload={"task": "hello"},
    ).to_dict()

    # Ensure we still include a terminal event so the failure is about stage
    terminal = make_event(
        trace_id=trace_id,
        span_id=new_span_id(),
        agent_name="research_agent",
        agent_version="0.1.0",
        stage="system",
        event_type="task_completed",
        outcome={"status": "success"},
    ).to_dict()

    fp = tmp_path / "trace.jsonl"
    _write_jsonl(fp, [ev, terminal])

    issues = validate_trace_file(fp)
    codes = {i.code for i in issues}
    assert "invalid_stage" in codes


def test_multiple_trace_ids_fails_by_default(tmp_path: Path) -> None:
    t1 = new_trace_id()
    t2 = new_trace_id()

    e1 = make_event(
        trace_id=t1,
        span_id=new_span_id(),
        agent_name="research_agent",
        agent_version="0.1.0",
        stage="system",
        event_type="task_received",
        payload={"task": "hello"},
    ).to_dict()

    e2 = make_event(
        trace_id=t2,
        span_id=new_span_id(),
        agent_name="research_agent",
        agent_version="0.1.0",
        stage="system",
        event_type="task_completed",
        outcome={"status": "success"},
    ).to_dict()

    fp = tmp_path / "trace.jsonl"
    _write_jsonl(fp, [e1, e2])

    issues = validate_trace_file(fp)
    codes = {i.code for i in issues}
    assert "multiple_trace_ids" in codes


def test_multiple_trace_ids_can_be_allowed(tmp_path: Path) -> None:
    t1 = new_trace_id()
    t2 = new_trace_id()

    e1 = make_event(
        trace_id=t1,
        span_id=new_span_id(),
        agent_name="research_agent",
        agent_version="0.1.0",
        stage="system",
        event_type="task_received",
        payload={"task": "hello"},
    ).to_dict()

    e2 = make_event(
        trace_id=t2,
        span_id=new_span_id(),
        agent_name="research_agent",
        agent_version="0.1.0",
        stage="system",
        event_type="task_completed",
        outcome={"status": "success"},
    ).to_dict()

    fp = tmp_path / "trace.jsonl"
    _write_jsonl(fp, [e1, e2])

    issues = validate_trace_file(fp, require_single_trace_id=False)
    # Still should be valid because there is a terminal event and fields are fine
    assert issues == []


def test_invalid_json_line_fails(tmp_path: Path) -> None:
    fp = tmp_path / "trace.jsonl"
    fp.write_text('{"a": 1}\n{this is not json}\n', encoding="utf-8")

    issues = validate_trace_file(fp)
    codes = {i.code for i in issues}
    assert "invalid_json" in codes


def test_terminal_event_missing_outcome_status_fails(tmp_path: Path) -> None:
    trace_id = new_trace_id()

    e1 = make_event(
        trace_id=trace_id,
        span_id=new_span_id(),
        agent_name="research_agent",
        agent_version="0.1.0",
        stage="system",
        event_type="task_received",
        payload={"task": "hello"},
    ).to_dict()

    # terminal event but no outcome.status
    e2 = make_event(
        trace_id=trace_id,
        span_id=new_span_id(),
        agent_name="research_agent",
        agent_version="0.1.0",
        stage="system",
        event_type="task_completed",
        outcome={"confidence": 0.9},
    ).to_dict()

    fp = tmp_path / "trace.jsonl"
    _write_jsonl(fp, [e1, e2])

    issues = validate_trace_file(fp)
    codes = {i.code for i in issues}
    assert "missing_terminal_status" in codes