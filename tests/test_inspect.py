from __future__ import annotations

import json
from pathlib import Path

from tensorfoundry.logging.events import make_event, new_span_id, new_trace_id
from tensorfoundry.logging.inspect import read_jsonl, summarise_trace


def _write_jsonl(path: Path, objs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def test_inspect_summarises_counts_and_terminal(tmp_path: Path) -> None:
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
        payload={"result_summary": "done"},
        outcome={"status": "success", "reason": "ok", "confidence": 0.9},
    ).to_dict()

    fp = tmp_path / "trace.jsonl"
    _write_jsonl(fp, [e1, e2, e3])

    events = read_jsonl(fp)
    summary = summarise_trace(events, path=fp)

    assert summary.num_events == 3
    assert summary.trace_ids == [trace_id]

    assert summary.event_type_counts["task_received"] == 1
    assert summary.event_type_counts["plan_created"] == 1
    assert summary.event_type_counts["task_completed"] == 1

    assert summary.stage_counts["system"] == 2
    assert summary.stage_counts["planner"] == 1

    assert summary.terminal_event_type == "task_completed"
    assert summary.terminal_outcome["status"] == "success"
    assert summary.terminal_outcome["reason"] == "ok"


def test_inspect_handles_missing_terminal(tmp_path: Path) -> None:
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

    events = read_jsonl(fp)
    summary = summarise_trace(events, path=fp)

    assert summary.terminal_event_type is None
    assert summary.terminal_outcome == {}