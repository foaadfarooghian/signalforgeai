from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from signalforgeai.logging.emitter import JsonlEmitter
from signalforgeai.logging.validate import validate_trace_file
from signalforgeai.orchestration.pec import CritiqueResult, PECOrchestrator, State


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


class DummyPlanner:
    def __init__(self, *, plan_steps: List[str] | None = None) -> None:
        self.plan_steps = plan_steps or ["step1", "step2"]

    def plan(self, state: State, *, trace_id: str, parent_span_id: str) -> State:
        state["plan"] = self.plan_steps
        return state


class DummyExecutor:
    """Executor that returns a fixed result (optionally empty)."""

    def __init__(self, *, result: str) -> None:
        self.result = result

    def execute(self, state: State, *, trace_id: str, parent_span_id: str) -> State:
        state["result"] = self.result
        return state


class CriticSucceeds:
    def critique(self, state: State, *, trace_id: str, parent_span_id: str) -> CritiqueResult:
        return CritiqueResult(done=True, status="success", reason="ok", confidence=0.9)


class CriticRetriesOnceThenSucceeds:
    """First critique asks for retry; second critique succeeds."""

    def critique(self, state: State, *, trace_id: str, parent_span_id: str) -> CritiqueResult:
        attempt = int(state.get("attempt", 0))
        if attempt == 0:
            mutated = dict(state)
            mutated["constraints"] = ["be more specific"]
            return CritiqueResult(
                done=False,
                status="partial",
                reason="needs_retry",
                confidence=0.2,
                retry=True,
                mutated_state=mutated,
            )
        return CritiqueResult(done=True, status="success", reason="ok_after_retry", confidence=0.7)


class CriticAlwaysRetries:
    def critique(self, state: State, *, trace_id: str, parent_span_id: str) -> CritiqueResult:
        mutated = dict(state)
        mutated["constraints"] = list(mutated.get("constraints", [])) + ["retry_again"]
        return CritiqueResult(
            done=False,
            status="partial",
            reason="still_bad",
            confidence=0.1,
            retry=True,
            mutated_state=mutated,
        )


def _make_emitter(tmp_path: Path) -> tuple[JsonlEmitter, str, Path]:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    emitter = JsonlEmitter(
        logs_dir / "temp.jsonl",
        agent_name="orchestrator",
        agent_version="0.1.0",
        default_stage="system",
    )

    trace_id = emitter.start_trace()
    trace_path = logs_dir / f"{trace_id}.jsonl"
    emitter.file_path = trace_path
    return emitter, trace_id, trace_path


def test_orchestrator_emits_valid_trace_and_completes(tmp_path: Path) -> None:
    emitter, trace_id, trace_path = _make_emitter(tmp_path)

    planner = DummyPlanner()
    executor = DummyExecutor(result="hello world")
    critic = CriticSucceeds()

    with emitter:
        orch = PECOrchestrator(planner=planner, executor=executor, critic=critic, emitter=emitter)
        state = orch.run("test task")

    assert state["task"] == "test task"
    assert state["result"] == "hello world"

    # Trace file exists and is schema-valid
    assert trace_path.exists()
    issues = validate_trace_file(trace_path)
    assert issues == []

    # Ensure terminal event exists
    events = _read_jsonl(trace_path)
    terminal_types = {e["event_type"] for e in events}
    assert "task_completed" in terminal_types or "task_failed" in terminal_types


def test_orchestrator_retries_and_increments_attempt(tmp_path: Path) -> None:
    emitter, trace_id, trace_path = _make_emitter(tmp_path)

    planner = DummyPlanner()
    # executor returns something that "could" be ok, critic forces retry once anyway
    executor = DummyExecutor(result="short")
    critic = CriticRetriesOnceThenSucceeds()

    with emitter:
        orch = PECOrchestrator(
            planner=planner,
            executor=executor,
            critic=critic,
            emitter=emitter,
            max_retries=2,
        )
        state = orch.run("retry task")

    # After one retry, attempt should be 1
    assert int(state.get("attempt", 0)) == 1
    assert "constraints" in state

    issues = validate_trace_file(trace_path)
    assert issues == []

    events = _read_jsonl(trace_path)
    event_types = [e["event_type"] for e in events]
    assert "retry_requested" in event_types
    assert "task_completed" in event_types


def test_orchestrator_fails_when_max_retries_exceeded(tmp_path: Path) -> None:
    emitter, trace_id, trace_path = _make_emitter(tmp_path)

    planner = DummyPlanner()
    executor = DummyExecutor(result="")
    critic = CriticAlwaysRetries()

    with emitter:
        orch = PECOrchestrator(
            planner=planner,
            executor=executor,
            critic=critic,
            emitter=emitter,
            max_retries=1,
        )
        state = orch.run("fail task")

    # With max_retries=1, attempt should be 1 after one retry request
    assert int(state.get("attempt", 0)) == 1

    issues = validate_trace_file(trace_path)
    assert issues == []

    events = _read_jsonl(trace_path)
    event_types = [e["event_type"] for e in events]
    assert "retry_requested" in event_types
    assert "task_failed" in event_types