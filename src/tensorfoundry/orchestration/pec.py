"""Planner → Executor → Critic orchestrator (PEC).

This module implements a minimal, schema-aligned orchestration loop:
- planner creates a plan (or updates state)
- executor executes actions (tools/models)
- critic evaluates outcome and requests retries with state mutation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Protocol

from tensorfoundry.logging.emitter import JsonlEmitter


State = Dict[str, Any]


@dataclass(frozen=True)
class CritiqueResult:
    done: bool
    status: Literal["success", "partial", "failure"]
    reason: str
    confidence: Optional[float] = None
    retry: bool = False
    mutated_state: Optional[State] = None


class Planner(Protocol):
    def plan(self, state: State, *, trace_id: str, parent_span_id: str) -> State: ...


class Executor(Protocol):
    def execute(self, state: State, *, trace_id: str, parent_span_id: str) -> State: ...


class Critic(Protocol):
    def critique(self, state: State, *, trace_id: str, parent_span_id: str) -> CritiqueResult: ...


class PECOrchestrator:
    """Generic Planner→Executor→Critic orchestrator."""

    def __init__(
        self,
        *,
        planner: Planner,
        executor: Executor,
        critic: Critic,
        emitter: JsonlEmitter,
        max_retries: int = 2,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.critic = critic
        self.emitter = emitter
        self.max_retries = max_retries

    def run(self, task: str, *, initial_state: Optional[State] = None) -> State:
        """Run a task through planner→executor→critic loop with retries."""
        state: State = dict(initial_state or {})
        state.setdefault("task", task)
        state.setdefault("attempt", 0)
        state.setdefault("errors", [])
        state.setdefault("artifacts", {})
        state.setdefault("result", None)

        # Root event
        root = self.emitter.emit(
            event_type="task_received",
            stage="system",
            payload={"task": task, "initial_state_keys": sorted(state.keys())},
        )
        trace_id = root.trace_id
        root_span = root.span_id

        # Planner
        planned_state = self.planner.plan(state, trace_id=trace_id, parent_span_id=root_span)
        self.emitter.emit(
            event_type="stage_completed",
            stage="system",
            trace_id=trace_id,
            parent_span_id=root_span,
            payload={"stage": "planner", "state_keys": sorted(planned_state.keys())},
        )

        current = planned_state

        # Loop: execute → critique → maybe retry
        retries = 0
        while True:
            current = self.executor.execute(current, trace_id=trace_id, parent_span_id=root_span)
            self.emitter.emit(
                event_type="stage_completed",
                stage="system",
                trace_id=trace_id,
                parent_span_id=root_span,
                payload={"stage": "executor", "attempt": current.get("attempt", 0)},
            )

            critique = self.critic.critique(current, trace_id=trace_id, parent_span_id=root_span)

            # Terminal?
            if critique.done:
                terminal_type = "task_completed" if critique.status != "failure" else "task_failed"
                self.emitter.emit(
                    event_type=terminal_type,
                    stage="system",
                    trace_id=trace_id,
                    parent_span_id=root_span,
                    payload={
                        "result_summary": _safe_summary(current.get("result")),
                        "attempts": current.get("attempt", 0),
                    },
                    outcome={
                        "status": critique.status,
                        "reason": critique.reason,
                        "confidence": critique.confidence,
                    },
                )
                return current

            # Retry?
            if not critique.retry or retries >= self.max_retries:
                self.emitter.emit(
                    event_type="task_failed",
                    stage="system",
                    trace_id=trace_id,
                    parent_span_id=root_span,
                    payload={
                        "result_summary": _safe_summary(current.get("result")),
                        "attempts": current.get("attempt", 0),
                    },
                    outcome={
                        "status": "failure",
                        "reason": critique.reason if critique.reason else "max_retries_exceeded",
                        "confidence": critique.confidence,
                    },
                )
                return current

            retries += 1
            mutated = critique.mutated_state or current
            mutated["attempt"] = int(mutated.get("attempt", 0)) + 1

            self.emitter.emit(
                event_type="retry_requested",
                stage="system",
                trace_id=trace_id,
                parent_span_id=root_span,
                payload={
                    "retry_number": retries,
                    "reason": critique.reason,
                    "mutation_keys": sorted(mutated.keys()),
                },
                outcome={"status": "partial"},
            )
            current = mutated


def _safe_summary(value: Any, max_len: int = 240) -> str:
    if value is None:
        return ""
    s = str(value)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."