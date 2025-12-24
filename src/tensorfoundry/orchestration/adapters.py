from __future__ import annotations

from dataclasses import dataclass
from typing import List

from tensorfoundry.agents.research_agent import ResearchAgent
from tensorfoundry.orchestration.pec import CritiqueResult, State


@dataclass
class ResearchPlanner:
    agent: ResearchAgent

    def plan(self, state: State, *, trace_id: str, parent_span_id: str) -> State:
        plan_steps = self.agent.plan(state["task"], trace_id=trace_id, parent_span_id=parent_span_id)
        state["plan"] = plan_steps
        return state


@dataclass
class ResearchExecutor:
    agent: ResearchAgent

    def execute(self, state: State, *, trace_id: str, parent_span_id: str) -> State:
        findings: List[str] = self.agent.search(state["task"], trace_id=trace_id, parent_span_id=parent_span_id)
        summary: str = self.agent.summarize(findings, trace_id=trace_id, parent_span_id=parent_span_id)
        state["artifacts"]["findings"] = findings
        state["result"] = summary
        return state


@dataclass
class SimpleCritic:
    """A tiny critic that succeeds if result is non-empty; otherwise retries once."""
    min_len: int = 10

    def critique(self, state: State, *, trace_id: str, parent_span_id: str) -> CritiqueResult:
        result = state.get("result") or ""
        if isinstance(result, str) and len(result.strip()) >= self.min_len:
            return CritiqueResult(done=True, status="success", reason="result_non_empty", confidence=0.7)
        # Retry by mutating state (example: add constraint)
        mutated = dict(state)
        mutated.setdefault("constraints", [])
        mutated["constraints"] = list(mutated["constraints"]) + ["be more specific"]
        return CritiqueResult(
            done=False,
            status="partial",
            reason="result_too_short",
            retry=True,
            mutated_state=mutated,
            confidence=0.3,
        )
