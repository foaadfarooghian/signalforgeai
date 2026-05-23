from __future__ import annotations

from dataclasses import dataclass
from typing import List

from signalforgeai.agents.research_agent import ResearchAgent
from signalforgeai.orchestration.pec import CritiqueResult, State


@dataclass
class ResearchPlanner:
    """Planner adapter that delegates planning to a ResearchAgent."""
    agent: ResearchAgent

    def plan(self, state: State, *, trace_id: str, parent_span_id: str) -> State:
        """Attach plan steps to state using the underlying agent."""
        plan_steps = self.agent.plan(state["task"], trace_id=trace_id, parent_span_id=parent_span_id)
        state["plan"] = plan_steps
        return state


@dataclass
class ResearchExecutor:
    """Executor adapter that runs search + summarization on a ResearchAgent."""
    agent: ResearchAgent

    def execute(self, state: State, *, trace_id: str, parent_span_id: str) -> State:
        """Populate findings and result in state using the agent."""
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
        """Return a critique result and optional mutated state."""
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
