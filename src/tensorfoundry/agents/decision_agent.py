"""Decision-style agent that produces a structured recommendation with logging hooks."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from tensorfoundry.logging.emitter import JsonlEmitter


class DecisionAgent:
    """A simple decision agent producing structured recommendations.

    This is intentionally model/tool-agnostic. Replace TODOs with real LLM/tool calls later.
    """

    def __init__(self, *, emitter: JsonlEmitter) -> None:
        self.emitter = emitter

    def plan(self, task: str, trace_id: str, parent_span_id: str) -> List[str]:
        plan_steps = [
            "Extract constraints and objectives",
            "Generate options",
            "Compare trade-offs",
            "Recommend with rationale and risks",
        ]
        self.emitter.emit(
            event_type="plan_created",
            stage="planner",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={"plan": plan_steps, "task": task},
        )
        return plan_steps

    def decide(
        self,
        *,
        task: str,
        constraints: Optional[List[str]],
        options: Optional[List[str]],
        trace_id: str,
        parent_span_id: str,
    ) -> Dict[str, Any]:
        """Produce a structured decision memo (placeholder logic)."""
        constraints = constraints or []
        options = options or []

        # Minimal heuristic placeholders
        assumptions = [
            "We prioritise reliability and clarity over novelty",
            "We prefer incremental delivery with measurable outcomes",
        ]
        if not options:
            options = [
                "Option A: Build the simplest working version first",
                "Option B: Build a feature-complete version before release",
                "Option C: Prototype multiple approaches and choose later",
            ]

        tradeoffs = []
        for opt in options:
            tradeoffs.append(
                {
                    "option": opt,
                    "pros": ["Fast iteration", "Lower complexity"],
                    "cons": ["May require later refactor", "Limited early capabilities"],
                }
            )

        recommendation = options[0]
        risks = [
            "Risk: underspecified requirements lead to rework",
            "Risk: evaluation missing means regressions go unnoticed",
        ]
        next_steps = [
            "Define success criteria",
            "Implement minimal evaluator",
            "Ship and iterate",
        ]

        memo = {
            "task": task,
            "constraints": constraints,
            "assumptions": assumptions,
            "options": options,
            "tradeoffs": tradeoffs,
            "recommendation": recommendation,
            "risks": risks,
            "next_steps": next_steps,
        }

        self.emitter.emit(
            event_type="model_called",
            stage="executor",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={
                "model": "provider:model-name",
                "input_summary": "Decision memo generation",
                "output_summary": "Structured recommendation",
                "content_policy": {"raw_input_logged": False, "raw_output_logged": False},
            },
            outcome={"result": {"recommendation": recommendation}},
        )

        return memo

    def run(
        self,
        task: str,
        *,
        constraints: Optional[List[str]] = None,
        options: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        root = self.emitter.emit(
            event_type="task_received",
            stage="system",
            payload={"task": task, "constraints": constraints or [], "options": options or []},
        )

        plan_steps = self.plan(task, trace_id=root.trace_id, parent_span_id=root.span_id)
        memo = self.decide(
            task=task,
            constraints=constraints,
            options=options,
            trace_id=root.trace_id,
            parent_span_id=root.span_id,
        )

        self.emitter.emit(
            event_type="task_completed",
            stage="system",
            trace_id=root.trace_id,
            parent_span_id=root.span_id,
            payload={"result_summary": memo.get("recommendation", ""), "plan": plan_steps},
            outcome={"status": "success", "reason": "decision_memo_created", "confidence": 0.7},
        )

        return {"trace_id": root.trace_id, "plan": plan_steps, "memo": memo}