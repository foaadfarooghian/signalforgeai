"""Decision-style agent that produces a structured recommendation with logging hooks."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from tensorfoundry.logging.emitter import JsonlEmitter
from tensorfoundry.models.types import ModelMetrics
from tensorfoundry.models.registry import get_provider_for_model

class DecisionAgent:
    """A simple decision agent producing structured recommendations.

    This is intentionally model/tool-agnostic. Replace TODOs with real LLM/tool calls later.
    """

    def __init__(self, *, emitter: JsonlEmitter) -> None:
        self.emitter = emitter

    def plan(self, task: str, trace_id: str, parent_span_id: str) -> List[str]:
        """Create a simple plan and emit a planning event."""
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
    ) -> tuple[Dict[str, Any], ModelMetrics]:
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
        
        model_id = os.getenv("TENSORFOUNDRY_MODEL_ID", "ollama:ministral-3:8b")
        provider = get_provider_for_model(model_id)

        prompt = f"""Task: {task}
        Constraints: {constraints}
        Options: {options}
        Return a short recommendation + next steps."""
        out = provider.generate(prompt=prompt, model_id=model_id, task_type="decision")

        # Use provider output to populate next_steps (so benchmark learns without hardcoding)
        next_steps = [out.text]

        memo: Dict[str, Any] = {
            "task": task,
            "constraints": constraints,
            "assumptions": assumptions,
            "options": options,
            "tradeoffs": tradeoffs,
            "recommendation": recommendation,
            "risks": risks,
            "next_steps": next_steps,
        }

        metrics = {
            "latency_ms": out.metrics.latency_ms,
            "cost_usd": out.metrics.cost_usd,
            "extra": out.metrics.extra
        }
        self.emitter.emit(
            event_type="model_called",
            stage="executor",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={
                "model": model_id,
                "input_summary": "Decision memo generation",
                "output_summary": "Structured recommendation",
                "content_policy": {"raw_input_logged": False, "raw_output_logged": False},
            },
            metrics=metrics,
            outcome={"result": {"recommendation": recommendation}},
        )

        return memo, out.metrics

    def run(
        self,
        task: str,
        *,
        constraints: Optional[List[str]] = None,
        options: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run the full decision workflow and return a memo plus trace metadata."""
        root = self.emitter.emit(
            event_type="task_received",
            stage="system",
            payload={"task": task, "constraints": constraints or [], "options": options or []},
        )

        plan_steps = self.plan(task, trace_id=root.trace_id, parent_span_id=root.span_id)
        memo, model_metrics = self.decide(
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
            metrics={
                "latency_ms": model_metrics.latency_ms,
                "cost_usd": model_metrics.cost_usd,
                
            },
            outcome={"status": "success", "reason": "decision_memo_created", "confidence": 0.7},
        )

        return {"trace_id": root.trace_id, "plan": plan_steps, "memo": memo}
