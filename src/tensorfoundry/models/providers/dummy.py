from __future__ import annotations

import random
import time
from typing import Optional
from ..types import ModelOutput, ModelMetrics

class DummyProvider:
    """
    Dummy provider for testing orchestration/eval/learning without external LLMs.

    model_id:
      - dummy_good: consistently "good" keyword-rich responses
      - dummy_mid: 50/50 good vs bad
      - dummy_bad: consistently "bad" generic responses
    """

    def generate(self, *, prompt: str, model_id: str, task_type: Optional[str] = None) -> ModelOutput:
        t0 = time.time()

        # Use prompt content to tailor keywords for your benchmark suites
        p = (prompt or "").lower()

        good = self._good_text(p, task_type=task_type)
        bad = self._bad_text(task_type=task_type)

        if model_id == "dummy_good":
            text = good
        elif model_id == "dummy_mid":
            text = good if random.random() < 0.5 else bad
        else:
            text = bad

        latency_ms = int((time.time() - t0) * 1000)
        return ModelOutput(
            text=text,
            metrics=ModelMetrics(latency_ms=latency_ms, cost_usd=0.0, extra={}),
        )

    def _good_text(self, p: str, *, task_type: Optional[str]) -> str:
        # Make decision benchmark strongly separable
        if task_type == "decision":
            return (
                "Next steps: expand benchmark tasks; add learning-from-logs converters so logs become a dataset; "
                "introduce schema versioning with additive compat; protect prod and merge from dev; "
                "keep ruff lint and mypy type checks as fast quality gates."
            )

        # Research benchmark keyword targets
        if "agentic workflow" in p:
            return "An agentic workflow is a plan-driven workflow where an agent uses tools and state to execute tasks."
        if "structured logging" in p:
            return "Structured logging produces traces that help debug, evaluate, and improve agent behaviour."
        if "planner" in p and "executor" in p and "critic" in p:
            return "Planner proposes steps, Executor runs tools, Critic validates outcomes and triggers retry/validate."
        if "evaluation" in p and "essential" in p:
            return "Evaluation tracks success, cost, latency, detects regression, and makes systems robust."
        if "logs are the dataset" in p:
            return "Logs and traces become a dataset for learning, so systems improve over time."

        # Safe fallback for non-benchmark prompts
        return "Agents use tools, planning, and state; traces support evaluation and improvement."


    def _bad_text(self, *, task_type: Optional[str]) -> str:
        # Avoid decision benchmark keywords explicitly
        if task_type == "decision":
            return "Next steps: move fast; ship features; reduce overhead; focus on delivery."

        return "This is a generic response. It provides minimal detail."