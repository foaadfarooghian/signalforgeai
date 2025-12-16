"""Simple research-style agent with logging hooks."""
from __future__ import annotations
from tensorfoundry.logging import JsonlEmitter
from typing import Any, Dict, List


class ResearchAgent:
    """Placeholder research agent that records actions."""

    def __init__(self, *, emitter: JsonlEmitter) -> None:
        self.journal: List[str] = []
        self.emitter = emitter

    def plan(self, query: str, trace_id: str, parent_span_id: str) -> List[str]:
        """Sketch a simple plan and log the intent."""
        plan_steps = [
            "search for relevant information",
            "Extract key facts",
            "Synthesize answer",
        ]
        self.journal.append("planned steps")
        self.emitter.emit(
            event_type="plan_created",
            stage="planner",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={"plan": plan_steps, "query": query},
        )
        return plan_steps

    def search(self, query: str, trace_id: str, parent_span_id: str) -> List[str]:
        """Call a placeholder search tool and log request/response."""
        self.journal.append(f"searched for query: {query}")

        self.emitter.emit(
            event_type="tool_called",
            stage="tool",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={"tool_name": "web_search", "tool_input": {"query": query}},
        )

        results = [f"result for {query} (placeholder)"]

        self.emitter.emit(
            event_type="tool_result",
            stage="tool",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={
                "tool_name": "web_search",
                "success": True,
                "tool_output_summary": "Returned 1 placeholder result",
            },
            outcome={"results": results},
        )
        return results

    def summarize(self, findings: List[str], trace_id: str, parent_span_id: str) -> str:
        """Summarize findings using a placeholder model call."""
        summary = " | ".join(findings)
        self.journal.append("summarized findings")

        self.emitter.emit(
            event_type="model_called",
            stage="executor",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={
                "model": "provider:model-name",
                "input_summary": "Summarise findings",
                "output_summary": "Produced summary",
                "content_policy": {"raw_input_logged": False, "raw_output_logged": False},
            },
            outcome={"summary": summary},
        )
        return summary

    def run(self, query: str) -> Dict[str, Any]:
        """Execute the simple research workflow end-to-end."""
        root = self.emitter.emit(
            event_type="task_received",
            stage="system",
            payload={"task": query},
        )

        plan_steps = self.plan(query, trace_id=root.trace_id, parent_span_id=root.span_id)
        findings = self.search(query, trace_id=root.trace_id, parent_span_id=root.span_id)
        summary = self.summarize(findings, trace_id=root.trace_id, parent_span_id=root.span_id)

        self.emitter.emit(
            event_type="task_completed",
            stage="system",
            trace_id=root.trace_id,
            parent_span_id=root.span_id,
            payload={"result_summary": summary, "plan": plan_steps},
            outcome={"status": "success", "confidence": 0.8},
        )

        return {
            "query": query,
            "plan": plan_steps,
            "findings": findings,
            "summary": summary,
            "journal": self.journal,
            "trace_id": root.trace_id,
        }
