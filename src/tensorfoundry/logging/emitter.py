"""JSONL emitter implementing the TensorFoundry logging schema."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .events import Event, make_event, new_span_id, new_trace_id


class JsonlEmitter:
    """Emit events as JSONL lines."""

    def __init__(
        self,
        file_path: Path | str,
        *,
        agent_name: str,
        agent_version: str,
        default_stage: str = "executor"
    ) -> None:
        self.file_path = Path(file_path)
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.default_stage = default_stage
        self._active_trace_id: Optional[str] = None

    def new_trace_id(self) -> str:
        """Generate a fresh trace ID (does not mutate active trace)."""
        return new_trace_id()

    def new_span_id(self) -> str:
        """Generate a fresh span ID."""
        return new_span_id()
    
    def start_trace(self) -> str:
        """Start and remember a new active trace."""
        self._active_trace_id = self.new_trace_id()
        return self._active_trace_id

    def emit(
        self,
        *,
        event_type: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        stage: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        outcome: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """Write a single event to the JSONL file."""
        VALID_STAGES = {"planner", "executor", "critic", "tool", "system"}
        resolved_stage = stage or self.default_stage
        if resolved_stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage: {resolved_stage!r}. Must be one of {VALID_STAGES}")
        
        if trace_id is None:
            trace_id = self._active_trace_id or self.new_trace_id()
        
        event = make_event(
            trace_id=trace_id,
            span_id=span_id or self.new_span_id(),
            parent_span_id=parent_span_id,
            agent_name=self.agent_name,
            agent_version=self.agent_version,
            stage=resolved_stage,
            event_type=event_type,
            payload=payload,
            metrics=metrics,
            outcome=outcome,
        )
        self._write(event)
        return event
    
    def __enter__(self):
        """Open the JSONL file for streaming writes."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.file_path.open("a", encoding="utf-8")
        return self
    
    def __exit__(self, exc_type, exc, tb):
        """Close any open file handle."""
        if getattr(self, "_fp", None):
            self._fp.close()
            self._fp = None

    def _write(self, event: Event) -> None:
        """Append a single event line to the JSONL file (streaming-safe)."""
        fp = getattr(self, "_fp", None)
        if fp is None:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.file_path.open("a", encoding="utf-8") as fp2:
                fp2.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        else:
            fp.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            fp.flush()
