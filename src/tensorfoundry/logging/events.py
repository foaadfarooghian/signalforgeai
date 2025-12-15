"""Event primitives for TensorFoundry logging.

Defines the canonical event shape and helpers for generating IDs and timestamps.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def new_trace_id() -> str:
    """Generate a new trace identifier."""
    return uuid.uuid4().hex


def new_span_id() -> str:
    """Generate a new span identifier."""
    return uuid.uuid4().hex


def utc_now_iso() -> str:
    """Return an ISO-8601 timestamp with timezone (UTC)."""
    dt = datetime.now(timezone.utc).replace(tzinfo=timezone.utc)
    return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _truncate_str(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."


def _sanitize_value(value: Any, max_string: int = 500, max_items: int = 50) -> Any:
    """Keep payloads small/safe by truncating and stringifying unknown objects."""
    if isinstance(value, str):
        return _truncate_str(value, max_string)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        sanitized: Dict[str, Any] = {}
        for idx, (key, val) in enumerate(value.items()):
            if idx >= max_items:
                sanitized["__truncated__"] = f"{len(value) - max_items} more entries"
                break
            sanitized[str(key)] = _sanitize_value(val, max_string, max_items)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        seq = list(value)[:max_items]
        sanitized_list = [_sanitize_value(item, max_string, max_items) for item in seq]
        if len(value) > max_items:
            sanitized_list.append(f"...truncated {len(value) - max_items} items")
        return sanitized_list
    # Fallback: represent objects as truncated repr strings
    return _truncate_str(repr(value), max_string)


def sanitize_payload(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not payload:
        return {}
    sanitized = _sanitize_value(payload)
    if isinstance(sanitized, dict):
        return sanitized
    return {"value": sanitized}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Event:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    timestamp: str
    agent: Dict[str, str]
    stage: str
    event_type: str
    payload: Dict[str, Any]
    metrics: Dict[str, Any]
    outcome: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "timestamp": self.timestamp,
            "agent": self.agent,
            "stage": self.stage,
            "event_type": self.event_type,
            "payload": sanitize_payload(self.payload),
            "metrics": sanitize_payload(self.metrics),
            "outcome": sanitize_payload(self.outcome),
        }


def make_event(
    *,
    trace_id: str,
    span_id: str,
    agent_name: str,
    agent_version: str,
    stage: str,
    event_type: str,
    parent_span_id: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    outcome: Optional[Dict[str, Any]] = None,
) -> Event:
    """Factory for a well-formed event."""
    return Event(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        timestamp=utc_now_iso(),
        agent={"name": agent_name, "version": agent_version},
        stage=stage,
        event_type=event_type,
        payload=payload or {},
        metrics=metrics or {},
        outcome=outcome or {},
    )
