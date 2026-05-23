"""Failure diagnosis helpers for evaluation artifacts."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


FAILURE_MODE_SEVERITY: Dict[str, int] = {
    "success": 0,
    "none": 0,
    "expectation_failed": 1,
    "agent_failure": 2,
    "tool_error": 3,
    "provider_error": 3,
    "scoring_error": 4,
    "trace_invalid": 5,
    "runtime_error": 5,
}

_FAILURE_EVENT_TYPES = {"task_failed", "tool_error"}
_RECOVERY_EVENT_TYPES = {"retry_requested", "tool_result", "task_completed"}
_FAILURE_STATUSES = {"failure", "failed", "error"}


def normalize_failure_mode(mode: Any, *, passed: bool = False) -> str:
    """Normalize optional failure-mode values for comparison and reports."""
    if passed:
        return "success"
    if mode is None:
        return "none"
    normalized = str(mode).strip().lower()
    if normalized in {"", "null", "none"}:
        return "none"
    return normalized


def failure_mode_severity(mode: Any, *, passed: bool = False) -> int:
    """Return an ordinal severity for a failure mode."""
    normalized = normalize_failure_mode(mode, passed=passed)
    return FAILURE_MODE_SEVERITY.get(normalized, max(FAILURE_MODE_SEVERITY.values()))


def diagnose_trace_events(
    events: List[Dict[str, Any]],
    *,
    terminal_outcome: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract failure origin and recovery evidence from trace events."""
    terminal = terminal_outcome or _find_terminal_outcome(events)
    origin = _find_failure_origin(events)
    diagnosis: Dict[str, Any] = {
        "recovery": {
            "attempted": False,
            "events": [],
            "terminal_status": _as_optional_str(terminal.get("status")),
            "terminal_reason": _as_optional_str(terminal.get("reason")),
        }
    }
    if origin is None:
        return diagnosis

    diagnosis["failure_origin"] = origin
    recovery_events = _find_recovery_events(events, after_index=int(origin["index"]))
    diagnosis["recovery"] = {
        "attempted": bool(recovery_events),
        "events": recovery_events,
        "terminal_status": _as_optional_str(terminal.get("status")),
        "terminal_reason": _as_optional_str(terminal.get("reason")),
    }
    return diagnosis


def _find_terminal_outcome(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    for ev in reversed(events):
        event_type = ev.get("event_type")
        if event_type in {"task_completed", "task_failed"}:
            outcome = ev.get("outcome")
            return outcome if isinstance(outcome, dict) else {}
    return {}


def _find_failure_origin(events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for index, ev in enumerate(events):
        event_type = str(ev.get("event_type", ""))
        stage = str(ev.get("stage", ""))
        outcome = _dict_field(ev, "outcome")
        payload = _dict_field(ev, "payload")
        status = str(outcome.get("status", "")).lower()
        tool_success = payload.get("success")

        if (
            event_type in _FAILURE_EVENT_TYPES
            or status in _FAILURE_STATUSES
            or tool_success is False
        ):
            return {
                "index": index,
                "event_type": event_type,
                "stage": stage,
                "reason": _first_str(
                    outcome.get("reason"),
                    outcome.get("failure_mode"),
                    payload.get("error"),
                    payload.get("tool_output_summary"),
                ),
                "failure_mode": _as_optional_str(outcome.get("failure_mode")),
            }
    return None


def _find_recovery_events(events: List[Dict[str, Any]], *, after_index: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for index, ev in enumerate(events[after_index + 1 :], start=after_index + 1):
        event_type = str(ev.get("event_type", ""))
        if event_type not in _RECOVERY_EVENT_TYPES and "retry" not in event_type:
            continue
        out.append(
            {
                "index": index,
                "event_type": event_type,
                "stage": str(ev.get("stage", "")),
            }
        )
    return out


def _first_str(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value
        text = str(value)
        if text.strip():
            return text
    return None


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _dict_field(obj: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = obj.get(key)
    return value if isinstance(value, dict) else {}
