from typing import Any, Dict, List, Optional


def extract_instruction(events: List[Dict[str, Any]]) -> Optional[str]:
    task = None
    constraints = None
    options = None

    for ev in events:
        if ev.get("event_type") == "task_received":
            payload = ev.get("payload") or {}
            if isinstance(payload, dict):
                task = payload.get("task")
                constraints = payload.get("constraints")
                options = payload.get("options")
            break

    if not task:
        return None

    parts = [f"Task: {task}"]
    if constraints:
        parts.append(f"Constraints: {constraints}")
    if options:
        parts.append(f"Options: {options}")
    return "\n".join(parts)


def extract_response(events: List[Dict[str, Any]], *, max_chars: int = 2000) -> Optional[str]:
    """
    Extraction priority:
    1. model_called.outcome.result.text (preferred)
    2. model_called.outcome.result.recommendation
    3. any other string fields in outcome.result
    4. task_completed.payload.result_summary
    """
    def _clean(s: str) -> str:
        s = s.strip()
        return (s[:max_chars] + "…") if len(s) > max_chars else s

    for ev in reversed(events):
        if ev.get("event_type") != "model_called":
            continue

        outcome = ev.get("outcome")
        if not isinstance(outcome, dict):
            continue

        res = outcome.get("result")
        if not isinstance(res, dict):
            continue

        text = res.get("text")
        if isinstance(text, str) and text.strip():
            return _clean(text)

        rec = res.get("recommendation")
        if isinstance(rec, str) and rec.strip():
            return _clean(rec)

        chunks = [v.strip() for v in res.values() if isinstance(v, str) and v.strip()]
        if chunks:
            return _clean("\n".join(chunks))

    for ev in reversed(events):
        if ev.get("event_type") != "task_completed":
            continue
        payload = ev.get("payload")
        if not isinstance(payload, dict):
            continue
        rs = payload.get("result_summary")
        if isinstance(rs, str) and rs.strip():
            return _clean(rs)

    return None