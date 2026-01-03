from typing import Any, Dict, List, Optional
import re
import json 

_FENCE_RE = re.compile(
    r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n\s*```\s*$",
    re.DOTALL,
)

FINAL_STEPS = {"final", "final_answer", "compose_answer", "answer"}  # adjust to your agent
SKIP_STEPS  = {"extract_claims", "critique", "plan", "outline"}      # adjust to your agent


def _strip_code_fences_loose(s: str) -> str:
    s = (s or "").strip()
    if not s.startswith("```"):
        return s

    lines = s.splitlines()
    if not lines:
        return s

    lines = lines[1:]  # drop opening fence
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines).strip()

def _normalize_json_if_possible(s: str) -> str:
    try:
        obj = json.loads(s)
        # Canonical JSON to reduce formatting noise for training
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return s

def extract_step_prompt_full(events: List[Dict[str, Any]], *, step: str) -> Optional[str]:
    for ev in reversed(events):
        if ev.get("event_type") != "model_called":
            continue
        payload = ev.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        if payload.get("input_summary") != step:
            continue

        outcome = ev.get("outcome")
        if not isinstance(outcome, dict):
            continue
        res = outcome.get("result")
        if not isinstance(res, dict):
            continue

        p = res.get("prompt_full")
        if isinstance(p, str) and p.strip():
            return p.strip()
    return None


def extract_step_text_full(events: List[Dict[str, Any]], *, step: str) -> Optional[str]:
    for ev in reversed(events):
        if ev.get("event_type") != "model_called":
            continue
        payload = ev.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        if payload.get("input_summary") != step:
            continue

        outcome = ev.get("outcome")
        if not isinstance(outcome, dict):
            continue
        res = outcome.get("result")
        if not isinstance(res, dict):
            continue

        t = res.get("text_full") or res.get("text")
        if isinstance(t, str) and t.strip():
            return t.strip()
    return None

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

def extract_response(
    events: List[Dict[str, Any]],
    *,
    max_chars: Optional[int] = None,   # None = no truncation
    add_ellipsis: bool = False,        # training default: False
) -> Optional[str]:
    """
    Extraction priority:
    1. model_called where payload.input_summary == "critic_check": outcome.result.text_full
    2. model_called where payload.input_summary == "critic_check": outcome.result.text
    3. model_called where payload.input_summary == "draft_answer": outcome.result.text_full
    4. model_called where payload.input_summary == "draft_answer": outcome.result.text
    5. any model_called outcome.result.text_full / recommendation_full / text / recommendation
    6. any other string fields in outcome.result
    7. task_completed.payload.result_summary
    """
    def _clean(s: str) -> str:
        s = _strip_code_fences_loose(s).strip()
        s = _normalize_json_if_possible(s)
        if max_chars is not None and len(s) > max_chars:
            return (s[:max_chars] + "…") if add_ellipsis else s[:max_chars]
        return s

    def _get_text(res: Dict[str, Any]) -> Optional[str]:
        for key in ("text_full", "recommendation_full", "text", "recommendation"):
            v = res.get(key)
            if isinstance(v, str) and v.strip():
                return _clean(v)
        chunks = [v.strip() for v in res.values() if isinstance(v, str) and v.strip()]
        if chunks:
            return _clean("\n".join(chunks))
        return None

    # 1) Best: critic_check output
    for ev in reversed(events):
        if ev.get("event_type") != "model_called":
            continue
        payload = ev.get("payload") or {}
        step = payload.get("input_summary") if isinstance(payload, dict) else None
        if step != "critic_check":
            continue
        outcome = ev.get("outcome")
        res = (outcome or {}).get("result") if isinstance(outcome, dict) else None
        if isinstance(res, dict):
            txt = _get_text(res)
            if txt:
                return txt

    # 2) Next best: draft_answer output
    for ev in reversed(events):
        if ev.get("event_type") != "model_called":
            continue
        payload = ev.get("payload") or {}
        step = payload.get("input_summary") if isinstance(payload, dict) else None
        if step != "draft_answer":
            continue
        outcome = ev.get("outcome")
        res = (outcome or {}).get("result") if isinstance(outcome, dict) else None
        if isinstance(res, dict):
            txt = _get_text(res)
            if txt:
                return txt

    # 3) Fallback: last model_called (any step)
    for ev in reversed(events):
        if ev.get("event_type") != "model_called":
            continue
        outcome = ev.get("outcome")
        res = (outcome or {}).get("result") if isinstance(outcome, dict) else None
        if isinstance(res, dict):
            txt = _get_text(res)
            if txt:
                return txt

    return None
