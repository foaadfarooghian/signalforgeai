"""Scoring functions for evaluation suites."""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
import os
import re
from typing import Any, Callable, Dict, List

ScoreResult = tuple[bool, float, List[str]]
ScorerFn = Callable[["ScoringContext"], ScoreResult]


@dataclass(frozen=True)
class ScoringContext:
    """Input context passed to suite scoring functions."""
    suite_name: str
    case: Dict[str, Any]
    result_obj: Dict[str, Any]
    terminal_outcome: Dict[str, Any]
    result_text: str
    events: List[Dict[str, Any]]
    scoring_params: Dict[str, Any] = field(default_factory=dict)


def default_v0(ctx: ScoringContext) -> ScoreResult:
    """Default scoring: status match + optional contains_any keyword checks."""
    params = ctx.scoring_params or {}
    status_weight = _clamp01(_as_float(params.get("status_weight"), 0.7))
    contains_weight = _clamp01(_as_float(params.get("contains_weight"), 0.3))
    if "min_score" in params:
        min_score = _clamp01(_as_float(params.get("min_score"), status_weight))
    else:
        min_score = status_weight

    expect = ctx.case.get("expect", {})
    if not isinstance(expect, dict):
        expect = {}
    return _score_case(
        expect=expect,
        terminal_outcome=ctx.terminal_outcome,
        result_text=ctx.result_text,
        status_weight=status_weight,
        contains_weight=contains_weight,
        min_score=min_score,
    )


def synth_v1(ctx: ScoringContext) -> ScoreResult:
    """Synth v1 scoring: schema + citation grounding + conflict/noise handling."""
    case_id = str(ctx.case.get("id", ""))
    inputs = ctx.case.get("inputs", {})
    sources = inputs.get("sources", []) if isinstance(inputs, dict) else []
    if not isinstance(sources, list):
        sources = []
    return _score_synth_case(case_id=case_id, result_obj=ctx.result_obj, sources=sources)


def _score_case(
    *,
    expect: Dict[str, Any],
    terminal_outcome: Dict[str, Any],
    result_text: str,
    status_weight: float = 0.7,
    contains_weight: float = 0.3,
    min_score: float = 0.7,
) -> ScoreResult:
    """Score a case by status match and optional substring checks."""
    notes: List[str] = []
    score = 0.0

    expected_status = expect.get("status")
    actual_status = terminal_outcome.get("status")

    if expected_status is not None:
        if actual_status == expected_status:
            score += status_weight
        else:
            notes.append(f"status mismatch: expected {expected_status!r}, got {actual_status!r}")

    contains_any = expect.get("contains_any", [])
    if contains_any:
        if any(token.lower() in (result_text or "").lower() for token in contains_any):
            score += contains_weight
        else:
            notes.append(f"result missing any of: {contains_any}")

    score = _clamp01(score)
    passed = score >= min_score
    return passed, score, notes


def _score_synth_case(
    *,
    case_id: str,
    result_obj: Dict[str, Any],
    sources: List[Dict[str, Any]],
) -> ScoreResult:
    """
    Continuous score 0..1:
    - schema validity (0.2)
    - citations present + grounded (0.3)
    - conflict handling (0.2)   [only for conflict cases]
    - relevance/noise handling (0.2) [only for noisy cases]
    - clarity/length (0.1)
    """
    notes: List[str] = []
    score = 0.0

    def _as_str(value: Any) -> str:
        return value if isinstance(value, str) else ""

    def _as_list(value: Any) -> List[Any]:
        return value if isinstance(value, list) else []

    res = (result_obj or {}).get("result")
    if not isinstance(res, dict):
        return False, 0.0, ["missing result dict"]

    required = ["answer", "citations", "assumptions", "risks", "confidence"]
    if all(k in res for k in required):
        score += 0.2
    else:
        notes.append("schema missing keys")

    answer = _as_str(res.get("answer"))
    citations = _as_list(res.get("citations"))

    # Build source maps
    src_by_id = {s.get("source_id"): s for s in sources if isinstance(s, dict)}
    src_text_by_id = {sid: _as_str(src_by_id[sid].get("text")) for sid in src_by_id}
    src_title_by_id = {sid: _as_str(src_by_id[sid].get("title")) for sid in src_by_id}

    # --- citations present + grounded (0.3)
    if not citations:
        notes.append("no citations")
    else:
        grounded = 0
        invalid_sid = 0
        bad_quote = 0
        placeholder = 0

        for c in citations:
            if not isinstance(c, dict):
                bad_quote += 1
                continue
            sid = c.get("source_id")
            quote = c.get("quote") or ""
            if sid not in src_by_id:
                invalid_sid += 1
                continue
            # reject placeholders like "[S1 ...]" that aren't verbatim source text
            if "[" in quote and "]" in quote:
                placeholder += 1
            if not _quote_in_source(quote, src_text_by_id.get(sid, "")):
                bad_quote += 1
            else:
                grounded += 1

        if grounded > 0 and invalid_sid == 0 and bad_quote == 0 and placeholder == 0:
            score += 0.3
        else:
            if invalid_sid:
                notes.append("invalid citation source_id")
            if bad_quote:
                notes.append("ungrounded_or_empty_quotes")
            if placeholder:
                notes.append("placeholder_quotes")

    # --- conflict handling (0.2) for conflict case
    is_conflict = "conflict" in case_id
    if is_conflict:
        # Must cite both S1 and S2
        cited_ids = {c.get("source_id") for c in citations if isinstance(c, dict)}
        if {"S1", "S2"}.issubset(cited_ids) and any(w in answer.lower() for w in ["uncertain", "disagree", "conflict", "mixed"]):
            score += 0.2
        else:
            notes.append("conflict_handling_missing")

    # --- noisy handling (0.2) for noisy case
    is_noisy = "noisy" in case_id
    if is_noisy:
        # In your suite, S2 is explicitly titled "Irrelevant"
        irrelevant_ids = {sid for sid, title in src_title_by_id.items() if title.lower().strip() == "irrelevant"}
        cited_ids = {c.get("source_id") for c in citations if isinstance(c, dict)}

        # Reward: does NOT cite irrelevant + answer does not mention obvious irrelevant topic
        mentions_irrelevant = any(tok in answer.lower() for tok in ["banana", "bananas", "strawberry", "strawberries", "berry", "berries"])
        if not (cited_ids & irrelevant_ids) and not mentions_irrelevant:
            score += 0.2
        else:
            notes.append("irrelevant_source_leakage")

    # --- clarity/length (0.1)
    if isinstance(answer, str) and 50 <= len(answer) <= 1200:
        score += 0.1

    passed = score >= 0.7
    return passed, score, notes


def _norm(s: str) -> str:
    # whitespace-normalize for robust substring checks
    return re.sub(r"\s+", " ", (s or "").strip())


def _quote_in_source(quote: str, source_text: str) -> bool:
    q = _norm(quote)
    t = _norm(source_text)
    if not q or len(q) < 6:  # avoid empty / trivial quotes
        return False
    # case-insensitive substring match
    return q.lower() in t.lower()


SCORERS: Dict[str, ScorerFn] = {
    "default_v0": default_v0,
    "synth_v1": synth_v1,
}


def resolve_scorer(name: str) -> ScorerFn:
    """Resolve a scorer by registry name, with optional import escape hatch."""
    name = (name or "").strip()
    if not name:
        return default_v0
    if ":" in name:
        if not _allow_scoring_imports():
            raise ValueError(
                "Import-based scoring is disabled. Set TENSORFOUNDRY_ALLOW_SCORING_IMPORTS=1 to enable."
            )
        mod_name, func_name = name.split(":", 1)
        mod = import_module(mod_name)
        fn = getattr(mod, func_name, None)
        if fn is None or not callable(fn):
            raise ValueError(f"scoring function not found: {name}")
        return fn
    if name in SCORERS:
        return SCORERS[name]
    raise ValueError(f"Unknown scorer: {name!r}. Available: {sorted(SCORERS.keys())}")


def _allow_scoring_imports() -> bool:
    return os.getenv("TENSORFOUNDRY_ALLOW_SCORING_IMPORTS", "0").lower() in {"1", "true", "yes"}


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
