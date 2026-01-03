"""
Preference dataset exporter for TensorFoundry.

Builds DPO-style preference pairs from reward.jsonl + trace files.

v0 approach:
- group runs by (suite_id, case_id)
- require at least 2 models for same case
- prefer the model with higher effective reward:
    effective = score - lambda_cost * cost_usd - mu_latency * latency_seconds
- export (prompt, response_a, response_b, preferred)
"""

from __future__ import annotations

import argparse
import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
from collections import defaultdict
from tensorfoundry.export.extract import (
    extract_instruction,
    extract_response,
    extract_step_prompt_full,
    extract_step_text_full,
)

# ----------------------------
# Models
# ----------------------------

@dataclass(frozen=True)
class PreferenceExample:
    prompt: str
    response_a: str
    response_b: str
    preferred: str  # "a" or "b"
    meta: Dict[str, Any]

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "prompt": self.prompt,
                "response_a": self.response_a,
                "response_b": self.response_b,
                "preferred": self.preferred,
                "meta": self.meta,
            },
            ensure_ascii=False,
        )


@dataclass(frozen=True)
class _Candidate:
    prompt: str
    response: str
    prompt_step: str
    model_id: str
    trace_id: str
    trace_path: str
    score: float
    cost_usd: Optional[float]
    latency_ms: Optional[int]
    total_tokens: Any
    effective: float
    created_at: Any


# ----------------------------
# IO helpers
# ----------------------------

def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def _find_reward_files(logs_root: Path) -> List[Path]:
    return sorted(logs_root.rglob("reward.jsonl"))


def _trace_path_for_reward(reward_file: Path, trace_id: str) -> Path:
    run_dir = reward_file.parent
    return run_dir / f"{trace_id}.jsonl"


def _read_trace_events(trace_path: Path) -> List[Dict[str, Any]]:
    return list(_iter_jsonl(trace_path))


# ----------------------------
# Filtering
# ----------------------------

def _reward_passes_filters(
    r: Dict[str, Any],
    *,
    suite: Optional[str],
    min_score: float,
    success_only: bool,
    include_prefixes: List[str],
    exclude_prefixes: List[str],
    exclude_model_ids: Set[str],
) -> bool:
    if r.get("version") != "reward.v0":
        return False
    if suite and r.get("suite_id") != suite:
        return False

    score = r.get("overall_score")
    if not isinstance(score, (int, float)) or float(score) < min_score:
        return False

    if success_only and r.get("success") is not True:
        return False

    model_id = r.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        return False

    if model_id in exclude_model_ids:
        return False
    if any(model_id.startswith(px) for px in exclude_prefixes):
        return False
    if include_prefixes and not any(model_id.startswith(px) for px in include_prefixes):
        return False

    return True


# ----------------------------
# Preference logic
# ----------------------------

def _as_float(x: Any) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) else None


def _as_int(x: Any) -> Optional[int]:
    return int(x) if isinstance(x, int) else None


def _effective_reward(
    *,
    score: float,
    cost_usd: Optional[float],
    latency_ms: Optional[int],
    lambda_cost: float,
    mu_latency: float,
) -> float:
    eff = float(score)
    if cost_usd is not None:
        eff -= lambda_cost * float(cost_usd)
    if latency_ms is not None:
        eff -= mu_latency * (float(latency_ms) / 1000.0)
    # keep bounded for stable preferences
    return max(0.0, min(1.0, eff))

def _normalize_prompt(prompt: str, mode: str) -> str:
    if mode == "strip_json":
        for marker in ("DRAFT_JSON:", "RESOLVED_JSON:"):
            idx = prompt.find(marker)
            if idx != -1:
                return prompt[: idx + len(marker)].rstrip()
    elif mode == "mask_json":
        for marker in ("DRAFT_JSON:", "RESOLVED_JSON:"):
            idx = prompt.find(marker)
            if idx != -1:
                head = prompt[: idx + len(marker)].rstrip()
                return head + "\n<OMITTED_JSON>"
    return prompt


def export_preferences(
    *,
    logs_root: Path,
    out_path: Path,
    suite: Optional[str],
    min_score: float,
    success_only: bool,
    include_prefixes: List[str],
    exclude_prefixes: List[str],
    exclude_model_ids: Set[str],
    lambda_cost: float,
    mu_latency: float,
    max_abs_score_gap: float,
    limit: Optional[int],
    prompt_normalize: str,
) -> Tuple[int, int]:
    """
    Returns: (written_pairs, skipped_pairs)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Gather reward rows grouped by (suite_id, case_id)
    grouped: Dict[Tuple[str, str], List[Tuple[Path, Dict[str, Any]]]] = defaultdict(list)

    for rf in _find_reward_files(logs_root):
        for r in _iter_jsonl(rf):
            if not _reward_passes_filters(
                r,
                suite=suite,
                min_score=min_score,
                success_only=success_only,
                include_prefixes=include_prefixes,
                exclude_prefixes=exclude_prefixes,
                exclude_model_ids=exclude_model_ids,
            ):
                continue

            suite_id = r.get("suite_id")
            case_id = r.get("case_id")
            if not isinstance(suite_id, str) or not isinstance(case_id, str):
                continue

            grouped[(suite_id, case_id)].append((rf, r))

    written = 0
    skipped = 0

    with out_path.open("w", encoding="utf-8") as f:
        for (suite_id, case_id), items in grouped.items():
            # Need at least 2 candidate runs (different models)
            if len(items) < 2:
                continue

            # Build candidates with extracted prompt/response + effective reward
            candidates: List[_Candidate] = []
            for rf, r in items:
                trace_id = r.get("trace_id")
                if not isinstance(trace_id, str) or not trace_id:
                    continue

                trace_path = _trace_path_for_reward(rf, trace_id)
                if not trace_path.exists():
                    continue

                events = _read_trace_events(trace_path)

                step = None
                prompt = None
                for candidate_step in ("critic_check", "draft_answer"):
                    prompt = extract_step_prompt_full(events, step=candidate_step)
                    if prompt:
                        step = candidate_step
                        break
                if not prompt:
                    prompt = extract_instruction(events)

                response = None
                if step:
                    response = extract_step_text_full(events, step=step)
                if not response:
                    response = extract_response(events, max_chars=None)

                if prompt_normalize != "none" and prompt:
                    prompt = _normalize_prompt(prompt, prompt_normalize)

                if not isinstance(prompt, str) or not isinstance(response, str) or not prompt or not response:
                    continue

                score = _as_float(r.get("overall_score"))
                if score is None:
                    continue

                model_id = r.get("model_id")
                if not isinstance(model_id, str) or not model_id:
                    continue

                cost = _as_float(r.get("cost_usd"))
                lat = _as_int(r.get("latency_ms"))
                eff = _effective_reward(
                    score=score,
                    cost_usd=cost,
                    latency_ms=lat,
                    lambda_cost=lambda_cost,
                    mu_latency=mu_latency,
                )

                candidates.append(
                    _Candidate(
                        prompt=prompt,
                        response=response,
                        model_id=model_id,
                        trace_id=trace_id,
                        trace_path=str(trace_path),
                        score=score,
                        cost_usd=cost,
                        latency_ms=lat,
                        total_tokens=r.get("total_tokens"),
                        effective=eff,
                        created_at=r.get("created_at"),
                        prompt_step=step or "task_received",
                    )
                )

            # Need at least 2 extracted candidates with the same prompt
            if len(candidates) < 2:
                continue

            # Ensure prompt consistency (same case should yield same prompt; guard anyway)
            prompt0 = candidates[0].prompt
            candidates = [c for c in candidates if c.prompt == prompt0]
            if len(candidates) < 2:
                continue

            # Sort by effective reward
            candidates.sort(key=lambda c: c.effective, reverse=True)
            best_by_model: Dict[str, _Candidate] = {}
            for c in candidates:
                mid = c.model_id
                if mid not in best_by_model:
                    best_by_model[mid] = c
            candidates = list(best_by_model.values())

            if len(candidates) < 2:
                continue

            candidates.sort(key=lambda c: c.effective, reverse=True)
            best = candidates[0]

            # Pair best vs a runner-up that is "close enough" in raw score (so it's a fair preference)
            paired = False
            for other in candidates[1:]:
                if other.model_id == best.model_id:
                    continue
                if other.response.strip() == best.response.strip():
                    continue
                if abs(best.score - other.score) > max_abs_score_gap:
                    continue

                # Build a single preference pair: best (a) vs other (b)
                winner = best
                loser = other  # candidates are sorted by effective desc so loser has <= effective

                # Randomize which side gets winner
                if random.random() < 0.5:
                    a, b = winner, loser
                    preferred = "a"
                else:
                    a, b = loser, winner
                    preferred = "b"

                ex = PreferenceExample(
                    prompt=prompt0,
                    response_a=a.response,
                    response_b=b.response,
                    preferred=preferred,
                    meta={
                        "suite_id": suite_id,
                        "case_id": case_id,
                        "lambda_cost": lambda_cost,
                        "mu_latency": mu_latency,
                        "prompt_step": a.prompt_step,
                        "prompt_normalize": prompt_normalize,
                        "a": {
                            "model_id": a.model_id,
                            "trace_id": a.trace_id,
                            "score": a.score,
                            "cost_usd": a.cost_usd,
                            "latency_ms": a.latency_ms,
                            "total_tokens": a.total_tokens,
                            "effective": a.effective,
                            "trace_path": a.trace_path,
                            "prompt_step": a.prompt_step,
                        },
                        "b": {
                            "model_id": b.model_id,
                            "trace_id": b.trace_id,
                            "score": b.score,
                            "cost_usd": b.cost_usd,
                            "latency_ms": b.latency_ms,
                            "total_tokens": b.total_tokens,
                            "effective": b.effective,
                            "trace_path": b.trace_path,
                            "prompt_step": b.prompt_step,
                        },
                        "deltas": {
                            "score": a.score - b.score,
                            "cost_usd": (a.cost_usd if a.cost_usd is not None else 0.0)
                                        - (b.cost_usd if b.cost_usd is not None else 0.0),
                            "latency_ms": (a.latency_ms if a.latency_ms is not None else 0)
                                        - (b.latency_ms if b.latency_ms is not None else 0),
                            "effective": a.effective - b.effective,
                        },
                        # optional: explicit winner for sanity checks
                        "winner_model_id": winner.model_id,
                    },

                )

                f.write(ex.to_jsonl() + "\n")
                written += 1
                paired = True
                
                break  # v0: emit at most one pair per case

            if not paired:
                skipped += 1

            if limit is not None and written >= limit:
                return written, skipped

    return written, skipped


# ----------------------------
# CLI
# ----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Export preference datasets from TensorFoundry logs.")
    p.add_argument("--logs-root", type=str, default="logs", help="Path to logs/ root")
    p.add_argument("--out", type=str, required=True, help="Output JSONL path")
    p.add_argument("--suite", type=str, default="", help="Filter suite_id")
    p.add_argument("--min-score", type=float, default=0.7, help="Minimum overall_score")
    p.add_argument("--success-only", action="store_true", help="Only export success=true rows")
    p.add_argument("--limit", type=int, default=0, help="Max pairs (0 = no limit)")

    p.add_argument("--include-model-prefix", type=str, default="openai:,ollama:", help="Comma-separated prefixes to include")
    p.add_argument("--exclude-model-prefix", type=str, default="dummy:", help="Comma-separated prefixes to exclude")
    p.add_argument("--exclude-model-id", type=str, default="dummy_good,dummy_mid,dummy_bad,gpt-5-mini", help="Comma-separated model IDs to exclude")

    p.add_argument("--lambda-cost", type=float, default=0.0, help="Cost penalty weight")
    p.add_argument("--mu-latency", type=float, default=0.0, help="Latency penalty per second")
    p.add_argument("--max-abs-score-gap", type=float, default=0.15, help="Only pair if |score_a-score_b| <= gap")
    p.add_argument("--seed", type=int, default=42, help="Random seed for A/B assignment")
    p.add_argument("--prompt-normalize", type=str, default="none", choices=["none", "strip_json", "mask_json"], help="Normalize prompt_full by removing variable JSON blocks")

    args = p.parse_args(argv)

    random.seed(args.seed)
    logs_root = Path(args.logs_root)
    out_path = Path(args.out)

    suite = args.suite.strip() or None
    limit = None if args.limit <= 0 else int(args.limit)

    include_prefixes = [s.strip() for s in args.include_model_prefix.split(",") if s.strip()]
    exclude_prefixes = [s.strip() for s in args.exclude_model_prefix.split(",") if s.strip()]
    exclude_model_ids = {s.strip() for s in args.exclude_model_id.split(",") if s.strip()}

    written, skipped = export_preferences(
        logs_root=logs_root,
        out_path=out_path,
        suite=suite,
        min_score=float(args.min_score),
        success_only=bool(args.success_only),
        include_prefixes=include_prefixes,
        exclude_prefixes=exclude_prefixes,
        exclude_model_ids=exclude_model_ids,
        lambda_cost=float(args.lambda_cost),
        mu_latency=float(args.mu_latency),
        max_abs_score_gap=float(args.max_abs_score_gap),
        limit=limit,
        prompt_normalize=str(args.prompt_normalize),
    )

    print(f"Exported preference dataset: {out_path}")
    print(f"Pairs written: {written} (skipped cases: {skipped})")
    return 0 if written > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
