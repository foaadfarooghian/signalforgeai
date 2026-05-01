"""
Curriculum builder for TensorFoundry.

Assigns difficulty buckets (easy/repair/escalation) from reward.jsonl + traces.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from tensorfoundry.export.extract import (
    extract_instruction,
    extract_response,
    extract_step_prompt_full,
    extract_step_text_full,
)
from tensorfoundry.export.quality import attach_split_meta

CURRICULUM_SCHEMA_VERSION = "curriculum.v0"


@dataclass(frozen=True)
class CurriculumRow:
    trace_id: str
    run_id: str
    suite_id: str
    case_id: str
    model_id: str
    bucket: str
    difficulty: int
    prompt: str
    response: str
    signals: Dict[str, Any]
    meta: Dict[str, Any]
    version: str = CURRICULUM_SCHEMA_VERSION

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "trace_id": self.trace_id,
                "run_id": self.run_id,
                "suite_id": self.suite_id,
                "case_id": self.case_id,
                "model_id": self.model_id,
                "bucket": self.bucket,
                "difficulty": self.difficulty,
                "prompt": self.prompt,
                "response": self.response,
                "signals": self.signals,
                "meta": self.meta,
            },
            ensure_ascii=False,
        )


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


def _count_retries(events: List[Dict[str, Any]]) -> int:
    return sum(1 for ev in events if ev.get("event_type") == "retry_requested")


def _assign_bucket(
    *,
    success: bool,
    score: float,
    retry_count: int,
    violations: List[str],
    easy_min_score: float,
    escalation_max_score: float,
) -> Tuple[str, int]:
    if (not success) or score <= escalation_max_score:
        return "escalation", 3
    if retry_count > 0 or violations:
        return "repair", 2
    if score >= easy_min_score:
        return "easy", 1
    return "repair", 2


def export_curriculum(
    *,
    logs_root: Path,
    out_path: Path,
    suite: Optional[str],
    min_score: float,
    success_only: bool,
    include_prefixes: List[str],
    exclude_prefixes: List[str],
    exclude_model_ids: Set[str],
    step: str,
    use_step_prompt: bool,
    min_response_chars: int,
    easy_min_score: float,
    escalation_max_score: float,
    buckets: Optional[Set[str]] = None,
    limit: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Returns: (written, skipped)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with out_path.open("w", encoding="utf-8") as f:
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

                trace_id = r.get("trace_id")
                if not isinstance(trace_id, str) or not trace_id:
                    skipped += 1
                    continue

                trace_path = _trace_path_for_reward(rf, trace_id)
                if not trace_path.exists():
                    skipped += 1
                    continue

                events = _read_trace_events(trace_path)
                instruction = None
                if use_step_prompt:
                    instruction = extract_step_prompt_full(events, step=step)
                instruction = instruction or extract_instruction(events)
                response = extract_step_text_full(events, step=step) or extract_response(events)

                if not instruction or not response:
                    skipped += 1
                    continue
                if len(response.strip()) < min_response_chars:
                    skipped += 1
                    continue

                score = float(r.get("overall_score", 0.0))
                success = r.get("success") is True
                raw_violations = r.get("violations")
                violations: List[str] = []
                if isinstance(raw_violations, list):
                    for v in raw_violations:
                        if isinstance(v, str):
                            violations.append(v)
                        else:
                            violations.append(str(v))
                retry_count = _count_retries(events)

                bucket, difficulty = _assign_bucket(
                    success=success,
                    score=score,
                    retry_count=retry_count,
                    violations=violations,
                    easy_min_score=easy_min_score,
                    escalation_max_score=escalation_max_score,
                )

                if buckets is not None and bucket not in buckets:
                    skipped += 1
                    continue

                meta = {
                    "prompt_step": step,
                    "trace_path": str(trace_path),
                    "provenance": {
                        "inputs": {
                            "logs_root": str(logs_root),
                            "reward_jsonl": str(rf),
                            "trace_path": str(trace_path),
                        }
                    },
                }
                attach_split_meta(meta, suite_id=r.get("suite_id"), case_id=r.get("case_id"))

                row = CurriculumRow(
                    trace_id=trace_id,
                    run_id=str(r.get("run_id", "")),
                    suite_id=str(r.get("suite_id", "")),
                    case_id=str(r.get("case_id", "")),
                    model_id=str(r.get("model_id", "")),
                    bucket=bucket,
                    difficulty=difficulty,
                    prompt=instruction,
                    response=response,
                    signals={
                        "success": success,
                        "overall_score": score,
                        "violations": violations,
                        "retry_count": retry_count,
                        "latency_ms": r.get("latency_ms"),
                        "cost_usd": r.get("cost_usd"),
                    },
                    meta=meta,
                )

                f.write(row.to_jsonl() + "\n")
                written += 1

                if limit is not None and written >= limit:
                    return written, skipped

    return written, skipped


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build a curriculum dataset from reward.jsonl logs.")
    p.add_argument("--logs-root", type=str, default="logs", help="Path to logs/ root")
    p.add_argument("--out", type=str, required=True, help="Output JSONL path")
    p.add_argument("--suite", type=str, default="", help="Filter suite_id")
    p.add_argument("--min-score", type=float, default=0.0, help="Minimum overall_score to include")
    p.add_argument("--success-only", action="store_true", help="Only export success=true rows")
    p.add_argument("--limit", type=int, default=0, help="Max rows (0 = no limit)")

    p.add_argument("--include-model-prefix", type=str, default="openai:,ollama:", help="Comma-separated prefixes to include")
    p.add_argument("--exclude-model-prefix", type=str, default="dummy", help="Comma-separated prefixes to exclude")
    p.add_argument("--exclude-model-id", type=str, default="dummy_good,dummy_mid,dummy_bad,gpt-5-mini", help="Comma-separated model IDs to exclude")

    p.add_argument("--step", type=str, default="critic_check", help="Step name to extract from traces")
    p.add_argument("--use-step-prompt", type=int, default=0, help="Use step-specific prompt_full instead of Task instruction")
    p.add_argument("--min-response-chars", type=int, default=200, help="Skip responses shorter than this length")

    p.add_argument("--easy-min-score", type=float, default=0.8, help="Score threshold for easy cases")
    p.add_argument("--escalation-max-score", type=float, default=0.5, help="Score threshold for escalation cases")
    p.add_argument("--bucket", type=str, default="", help="Comma-separated buckets to include (easy,repair,escalation)")

    args = p.parse_args(argv)

    include_prefixes = [s.strip() for s in args.include_model_prefix.split(",") if s.strip()]
    exclude_prefixes = [s.strip() for s in args.exclude_model_prefix.split(",") if s.strip()]
    exclude_model_ids = {s.strip() for s in args.exclude_model_id.split(",") if s.strip()}

    suite = args.suite.strip() or None
    limit = None if args.limit <= 0 else int(args.limit)
    buckets = {b.strip() for b in args.bucket.split(",") if b.strip()} or None

    written, skipped = export_curriculum(
        logs_root=Path(args.logs_root),
        out_path=Path(args.out),
        suite=suite,
        min_score=float(args.min_score),
        success_only=bool(args.success_only),
        include_prefixes=include_prefixes,
        exclude_prefixes=exclude_prefixes,
        exclude_model_ids=exclude_model_ids,
        step=str(args.step),
        use_step_prompt=bool(args.use_step_prompt),
        min_response_chars=int(args.min_response_chars),
        easy_min_score=float(args.easy_min_score),
        escalation_max_score=float(args.escalation_max_score),
        buckets=buckets,
        limit=limit,
    )

    print(f"Exported curriculum dataset: {Path(args.out)}")
    print(f"Rows written: {written} (skipped: {skipped})")
    return 0 if written > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
