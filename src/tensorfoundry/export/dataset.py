"""
Dataset exporter for TensorFoundry.

Exports SFT-style JSONL rows from reward.jsonl + trace files.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from tensorfoundry.export.extract import (
    extract_instruction,
    extract_response,
)
# ----------------------------
# Models
# ----------------------------

@dataclass(frozen=True)
class SFTExample:
    instruction: str
    response: str
    meta: Dict[str, Any]

    def to_jsonl(self) -> str:
        return json.dumps(
            {"instruction": self.instruction, "response": self.response, "meta": self.meta},
            ensure_ascii=False,
        )


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
    # supports logs/<run_id>/reward.jsonl and nested layouts
    return sorted(logs_root.rglob("reward.jsonl"))


def _trace_path_for_reward(reward_file: Path, trace_id: str) -> Path:
    # reward.jsonl sits in logs/<run_id>/; trace is logs/<run_id>/<trace_id>.jsonl
    run_dir = reward_file.parent
    return run_dir / f"{trace_id}.jsonl"


def _read_trace_events(trace_path: Path) -> List[Dict[str, Any]]:
    return list(_iter_jsonl(trace_path))


# ----------------------------
# Export logic
# ----------------------------

def _reward_passes_filters(
    r: Dict[str, Any],
    *,
    suite: Optional[str],
    min_score: float,
    success_only: bool,
    include_prefixes: list[str],
    exclude_prefixes: list[str],
    exclude_model_ids: set[str],
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

    # Exclude exact IDs (legacy / known bad)
    if model_id in exclude_model_ids:
        return False

    # Exclude prefixes
    if any(model_id.startswith(px) for px in exclude_prefixes):
        return False

    # Include prefixes (if provided)
    if include_prefixes and not any(model_id.startswith(px) for px in include_prefixes):
        return False

    return True


def export_sft(
    *,
    logs_root: Path,
    out_path: Path,
    suite: Optional[str],
    min_score: float,
    success_only: bool,
    limit: Optional[int],
    include_prefixes: list[str],
    exclude_prefixes: list[str],
    exclude_model_ids: set[str],
) -> Tuple[int, int]:
    """
    Returns: (written, skipped)
    """
    written = 0
    skipped = 0

    reward_files = _find_reward_files(logs_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    

    with out_path.open("w", encoding="utf-8") as f:
        for rf in reward_files:
            for r in _iter_jsonl(rf):
                if not _reward_passes_filters(r, suite=suite, min_score=min_score, success_only=success_only, include_prefixes=include_prefixes, exclude_prefixes=exclude_prefixes, exclude_model_ids=exclude_model_ids):
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

                instruction = extract_instruction(events)
                response = extract_response(events)

                min_response_chars = int(os.getenv("TENSORFOUNDRY_MIN_RESPONSE_CHARS", "200"))
                if len(response.strip()) < min_response_chars:
                    skipped += 1
                    continue

                if not instruction or not response:
                    skipped += 1
                    continue

                meta = {
                    "trace_id": r.get("trace_id"),
                    "run_id": r.get("run_id"),
                    "suite_id": r.get("suite_id"),
                    "case_id": r.get("case_id"),
                    "agent_id": r.get("agent_id"),
                    "model_id": r.get("model_id"),
                    "score": r.get("overall_score"),
                    "success": r.get("success"),
                    "cost_usd": r.get("cost_usd"),
                    "latency_ms": r.get("latency_ms"),
                    "total_tokens": r.get("total_tokens"),
                    "created_at": r.get("created_at"),
                    "trace_path": str(trace_path),
                }

                ex = SFTExample(instruction=instruction, response=response, meta=meta)
                f.write(ex.to_jsonl() + "\n")
                written += 1

                if limit is not None and written >= limit:
                    return written, skipped

    return written, skipped


# ----------------------------
# CLI
# ----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Export SFT datasets from TensorFoundry logs.")
    p.add_argument("--logs-root", type=str, default="logs", help="Path to logs/ root")
    p.add_argument("--out", type=str, required=True, help="Output JSONL path")
    p.add_argument("--suite", type=str, default="", help="Filter suite_id")
    p.add_argument("--min-score", type=float, default=0.7, help="Minimum overall_score")
    p.add_argument("--success-only", action="store_true", help="Only export success=true rows")
    p.add_argument("--limit", type=int, default=0, help="Max rows (0 = no limit)")
    p.add_argument("--include-model-prefix", type=str, default="openai:,ollama:", help="Comma-separated list of model ID prefixes to include")
    p.add_argument("--exclude-model-prefix", type=str, default="dummy", help="Comma-separated list of model ID prefixes to exclude")
    p.add_argument("--exclude-model-id", type=str, default="dummy_good,dummy_mid,dummy_bad,gpt-5-mini", help="Comma-separated list of model IDs to exclude (to catch legacy)")
    args = p.parse_args(argv)

    include_prefixes = [s.strip() for s in args.include_model_prefix.split(",") if s.strip()]
    exclude_prefixes = [s.strip() for s in args.exclude_model_prefix.split(",") if s.strip()]
    exclude_model_ids = {s.strip() for s in args.exclude_model_id.split(",") if s.strip()}

    logs_root = Path(args.logs_root)
    out_path = Path(args.out)

    suite = args.suite.strip() or None
    limit = None if args.limit <= 0 else args.limit

    written, skipped = export_sft(
        logs_root=logs_root,
        out_path=out_path,
        suite=suite,
        min_score=float(args.min_score),
        success_only=bool(args.success_only),
        limit=limit,
        include_prefixes=include_prefixes,
        exclude_prefixes=exclude_prefixes,
        exclude_model_ids=exclude_model_ids,
    )

    print(f"Exported SFT dataset: {out_path}")
    print(f"Rows written: {written} (skipped: {skipped})")
    return 0 if written > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())