"""
Repair dataset exporter for TensorFoundry.

Builds failure -> success pairs from reward.jsonl + trace files.
"""

from __future__ import annotations

import argparse
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

REPAIRS_SCHEMA_VERSION = "repairs.v0"
DPO_SCHEMA_VERSION = "dpo.v0"


@dataclass(frozen=True)
class RepairExample:
    prompt: str
    failed_response: str
    repaired_response: str
    meta: Dict[str, Any]
    version: str = REPAIRS_SCHEMA_VERSION

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "prompt": self.prompt,
                "failed": self.failed_response,
                "repaired": self.repaired_response,
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
    include_prefixes: List[str],
    exclude_prefixes: List[str],
    exclude_model_ids: Set[str],
) -> bool:
    if r.get("version") != "reward.v0":
        return False
    if suite and r.get("suite_id") != suite:
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


def _as_float(x: Any) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) else None


def export_repairs(
    *,
    logs_root: Path,
    out_path: Path,
    suite: Optional[str],
    min_success_score: float,
    max_failure_score: float,
    include_prefixes: List[str],
    exclude_prefixes: List[str],
    exclude_model_ids: Set[str],
    limit: Optional[int],
    prompt_normalize: str,
    prompt_source: str = "auto",
    output_format: str = "repairs",
    trace_allowlist: Optional[Set[str]] = None,
) -> Tuple[int, int]:
    """
    Returns: (written_pairs, skipped_pairs)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    grouped: Dict[Tuple[str, str], List[Tuple[Path, Dict[str, Any]]]] = defaultdict(list)
    for rf in _find_reward_files(logs_root):
        for r in _iter_jsonl(rf):
            if not _reward_passes_filters(
                r,
                suite=suite,
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
            successes: List[Dict[str, Any]] = []
            failures: List[Dict[str, Any]] = []

            for rf, r in items:
                score = _as_float(r.get("overall_score"))
                if score is None:
                    continue
                success = r.get("success") is True
                if success and score >= min_success_score:
                    successes.append({"reward": r, "reward_file": rf, "score": score})
                if (not success) or score <= max_failure_score:
                    failures.append({"reward": r, "reward_file": rf, "score": score})

            if not successes or not failures:
                continue

            # Pick best success and worst failure
            best = max(successes, key=lambda x: x["score"])
            worst = min(failures, key=lambda x: x["score"])

            # Extract prompt/response for each trace
            def _extract_pair(rec: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
                r = rec["reward"]
                trace_id = r.get("trace_id")
                if not isinstance(trace_id, str) or not trace_id:
                    return None
                if trace_allowlist is not None and trace_id not in trace_allowlist:
                    return None
                trace_path = _trace_path_for_reward(rec["reward_file"], trace_id)
                if not trace_path.exists():
                    return None
                events = _read_trace_events(trace_path)

                step = None
                prompt = None
                if prompt_source == "instruction":
                    prompt = extract_instruction(events)
                    step = "task_received"
                elif prompt_source == "step_prompt":
                    for candidate_step in ("critic_check", "draft_answer"):
                        prompt = extract_step_prompt_full(events, step=candidate_step)
                        if prompt:
                            step = candidate_step
                            break
                else:
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
                    return None

                return prompt, response, step or "task_received"

            pair_best = _extract_pair(best)
            pair_worst = _extract_pair(worst)
            if not pair_best or not pair_worst:
                skipped += 1
                continue

            prompt_best, response_best, step_best = pair_best
            prompt_worst, response_worst, step_worst = pair_worst
            if prompt_best != prompt_worst:
                skipped += 1
                continue

            best_r = best["reward"]
            worst_r = worst["reward"]
            best_model = best_r.get("model_id", "unknown")
            worst_model = worst_r.get("model_id", "unknown")

            pair_type = "cross_model" if best_model != worst_model else "self_repair"

            meta = {
                "suite_id": suite_id,
                "case_id": case_id,
                "prompt_step": step_best,
                "pair_type": pair_type,
                "success": {
                    "model_id": best_model,
                    "trace_id": best_r.get("trace_id"),
                    "score": best.get("score"),
                    "cost_usd": best_r.get("cost_usd"),
                    "latency_ms": best_r.get("latency_ms"),
                    "trace_path": str(_trace_path_for_reward(best["reward_file"], best_r.get("trace_id"))),
                },
                "failure": {
                    "model_id": worst_model,
                    "trace_id": worst_r.get("trace_id"),
                    "score": worst.get("score"),
                    "cost_usd": worst_r.get("cost_usd"),
                    "latency_ms": worst_r.get("latency_ms"),
                    "trace_path": str(_trace_path_for_reward(worst["reward_file"], worst_r.get("trace_id"))),
                },
            }

            ex = RepairExample(
                prompt=prompt_best,
                failed_response=response_worst,
                repaired_response=response_best,
                meta=meta,
            )

            if output_format == "dpo":
                row = {
                    "version": DPO_SCHEMA_VERSION,
                    "prompt": ex.prompt,
                    "chosen": ex.repaired_response,
                    "rejected": ex.failed_response,
                    "meta": ex.meta,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                f.write(ex.to_jsonl() + "\n")

            written += 1
            if limit is not None and written >= limit:
                return written, skipped

    return written, skipped


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


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Export repair pairs (failure -> success) from TensorFoundry logs.")
    p.add_argument("--logs-root", type=str, default="logs", help="Path to logs/ root")
    p.add_argument("--out", type=str, required=True, help="Output JSONL path")
    p.add_argument("--suite", type=str, default="", help="Filter suite_id")
    p.add_argument("--min-success-score", type=float, default=0.7, help="Minimum score for success candidates")
    p.add_argument("--max-failure-score", type=float, default=0.4, help="Maximum score for failure candidates")
    p.add_argument("--limit", type=int, default=0, help="Max pairs (0 = no limit)")

    p.add_argument("--include-model-prefix", type=str, default="openai:,ollama:", help="Comma-separated prefixes to include")
    p.add_argument("--exclude-model-prefix", type=str, default="dummy:", help="Comma-separated prefixes to exclude")
    p.add_argument("--exclude-model-id", type=str, default="dummy_good,dummy_mid,dummy_bad,gpt-5-mini", help="Comma-separated model IDs to exclude")
    p.add_argument("--prompt-normalize", type=str, default="none", choices=["none", "strip_json", "mask_json"], help="Normalize prompt_full by removing variable JSON blocks")
    p.add_argument("--prompt-source", type=str, default="auto", choices=["auto", "instruction", "step_prompt"], help="Prompt source strategy")
    p.add_argument("--format", type=str, default="repairs", choices=["repairs", "dpo"], help="Output format")

    args = p.parse_args(argv)
    logs_root = Path(args.logs_root)
    out_path = Path(args.out)
    suite = args.suite.strip() or None
    limit = None if args.limit <= 0 else int(args.limit)

    include_prefixes = [s.strip() for s in args.include_model_prefix.split(",") if s.strip()]
    exclude_prefixes = [s.strip() for s in args.exclude_model_prefix.split(",") if s.strip()]
    exclude_model_ids = {s.strip() for s in args.exclude_model_id.split(",") if s.strip()}

    written, skipped = export_repairs(
        logs_root=logs_root,
        out_path=out_path,
        suite=suite,
        min_success_score=float(args.min_success_score),
        max_failure_score=float(args.max_failure_score),
        include_prefixes=include_prefixes,
        exclude_prefixes=exclude_prefixes,
        exclude_model_ids=exclude_model_ids,
        limit=limit,
        prompt_normalize=str(args.prompt_normalize),
        prompt_source=str(args.prompt_source),
        output_format=str(args.format),
    )

    print(f"Exported repair dataset: {out_path}")
    print(f"Pairs written: {written} (skipped cases: {skipped})")
    return 0 if written > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
