from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _iter_reward_files(logs_dir: Path) -> List[Path]:
    return sorted(logs_dir.glob("**/reward.jsonl"))


def build_routing_policy_v0(
    *,
    logs_dir: Path,
    default_model: str,
    min_cases: int = 3,
) -> Dict[str, Any]:
    # suite_id, model_id -> stats
    agg: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    for rf in _iter_reward_files(logs_dir):
        for r in _read_jsonl(rf):
            if r.get("version") != "reward.v0":
                continue
            suite_id = r.get("suite_id")
            model_id = r.get("model_id")
            score = r.get("overall_score")
            if not isinstance(suite_id, str) or not isinstance(model_id, str):
                continue
            if not isinstance(score, (int, float)):
                continue
            agg[(suite_id, model_id)].append(float(score))

    # pick best model per suite by mean score (with min_cases)
    by_suite: Dict[str, str] = {}
    suites = sorted({suite for (suite, _m) in agg.keys()})
    for suite in suites:
        candidates = []
        for (s, m), scores in agg.items():
            if s != suite:
                continue
            if len(scores) < min_cases:
                continue
            candidates.append((sum(scores) / len(scores), len(scores), m))
        if not candidates:
            continue
        # highest mean, then more samples
        candidates.sort(reverse=True)
        by_suite[suite] = candidates[0][2]

    return {
        "version": "routing.v0",
        "default_model": default_model,
        "by_suite": by_suite,
        "meta": {
            "logs_dir": str(logs_dir),
            "min_cases": min_cases,
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build routing policy from reward.jsonl logs.")
    parser.add_argument("--logs-dir", type=str, default="logs")
    parser.add_argument("--out", type=str, default="policies/routing_v0.json")
    parser.add_argument("--default-model", type=str, required=True)
    parser.add_argument("--min-cases", type=int, default=3)
    args = parser.parse_args(argv)

    policy = build_routing_policy_v0(
        logs_dir=Path(args.logs_dir),
        default_model=args.default_model,
        min_cases=args.min_cases,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(f"Wrote policy: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())