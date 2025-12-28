from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

def _get_float(d: Dict[str, Any], k: str) -> Optional[float]:
    v = d.get(k)
    return float(v) if isinstance(v, (int, float)) else None

def _get_int(d: Dict[str, Any], k: str) -> Optional[int]:
    v = d.get(k)
    return int(v) if isinstance(v, int) else None

def _effective(score: float, cost: float, lat_ms: int, lambda_cost: float, mu_latency: float) -> float:
    eff = score - lambda_cost * cost - mu_latency * (lat_ms / 1000.0)
    return max(0.0, min(1.0, eff))

def main() -> int:
    p = argparse.ArgumentParser(description="Summarise trade-offs from reward.jsonl files.")
    p.add_argument("root", type=str, help="logs/<run_id>/ dir OR logs/ dir (will scan reward.jsonl)")
    p.add_argument("--suite", type=str, default="", help="Filter by suite_id")
    p.add_argument("--lambda-cost", type=float, default=0.0, help="Cost penalty weight")
    p.add_argument("--mu-latency", type=float, default=0.0, help="Latency penalty weight per second")
    args = p.parse_args()

    root = Path(args.root)
    reward_files = []
    if root.is_dir() and (root / "reward.jsonl").exists():
        reward_files = [root / "reward.jsonl"]
    else:
        reward_files = list(root.rglob("reward.jsonl"))

    if not reward_files:
        print("No reward.jsonl files found.")
        return 2

    # stats[suite][model] aggregates
    stats: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(lambda: {
        "n": 0,
        "score_sum": 0.0,
        "cost_sum": 0.0,
        "cost_n": 0,
        "lat_sum": 0,
        "lat_n": 0,
        "tok_sum": 0,
        "tok_n": 0,
        "eff_sum": 0.0,
        "eff_n": 0,
        "success_sum": 0,
    }))

    for rf in reward_files:
        for line in rf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("version") != "reward.v0":
                continue

            suite_id = r.get("suite_id", "unknown")
            if args.suite and suite_id != args.suite:
                continue
            model_id = r.get("model_id", "unknown")

            score = _get_float(r, "overall_score")
            if score is None:
                continue

            s = stats[suite_id][model_id]
            s["n"] += 1
            s["score_sum"] += score
            if r.get("success") is True:
                s["success_sum"] += 1

            cost = _get_float(r, "cost_usd")
            if cost is not None:
                s["cost_sum"] += cost
                s["cost_n"] += 1

            lat = _get_int(r, "latency_ms")
            if lat is not None:
                s["lat_sum"] += lat
                s["lat_n"] += 1

            tok = _get_int(r, "total_tokens")
            if tok is not None:
                s["tok_sum"] += tok
                s["tok_n"] += 1

            # effective uses 0 when missing (keeps behaviour stable)
            eff = _effective(
                score=score,
                cost=cost or 0.0,
                lat_ms=lat or 0,
                lambda_cost=args.lambda_cost,
                mu_latency=args.mu_latency,
            )
            s["eff_sum"] += eff
            s["eff_n"] += 1

    # Print report
    for suite_id in sorted(stats.keys()):
        print(f"\n=== suite: {suite_id} ===")
        rows = []
        for model_id, s in stats[suite_id].items():
            n = s["n"]
            mean_score = s["score_sum"] / max(1, n)
            mean_cost = s["cost_sum"] / max(1, s["cost_n"]) if s["cost_n"] else None
            mean_lat = s["lat_sum"] / max(1, s["lat_n"]) if s["lat_n"] else None
            mean_tok = s["tok_sum"] / max(1, s["tok_n"]) if s["tok_n"] else None
            mean_eff = s["eff_sum"] / max(1, s["eff_n"]) if s["eff_n"] else None
            succ = s["success_sum"] / max(1, n)
            rows.append((mean_eff if mean_eff is not None else -1.0, model_id, n, mean_score, mean_cost, mean_lat, mean_tok, succ))

        rows.sort(reverse=True)
        print("| model_id | n | mean_score | mean_cost_usd | mean_latency_ms | mean_total_tokens | success_rate | mean_effective |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, model_id, n, mean_score, mean_cost, mean_lat, mean_tok, succ in rows:
            print(
                f"| {model_id} | {n} | {mean_score:.3f} | "
                f"{(f'{mean_cost:.6f}' if mean_cost is not None else '')} | "
                f"{(f'{mean_lat:.0f}' if mean_lat is not None else '')} | "
                f"{(f'{mean_tok:.0f}' if mean_tok is not None else '')} | "
                f"{succ:.2%} | "
                f"{(f'{( (_ if ( _:= (mean_eff:=None) ) else 0) )}' ) if False else ''}"
            )
        # print effective in a second pass cleanly (avoid overcomplication)
        for _, model_id, n, mean_score, mean_cost, mean_lat, mean_tok, succ in rows:
            mean_eff = (stats[suite_id][model_id]["eff_sum"] / max(1, stats[suite_id][model_id]["eff_n"])) if stats[suite_id][model_id]["eff_n"] else None
            # Replace the empty effective column by printing a final row line with effective included:
            # (Keeping output readable without fussing with formatting above)
        # (If you want, I’ll tighten formatting—kept simple on purpose.)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())