"""CLI entrypoint for running evaluation suites."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional
import os
import json
from tensorfoundry.evaluation.harness import run_suite
from tensorfoundry.learning.routing_policy import load_policy_if_present
from tensorfoundry.learning.bandits import (
    RoutingBanditsV0,
    candidate_models_from_env,
    load_rewards_for_run,
)
from tensorfoundry.learning.model_stats import RoutingStatsV0

def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for running an evaluation suite."""
    parser = argparse.ArgumentParser(description="Run a TensorFoundry evaluation suite.")
    parser.add_argument("suite", type=str, help="Path to suite JSON (e.g. src/.../suites/quickstart.json)")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for results outputs")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Directory for trace logs")
    parser.add_argument("--policy", type=str, default="", help="Path to routing policy JSON (optional)")
    parser.add_argument("--bandits", type=str, default="training/bandits/routing_bandits_v0.json", help="Path to bandit state file",)
    parser.add_argument("--no-bandits", action="store_true", help="Disable bandit routing",)
    parser.add_argument("--stats", type=str, default="training/bandits/routing_stats_v0.json", help="Path to routing stats file (cost/latency EWMA)")
    args = parser.parse_args(argv)

    suite_obj = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    suite_name = suite_obj["suite_name"]

    disable_bandits = args.no_bandits or (suite_name == "benchmark_v0_refactor")

    chosen_model: str = os.getenv("TENSORFOUNDRY_MODEL_ID") or "unknown"
    bandits = None
    candidates: list[str] = []

    policy = load_policy_if_present(args.policy)
    if policy:
        chosen_model = policy.pick_model(suite_name)
        os.environ["TENSORFOUNDRY_MODEL_ID"] = chosen_model
        print(f"Model (policy): {chosen_model}")

    elif disable_bandits:
        print(f"Model (env/default): {chosen_model}")

    else:
        bandits = RoutingBanditsV0.load(args.bandits)
        candidates = candidate_models_from_env()

        if not candidates:
            bandits = None
            print("Bandits disabled: no candidate models found in env")
        else:
            if bandits.default_model == "unknown":
                bandits.default_model = chosen_model if chosen_model != "unknown" else candidates[0]

            # Load stats BEFORE choosing so routing can use constraints/utility
            stats = RoutingStatsV0.load(args.stats)

            chosen_model = bandits.choose_model(
                suite_name,
                candidates,
                stats=stats,
                max_cost_usd=float(os.getenv("TENSORFOUNDRY_MAX_COST_USD", "0")) or None,
                max_latency_s=float(os.getenv("TENSORFOUNDRY_MAX_LATENCY_S", "0")) or None,
                utility_lambda_cost=float(os.getenv("TENSORFOUNDRY_UTILITY_LAMBDA_COST", "0.0")),
                utility_mu_latency=float(os.getenv("TENSORFOUNDRY_UTILITY_MU_LATENCY", "0.0")),
            )
            os.environ["TENSORFOUNDRY_MODEL_ID"] = chosen_model
            print(f"Model (bandit): {chosen_model}")

    # Run suite
    result = run_suite(
        suite_path=Path(args.suite),
        output_dir=Path(args.output_dir),
        logs_dir=Path(args.logs_dir),
    )

    # Update learning artifacts
    if bandits is not None:
        rows = load_rewards_for_run(result.run_logs_dir)

        # 1) Update stats for the model we actually used
        stats = RoutingStatsV0.load(args.stats)
        for r in rows:
            if r.get("version") != "reward.v0":
                continue
            if r.get("suite_id") != suite_name:
                continue
            if r.get("model_id") != chosen_model:
                continue
            stats.update_from_reward_row(suite_name, chosen_model, r)
        stats.save(args.stats)

        # 2) Update bandit using quality-only reward
        rewards: list[float] = []
        raw_scores: list[float] = []
        for r in rows:
            if r.get("version") != "reward.v0":
                continue
            if r.get("suite_id") != suite_name:
                continue
            if r.get("model_id") != chosen_model:
                continue
            score = r.get("overall_score", 0.0)
            if isinstance(score, (int, float)):
                score_f = float(score)
                raw_scores.append(score_f)
                rewards.append(max(0.0, min(1.0, score_f)))

        bandits.update_from_rewards(suite_name, chosen_model, rewards)
        bandits.save(args.bandits)

        mean_score = (sum(raw_scores) / len(raw_scores)) if raw_scores else 0.0
        mean_q = (sum(rewards) / len(rewards)) if rewards else 0.0
        print(f"Bandits updated: {args.bandits} (n={len(rewards)} mean_score={mean_score:.3f} mean_quality={mean_q:.3f})")

   

    print(f"Suite {result.suite_name!r} finished: pass_rate={result.pass_rate:.2%} ({result.passed}/{result.num_cases})")
    print(f"Run logs: {result.run_logs_dir}")
    print(f"Results: {Path(args.output_dir) / (result.suite_name + '.results.json')}")
    print(f"Summary:  {Path(args.output_dir) / (result.suite_name + '.summary.md')}")
    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
