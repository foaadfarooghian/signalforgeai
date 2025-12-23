"""CLI entrypoint for running evaluation suites."""

from __future__ import annotations

import argparse
from html import parser
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

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a TensorFoundry evaluation suite.")
    parser.add_argument("suite", type=str, help="Path to suite JSON (e.g. src/.../suites/quickstart.json)")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for results outputs")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Directory for trace logs")
    parser.add_argument("--policy", type=str, default="", help="Path to routing policy JSON (optional)")
    parser.add_argument("--bandits", type=str, default="learning/bandits/routing_bandits_v0.json", help="Path to bandit state file",)
    parser.add_argument("--no-bandits", action="store_true", help="Disable bandit routing",)
    args = parser.parse_args(argv)

    suite_obj = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    suite_name = suite_obj["suite_name"]

    policy = load_policy_if_present(args.policy)
    if policy:
        chosen_model = policy.pick_model(suite_name)
        os.environ["TENSORFOUNDRY_MODEL_ID"] = chosen_model
        print(f"Model (policy): {chosen_model}")
    else:
        print(f"Model (env/default): {os.getenv('TENSORFOUNDRY_MODEL_ID') or 'unknown'}")

    if not args.no_bandits:
        bandits = RoutingBanditsV0.load(args.bandits)
        candidates = candidate_models_from_env()
        # default model if unset
        if bandits.default_model == "unknown":
            bandits.default_model = os.getenv("TENSORFOUNDRY_MODEL_ID") or (candidates[0] if candidates else "unknown")

        chosen = bandits.choose_model(suite_name, candidates)
        os.environ["TENSORFOUNDRY_MODEL_ID"] = chosen
        print(f"Model (bandit): {chosen}")
    else:
        bandits = None
        print(f"Model (env/default): {os.getenv('TENSORFOUNDRY_MODEL_ID') or 'unknown'}")
        
    result = run_suite(
        suite_path=Path(args.suite),
        output_dir=Path(args.output_dir),
        logs_dir=Path(args.logs_dir),
    )

    if bandits is not None:
        # result.run_logs_dir should be printed by your run_suite result now
        rows = load_rewards_for_run(result.run_logs_dir)
        # collect rewards for this suite+model (should match anyway)
        rewards = []
        for r in rows:
            if r.get("version") == "reward.v0" and r.get("suite_id") == suite_name and r.get("model_id") == os.environ.get("TENSORFOUNDRY_MODEL_ID"):
                rewards.append(float(r.get("overall_score", 0.0)))

        bandits.update_from_rewards(suite_name, os.environ["TENSORFOUNDRY_MODEL_ID"], rewards)
        bandits.save(args.bandits)
        print(f"Bandits updated: {args.bandits}")

   

    print(f"Suite {result.suite_name!r} finished: pass_rate={result.pass_rate:.2%} ({result.passed}/{result.num_cases})")
    print(f"Run logs: {result.run_logs_dir}")
    print(f"Results: {Path(args.output_dir) / (result.suite_name + '.results.json')}")
    print(f"Summary:  {Path(args.output_dir) / (result.suite_name + '.summary.md')}")
    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())