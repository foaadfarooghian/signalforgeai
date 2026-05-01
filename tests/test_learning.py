from __future__ import annotations

import json
from pathlib import Path

from tensorfoundry.learning.bandits import RoutingBanditsV0, candidate_models_from_env
from tensorfoundry.learning.build_policy import build_routing_policy_v0
from tensorfoundry.learning.learn import main as learn_main
from tensorfoundry.learning.model_stats import RoutingStatsV0
from tensorfoundry.learning.routing_policy import RoutingPolicyV0


def test_bandits_update_from_rewards_creates_arm() -> None:
    bandits = RoutingBanditsV0()
    bandits.update_from_rewards("suite_a", "model_a", [1.0, 0.0])
    arm = bandits.by_suite["suite_a"]["model_a"]
    assert arm.alpha == 2.0
    assert arm.beta == 2.0


def test_candidate_models_from_env_fallback(monkeypatch) -> None:
    monkeypatch.delenv("TENSORFOUNDRY_CANDIDATE_MODELS", raising=False)
    monkeypatch.setenv("TENSORFOUNDRY_MODEL_ID", "dummy_good")
    assert candidate_models_from_env() == ["dummy_good"]


def test_routing_stats_update_and_load(tmp_path: Path) -> None:
    stats = RoutingStatsV0()
    stats.update_from_reward_row(
        "suite_a",
        "model_a",
        {"cost_usd": 0.5, "latency_ms": 1500, "total_tokens": 100},
    )
    path = tmp_path / "stats.json"
    stats.save(path)

    loaded = RoutingStatsV0.load(path)
    ms = loaded.by_suite["suite_a"]["model_a"]
    assert ms.cost_usd.value == 0.5
    assert ms.latency_s.value == 1.5
    assert ms.total_tokens.value == 100


def test_build_routing_policy_v0_min_cases(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs"
    run_dir = logs_root / "run1"
    rows = [
        {"version": "reward.v0", "suite_id": "s1", "model_id": "m1", "overall_score": 0.9},
        {"version": "reward.v0", "suite_id": "s1", "model_id": "m1", "overall_score": 0.8},
        {"version": "reward.v0", "suite_id": "s1", "model_id": "m2", "overall_score": 0.6},
        {"version": "reward.v0", "suite_id": "s1", "model_id": "m2", "overall_score": 0.6},
        {"version": "reward.v0", "suite_id": "s1", "model_id": "m2", "overall_score": 0.6},
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reward.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    policy = build_routing_policy_v0(logs_dir=logs_root, default_model="m0", min_cases=3)
    assert policy["by_suite"]["s1"] == "m2"


def test_routing_policy_load_and_pick(tmp_path: Path) -> None:
    data = {"version": "routing.v0", "default_model": "m0", "by_suite": {"s1": "m1"}}
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    policy = RoutingPolicyV0.load(path)
    assert policy.pick_model("s1") == "m1"
    assert policy.pick_model("unknown") == "m0"


def test_train_dry_run_validates_sft_and_dpo(tmp_path: Path) -> None:
    sft = tmp_path / "sft.jsonl"
    dpo = tmp_path / "dpo.jsonl"
    sft.write_text(
        json.dumps(
            {
                "version": "sft.v0",
                "instruction": "Task: train",
                "prompt": "Task: train",
                "response": "A training response",
                "meta": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dpo.write_text(
        json.dumps(
            {
                "version": "dpo.v0",
                "prompt": "Task: train",
                "chosen": "Better response",
                "rejected": "Worse response",
                "meta": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    code = learn_main(
        [
            "train",
            "--base-model",
            "dummy/base",
            "--sft",
            "--dpo",
            "--sft-data",
            str(sft),
            "--dpo-data",
            str(dpo),
            "--dry-run",
        ]
    )
    assert code == 0
