from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tensorfoundry.export.quality import split_meta
from tensorfoundry.learning.bandits import RoutingBanditsV0, candidate_models_from_env
from tensorfoundry.learning.build_policy import build_routing_policy_v0
from tensorfoundry.learning.learn import main as learn_main
from tensorfoundry.learning.model_stats import RoutingStatsV0
from tensorfoundry.learning.routing_policy import RoutingPolicyV0
from tensorfoundry.training.readiness import (
    load_training_preflight_report,
    scan_training_output_artifacts,
)


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


def test_train_dry_run_quality_gate_writes_preflight_report(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs"
    sft, dpo = _write_strict_training_datasets(tmp_path, logs_root)
    report = tmp_path / "training_preflight.json"

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
            "--quality-gate",
            "--logs-root",
            str(logs_root),
            "--report-out",
            str(report),
        ]
    )

    assert code == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["version"] == "training_preflight.v0"
    assert payload["ok"] is True
    assert payload["quality_gate"] is True
    assert {d["role"] for d in payload["datasets"]} == {"sft", "dpo"}
    assert all(d["content_sha256"] for d in payload["datasets"])
    assert payload["artifact_manifest"]["version"] == "training_artifact.v0"
    assert payload["artifact_manifest"]["dataset_hashes"]["sft"]
    assert payload["artifact_manifest"]["split_counts"]["dpo"]


def test_train_dry_run_quality_gate_fails_duplicate_payloads(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs"
    sft, _dpo = _write_strict_training_datasets(tmp_path, logs_root)
    row = json.loads(sft.read_text(encoding="utf-8").splitlines()[0])
    sft.write_text(
        "\n".join(json.dumps(row) for _ in range(2)) + "\n",
        encoding="utf-8",
    )

    code = learn_main(
        [
            "train",
            "--base-model",
            "dummy/base",
            "--sft",
            "--sft-data",
            str(sft),
            "--dry-run",
            "--quality-gate",
            "--logs-root",
            str(logs_root),
        ]
    )

    assert code == 2


def test_train_dry_run_smoke_reports_missing_optional_deps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sft = tmp_path / "sft.jsonl"
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
    report = tmp_path / "training_preflight.json"
    monkeypatch.setattr("tensorfoundry.training.readiness.find_spec", lambda _name: None)

    code = learn_main(
        [
            "train",
            "--base-model",
            "dummy/base",
            "--sft",
            "--sft-data",
            str(sft),
            "--dry-run",
            "--smoke",
            "--max-steps",
            "1",
            "--report-out",
            str(report),
        ]
    )

    assert code == 2
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["smoke_evidence"]["version"] == "training_smoke.v0"
    assert payload["smoke_evidence"]["status"] == "blocked_missing_dependencies"
    assert payload["smoke_evidence"]["would_launch_training"] is False


def test_train_sft_run_writes_training_run_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    logs_root = tmp_path / "logs"
    sft, _dpo = _write_strict_training_datasets(tmp_path, logs_root)
    report = tmp_path / "training_preflight.json"
    run_report = tmp_path / "training_run.json"
    sft_out = tmp_path / "sft_lora"
    deps = {name: True for name in ("torch", "datasets", "transformers", "trl", "unsloth")}

    def fake_sft_training() -> None:
        out = Path(os.environ["OUT_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
        (out / "adapter_config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("tensorfoundry.learning.learn._run_sft_training", fake_sft_training)
    monkeypatch.setattr("tensorfoundry.learning.learn.check_optional_training_dependencies", lambda: deps)

    code = learn_main(
        [
            "train",
            "--base-model",
            "dummy/base",
            "--sft",
            "--sft-data",
            str(sft),
            "--sft-out",
            str(sft_out),
            "--quality-gate",
            "--logs-root",
            str(logs_root),
            "--report-out",
            str(report),
            "--run-report-out",
            str(run_report),
            "--smoke",
            "--max-steps",
            "1",
        ]
    )

    assert code == 0
    payload = json.loads(run_report.read_text(encoding="utf-8"))
    assert payload["version"] == "training_run.v0"
    assert payload["ok"] is True
    assert payload["status"] == "succeeded"
    assert payload["smoke_bounded"] is True
    assert payload["preflight_path"] == str(report)
    assert payload["artifact_refs"]["adapters"] == [str(sft_out)]
    assert str(sft_out / "adapter_model.safetensors") in payload["file_checksums"]


def test_train_sft_run_smoke_missing_deps_writes_blocked_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sft = tmp_path / "sft.jsonl"
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
    report = tmp_path / "training_run.json"
    deps = {name: False for name in ("torch", "datasets", "transformers", "trl", "unsloth")}

    def fail_if_called() -> None:
        raise AssertionError("trainer should not launch when smoke dependencies are missing")

    monkeypatch.setattr("tensorfoundry.learning.learn._run_sft_training", fail_if_called)
    monkeypatch.setattr("tensorfoundry.learning.learn.check_optional_training_dependencies", lambda: deps)

    code = learn_main(
        [
            "train",
            "--base-model",
            "dummy/base",
            "--sft",
            "--sft-data",
            str(sft),
            "--smoke",
            "--max-steps",
            "1",
            "--run-report-out",
            str(report),
        ]
    )

    assert code == 2
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["version"] == "training_run.v0"
    assert payload["ok"] is False
    assert payload["status"] == "blocked_missing_dependencies"
    assert any("missing optional training dependencies" in issue for issue in payload["issues"])


def test_failed_preflight_report_is_rejected(tmp_path: Path) -> None:
    report = tmp_path / "training_preflight.json"
    report.write_text(
        json.dumps({"version": "training_preflight.v0", "ok": False}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not ok"):
        load_training_preflight_report(report)


def test_training_output_scan_hashes_files_and_ignores_unset_outputs(tmp_path: Path) -> None:
    out = tmp_path / "sft_lora"
    out.mkdir()
    artifact = out / "adapter_model.safetensors"
    artifact.write_text("weights", encoding="utf-8")

    payload = scan_training_output_artifacts({"sft_out": str(out), "dpo_out": None})

    assert payload["artifact_refs"]["adapters"] == [str(out)]
    assert str(artifact) in payload["file_checksums"]
    assert payload["missing_outputs"] == []


def _write_strict_training_datasets(tmp_path: Path, logs_root: Path) -> tuple[Path, Path]:
    run_dir = logs_root / "run1"
    run_dir.mkdir(parents=True)
    trace_a = run_dir / "trace-a.jsonl"
    trace_b = run_dir / "trace-b.jsonl"
    reward = run_dir / "reward.jsonl"
    trace_a.write_text("{}\n", encoding="utf-8")
    trace_b.write_text("{}\n", encoding="utf-8")
    reward.write_text("{}\n", encoding="utf-8")

    sft = tmp_path / "sft.jsonl"
    sft.write_text(
        json.dumps(
            {
                "version": "sft.v0",
                "instruction": "Task: train",
                "prompt": "Task: train",
                "response": "A training response",
                "meta": {
                    **split_meta("suite", "case-sft"),
                    "provenance": {
                        "inputs": {
                            "trace_path": str(trace_a),
                            "reward_jsonl": str(reward),
                        }
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dpo = tmp_path / "dpo.jsonl"
    dpo.write_text(
        json.dumps(
            {
                "version": "dpo.v0",
                "prompt": "Task: compare",
                "chosen": "Better response",
                "rejected": "Worse response",
                "meta": {
                    **split_meta("suite", "case-dpo"),
                    "provenance": {
                        "inputs": {
                            "trace_a_path": str(trace_a),
                            "trace_b_path": str(trace_b),
                            "reward_a_jsonl": str(reward),
                            "reward_b_jsonl": str(reward),
                        }
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return sft, dpo
