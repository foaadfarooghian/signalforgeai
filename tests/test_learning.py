from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import pytest

from signalforgeai.export.quality import split_meta
from signalforgeai.learning.bandits import RoutingBanditsV0, candidate_models_from_env
from signalforgeai.learning.build_policy import build_routing_policy_v0
from signalforgeai.learning.learn import _adapter_smoke_subprocess, main as learn_main
from signalforgeai.learning.model_stats import RoutingStatsV0
from signalforgeai.learning.routing_policy import RoutingPolicyV0
from signalforgeai.training.readiness import (
    ADAPTER_SMOKE_VERSION,
    load_training_preflight_report,
    scan_training_output_artifacts,
)

REAL_SMOKE_MODEL = "unsloth/tinyllama-chat-bnb-4bit"


def test_bandits_update_from_rewards_creates_arm() -> None:
    bandits = RoutingBanditsV0()
    bandits.update_from_rewards("suite_a", "model_a", [1.0, 0.0])
    arm = bandits.by_suite["suite_a"]["model_a"]
    assert arm.alpha == 2.0
    assert arm.beta == 2.0


def test_candidate_models_from_env_fallback(monkeypatch) -> None:
    monkeypatch.delenv("SIGNALFORGEAI_CANDIDATE_MODELS", raising=False)
    monkeypatch.setenv("SIGNALFORGEAI_MODEL_ID", "dummy_good")
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
            REAL_SMOKE_MODEL,
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
    monkeypatch.setattr("signalforgeai.training.readiness.find_spec", lambda _name: None)

    code = learn_main(
        [
            "train",
            "--base-model",
            REAL_SMOKE_MODEL,
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
    monkeypatch.setenv("SIGNALFORGEAI_ALLOW_UNSUPPORTED_TRAINING_PLATFORM", "1")
    monkeypatch.delenv("SIGNALFORGEAI_TRAINING_MAX_SEQ_LENGTH", raising=False)
    monkeypatch.delenv("SIGNALFORGEAI_TRAINING_BATCH_SIZE", raising=False)
    monkeypatch.delenv("SIGNALFORGEAI_TRAINING_GRADIENT_ACCUMULATION_STEPS", raising=False)

    def fake_sft_training() -> None:
        assert os.environ["SIGNALFORGEAI_TRAINING_SMOKE"] == "1"
        assert os.environ["SIGNALFORGEAI_TRAINING_MAX_SEQ_LENGTH"] == "512"
        assert os.environ["SIGNALFORGEAI_TRAINING_BATCH_SIZE"] == "1"
        assert os.environ["SIGNALFORGEAI_TRAINING_GRADIENT_ACCUMULATION_STEPS"] == "1"
        out = Path(os.environ["OUT_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
        (out / "adapter_config.json").write_text("{}", encoding="utf-8")

    def fake_adapter_smoke(_args, adapter_path: str) -> dict[str, object]:
        return {
            "version": ADAPTER_SMOKE_VERSION,
            "ok": True,
            "base_model": REAL_SMOKE_MODEL,
            "adapter_path": adapter_path,
            "prompt_type": "single_turn_json",
            "latency_ms": 1,
            "details": {"load_label": "test"},
            "reason": "",
        }

    monkeypatch.setattr("signalforgeai.learning.learn._run_sft_training", fake_sft_training)
    monkeypatch.setattr("signalforgeai.learning.learn._adapter_load_generate_smoke", fake_adapter_smoke)
    monkeypatch.setattr("signalforgeai.learning.learn.check_optional_training_dependencies", lambda: deps)

    code = learn_main(
        [
            "train",
            "--base-model",
            REAL_SMOKE_MODEL,
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
    assert payload["adapter_smoke"]["ok"] is True
    assert payload["adapter_smoke"]["adapter_path"] == str(sft_out)
    assert payload["preflight_path"] == str(report)
    assert payload["artifact_refs"]["adapters"] == [str(sft_out)]
    assert str(sft_out / "adapter_model.safetensors") in payload["file_checksums"]


def test_adapter_smoke_subprocess_parses_last_json_line(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        payload = {
            "version": ADAPTER_SMOKE_VERSION,
            "ok": True,
            "base_model": REAL_SMOKE_MODEL,
            "adapter_path": "/tmp/adapter",
            "prompt_type": "single_turn_json",
            "latency_ms": 1,
            "details": {},
            "reason": "",
        }
        return subprocess.CompletedProcess(cmd, 0, stdout="note\n" + json.dumps(payload) + "\n", stderr="")

    monkeypatch.setattr("signalforgeai.learning.learn.subprocess.run", fake_run)

    payload = _adapter_smoke_subprocess(
        argparse.Namespace(base_model=REAL_SMOKE_MODEL),
        "/tmp/adapter",
    )

    assert payload is not None
    assert payload["ok"] is True
    assert captured["env"]["SIGNALFORGEAI_ADAPTER_SMOKE_IN_PROCESS"] == "1"
    assert captured["env"]["SIGNALFORGEAI_HF_LOG_DEVICE_MAP"] == "0"


def test_train_dpo_run_uses_successful_sft_run_parent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    logs_root = tmp_path / "logs"
    _sft, dpo = _write_strict_training_datasets(tmp_path, logs_root)
    sft_out = tmp_path / "sft_lora"
    sft_run = _write_sft_run_report(tmp_path, sft_out)
    dpo_out = tmp_path / "dpo_lora"
    run_report = tmp_path / "training_run.json"
    deps = {name: True for name in ("torch", "datasets", "transformers", "trl", "unsloth")}
    monkeypatch.setenv("SIGNALFORGEAI_ALLOW_UNSUPPORTED_TRAINING_PLATFORM", "1")

    def fake_dpo_training() -> None:
        assert os.environ["SFT_DIR"] == str(sft_out)
        out = Path(os.environ["OUT_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_model.safetensors").write_text("dpo-weights", encoding="utf-8")

    def fake_adapter_smoke(_args, adapter_path: str) -> dict[str, object]:
        return {
            "version": ADAPTER_SMOKE_VERSION,
            "ok": True,
            "base_model": REAL_SMOKE_MODEL,
            "adapter_path": adapter_path,
            "prompt_type": "single_turn_json",
            "latency_ms": 1,
            "details": {"load_label": "test"},
            "reason": "",
        }

    def fail_sft_training() -> None:
        raise AssertionError("SFT should not run for DPO-only execution")

    monkeypatch.setattr("signalforgeai.learning.learn._run_sft_training", fail_sft_training)
    monkeypatch.setattr("signalforgeai.learning.learn._run_dpo_training", fake_dpo_training)
    monkeypatch.setattr("signalforgeai.learning.learn._adapter_load_generate_smoke", fake_adapter_smoke)
    monkeypatch.setattr("signalforgeai.learning.learn.check_optional_training_dependencies", lambda: deps)

    code = learn_main(
        [
            "train",
            "--base-model",
            REAL_SMOKE_MODEL,
            "--dpo",
            "--dpo-data",
            str(dpo),
            "--sft-run",
            str(sft_run),
            "--dpo-out",
            str(dpo_out),
            "--quality-gate",
            "--logs-root",
            str(logs_root),
            "--run-report-out",
            str(run_report),
            "--smoke",
            "--max-steps",
            "1",
        ]
    )

    assert code == 0
    payload = json.loads(run_report.read_text(encoding="utf-8"))
    assert payload["training_stage"] == "dpo"
    assert payload["dpo_parent_run"]["path"] == str(sft_run)
    assert payload["adapter_smoke"]["ok"] is True
    assert payload["adapter_smoke"]["adapter_path"] == str(dpo_out)
    assert payload["artifact_refs"]["adapters"] == [str(dpo_out)]
    assert payload["final_adapter_refs"] == [str(dpo_out)]
    assert str(dpo_out / "adapter_model.safetensors") in payload["file_checksums"]


def test_train_dpo_run_report_requires_sft_run_parent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dpo = tmp_path / "dpo.jsonl"
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
    report = tmp_path / "training_run.json"
    deps = {name: True for name in ("torch", "datasets", "transformers", "trl", "unsloth")}
    monkeypatch.setenv("SIGNALFORGEAI_ALLOW_UNSUPPORTED_TRAINING_PLATFORM", "1")

    def fail_if_called() -> None:
        raise AssertionError("DPO should not launch without parent SFT evidence")

    monkeypatch.setattr("signalforgeai.learning.learn._run_dpo_training", fail_if_called)
    monkeypatch.setattr("signalforgeai.learning.learn.check_optional_training_dependencies", lambda: deps)

    code = learn_main(
        [
            "train",
            "--base-model",
            REAL_SMOKE_MODEL,
            "--dpo",
            "--dpo-data",
            str(dpo),
            "--run-report-out",
            str(report),
        ]
    )

    assert code == 2
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked_dpo_parent_run"
    assert any("--sft-run is required" in issue for issue in payload["issues"])


def test_train_dpo_run_report_rejects_failed_sft_parent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _sft, dpo = _write_strict_training_datasets(tmp_path, tmp_path / "logs")
    sft_run = tmp_path / "failed_sft_training_run.json"
    sft_run.write_text(
        json.dumps(
            {
                "version": "training_run.v0",
                "ok": False,
                "status": "failed",
                "artifact_refs": {"adapters": [str(tmp_path / "sft_lora")]},
                "issues": ["SFT failed"],
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "training_run.json"
    deps = {name: True for name in ("torch", "datasets", "transformers", "trl", "unsloth")}
    monkeypatch.setenv("SIGNALFORGEAI_ALLOW_UNSUPPORTED_TRAINING_PLATFORM", "1")

    def fail_if_called() -> None:
        raise AssertionError("DPO should not launch with failed parent SFT evidence")

    monkeypatch.setattr("signalforgeai.learning.learn._run_dpo_training", fail_if_called)
    monkeypatch.setattr("signalforgeai.learning.learn.check_optional_training_dependencies", lambda: deps)

    code = learn_main(
        [
            "train",
            "--base-model",
            REAL_SMOKE_MODEL,
            "--dpo",
            "--dpo-data",
            str(dpo),
            "--sft-run",
            str(sft_run),
            "--run-report-out",
            str(report),
        ]
    )

    assert code == 2
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked_dpo_parent_run"
    assert any("training run is not ok" in issue for issue in payload["issues"])


def test_train_sft_run_report_requires_real_base_model(
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
    monkeypatch.setenv("SIGNALFORGEAI_ALLOW_UNSUPPORTED_TRAINING_PLATFORM", "1")

    def fail_if_called() -> None:
        raise AssertionError("trainer should not launch for placeholder base models")

    monkeypatch.setattr("signalforgeai.learning.learn._run_sft_training", fail_if_called)

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
    assert payload["status"] == "blocked_invalid_base_model"
    assert any("Hugging Face base model" in issue for issue in payload["issues"])


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
    monkeypatch.setenv("SIGNALFORGEAI_ALLOW_UNSUPPORTED_TRAINING_PLATFORM", "1")

    def fail_if_called() -> None:
        raise AssertionError("trainer should not launch when smoke dependencies are missing")

    monkeypatch.setattr("signalforgeai.learning.learn._run_sft_training", fail_if_called)
    monkeypatch.setattr("signalforgeai.learning.learn.check_optional_training_dependencies", lambda: deps)

    code = learn_main(
        [
            "train",
            "--base-model",
            REAL_SMOKE_MODEL,
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


def test_train_sft_run_blocks_unsupported_platform(
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
    deps = {name: True for name in ("torch", "datasets", "transformers", "trl", "unsloth")}

    def fail_if_called() -> None:
        raise AssertionError("trainer should not launch on an unsupported platform")

    monkeypatch.setattr("signalforgeai.learning.learn.platform.system", lambda: "Darwin")
    monkeypatch.setattr("signalforgeai.learning.learn._run_sft_training", fail_if_called)
    monkeypatch.setattr("signalforgeai.learning.learn.check_optional_training_dependencies", lambda: deps)

    code = learn_main(
        [
            "train",
            "--base-model",
            REAL_SMOKE_MODEL,
            "--sft",
            "--sft-data",
            str(sft),
            "--run-report-out",
            str(report),
        ]
    )

    assert code == 2
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked_unsupported_platform"
    assert any("Linux-only" in issue for issue in payload["issues"])


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


def test_training_output_scan_prefers_dpo_adapter_as_final(tmp_path: Path) -> None:
    sft = tmp_path / "sft_lora"
    dpo = tmp_path / "dpo_lora"
    sft.mkdir()
    dpo.mkdir()
    (sft / "adapter_model.safetensors").write_text("sft", encoding="utf-8")
    (dpo / "adapter_model.safetensors").write_text("dpo", encoding="utf-8")

    payload = scan_training_output_artifacts({"sft_out": str(sft), "dpo_out": str(dpo)})

    assert payload["artifact_refs_by_role"]["sft_out"] == [str(sft)]
    assert payload["artifact_refs_by_role"]["dpo_out"] == [str(dpo)]
    assert payload["artifact_refs"]["adapters"] == [str(dpo)]
    assert payload["final_adapter_refs"] == [str(dpo)]


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


def _write_sft_run_report(tmp_path: Path, sft_out: Path) -> Path:
    sft_out.mkdir(parents=True)
    artifact = sft_out / "adapter_model.safetensors"
    artifact.write_text("sft-weights", encoding="utf-8")
    digest = _sha256(artifact)
    report = tmp_path / "sft_training_run.json"
    report.write_text(
        json.dumps(
            {
                "version": "training_run.v0",
                "ok": True,
                "status": "succeeded",
                "base_model": "dummy/base",
                "training_stage": "sft",
                "artifact_refs": {
                    "adapters": [str(sft_out)],
                    "safetensors": [],
                    "gguf": [],
                    "ollama": {"modelfile": "", "tag": ""},
                },
                "final_adapter_refs": [str(sft_out)],
                "file_checksums": {str(artifact): digest},
                "issues": [],
            }
        ),
        encoding="utf-8",
    )
    return report


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
