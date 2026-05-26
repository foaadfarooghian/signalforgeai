from __future__ import annotations

import json
from pathlib import Path

import pytest

from signalforgeai.distillation.check import DISTILLATION_EVAL_VERSION
from signalforgeai.evaluation.matrix import BENCHMARK_MATRIX_VERSION
from signalforgeai.release.candidate import (
    RELEASE_CANDIDATE_VERSION,
    main as release_main,
    run_release_candidate_check,
    write_release_training_evidence,
)
from signalforgeai.training.readiness import (
    ADAPTER_SMOKE_VERSION,
    build_training_run,
    load_training_run_report,
    summarize_dpo_parent_run,
    utc_now,
    write_training_run,
)

REAL_SMOKE_MODEL = "unsloth/tinyllama-chat-bnb-4bit"


def test_mock_training_evidence_writes_sft_and_dpo_runs(tmp_path: Path) -> None:
    preflight_path = _write_preflight(tmp_path)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))

    payload = write_release_training_evidence(
        preflight=preflight,
        preflight_path=preflight_path,
        training_dir=tmp_path / "training",
        base_model="dummy/base",
    )

    assert payload["ok"] is True
    assert payload["mode"] == "mock"
    assert payload["final_stage"] == "dpo"
    assert Path(str(payload["sft_run_path"])).exists()
    assert Path(str(payload["dpo_run_path"])).exists()
    dpo = load_training_run_report(str(payload["final_run_path"]))
    assert dpo["training_stage"] == "dpo"
    assert dpo["command_config"]["training_evidence_mode"] == "mock"
    assert dpo["dpo_parent_run"]["adapter_refs"] == [str(tmp_path / "training" / "sft_lora")]
    assert dpo["artifact_refs"]["adapters"] == [str(tmp_path / "training" / "dpo_lora")]


def test_sft_run_override_generates_only_dpo_mock(tmp_path: Path) -> None:
    preflight_path = _write_preflight(tmp_path)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    initial = write_release_training_evidence(
        preflight=preflight,
        preflight_path=preflight_path,
        training_dir=tmp_path / "initial",
        base_model="dummy/base",
    )

    payload = write_release_training_evidence(
        preflight=preflight,
        preflight_path=preflight_path,
        training_dir=tmp_path / "override",
        base_model="dummy/base",
        sft_run=str(initial["sft_run_path"]),
    )

    assert payload["ok"] is True
    assert payload["mode"] == "external"
    assert Path(str(payload["dpo_run_path"])).exists()
    assert not (tmp_path / "override" / "sft_lora").exists()
    dpo = load_training_run_report(str(payload["final_run_path"]))
    assert dpo["dpo_parent_run"]["path"] == str(initial["sft_run_path"])


def test_dpo_run_override_is_selected_as_final(tmp_path: Path) -> None:
    preflight_path = _write_preflight(tmp_path)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    initial = write_release_training_evidence(
        preflight=preflight,
        preflight_path=preflight_path,
        training_dir=tmp_path / "initial",
        base_model="dummy/base",
    )

    payload = write_release_training_evidence(
        preflight=preflight,
        preflight_path=preflight_path,
        training_dir=tmp_path / "override",
        base_model="dummy/base",
        dpo_run=str(initial["dpo_run_path"]),
    )

    assert payload["ok"] is True
    assert payload["mode"] == "external_dpo"
    assert payload["real_training_evidence"] is False
    assert payload["final_run_path"] == str(Path(str(initial["dpo_run_path"])).resolve())
    assert not (tmp_path / "override" / "dpo_lora").exists()


def test_real_sft_final_evidence_can_be_required(tmp_path: Path) -> None:
    preflight_path = _write_preflight(tmp_path)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    real_sft_run = _write_real_sft_run(tmp_path, preflight_path)

    payload = write_release_training_evidence(
        preflight=preflight,
        preflight_path=preflight_path,
        training_dir=tmp_path / "override",
        base_model=REAL_SMOKE_MODEL,
        sft_run=real_sft_run,
        final_training_stage="sft",
        require_real_training_evidence=True,
    )

    assert payload["ok"] is True
    assert payload["mode"] == "external_sft"
    assert payload["real_training_evidence"] is True
    assert payload["final_stage"] == "sft"
    assert payload["final_run_path"] == str(real_sft_run.resolve())
    assert not (tmp_path / "override" / "dpo_lora").exists()


def test_real_sft_final_evidence_requires_file_checksums(tmp_path: Path) -> None:
    preflight_path = _write_preflight(tmp_path)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    real_sft_run = _write_real_sft_run_without_checksums(tmp_path, preflight_path)

    with pytest.raises(ValueError, match="file checksums"):
        write_release_training_evidence(
            preflight=preflight,
            preflight_path=preflight_path,
            training_dir=tmp_path / "override",
            base_model=REAL_SMOKE_MODEL,
            sft_run=real_sft_run,
            final_training_stage="sft",
            require_real_training_evidence=True,
        )


def test_training_run_success_requires_file_checksums(tmp_path: Path) -> None:
    out_dir = tmp_path / "empty_adapter"
    out_dir.mkdir()

    payload = build_training_run(
        base_model=REAL_SMOKE_MODEL,
        preflight=None,
        preflight_path=None,
        outputs={"sft_out": str(out_dir), "dpo_out": None, "sft_dir": None},
        optional_dependencies={},
        config={"source": "test", "sft": True, "dpo": False, "smoke": True, "max_steps": 1},
        status="succeeded",
        started_at=utc_now(),
        duration_seconds=0.0,
    )

    assert payload["ok"] is False
    assert any("file checksums" in issue for issue in payload["issues"])


def test_release_candidate_requires_real_training_evidence(tmp_path: Path) -> None:
    code = release_main(
        [
            "--work-dir",
            str(tmp_path),
            "--require-real-training-evidence",
        ]
    )

    payload = json.loads((tmp_path / "release_candidate.json").read_text(encoding="utf-8"))
    assert code == 1
    assert payload["ok"] is False
    assert payload["training_evidence"]["real_training_evidence"] is False
    assert any("real training evidence required" in issue for issue in payload["issues"])


def test_release_candidate_accepts_real_sft_as_final_training_evidence(tmp_path: Path) -> None:
    preflight_path = _write_preflight(tmp_path)
    real_sft_run = _write_real_sft_run(tmp_path, preflight_path)
    work_dir = tmp_path / "release"

    payload = run_release_candidate_check(
        work_dir=work_dir,
        sft_run=real_sft_run,
        final_training_stage="sft",
        require_real_training_evidence=True,
    )

    assert payload["ok"] is True
    assert payload["training_evidence"]["real_training_evidence"] is True
    assert payload["training_evidence"]["final_stage"] == "sft"
    assert payload["training_evidence"]["final_run_path"] == str(real_sft_run.resolve())
    assert payload["artifact_refs"]["adapters"] == [str(tmp_path / "real_sft_lora")]


def test_release_candidate_check_runs_offline_and_writes_bundle(tmp_path: Path) -> None:
    unrelated = tmp_path / "keep.txt"
    stale = tmp_path / "pilot" / "stale.txt"
    unrelated.write_text("keep", encoding="utf-8")
    stale.parent.mkdir(parents=True)
    stale.write_text("remove", encoding="utf-8")

    payload = run_release_candidate_check(work_dir=tmp_path)

    assert payload["version"] == RELEASE_CANDIDATE_VERSION
    assert payload["ok"] is True
    assert unrelated.exists()
    assert not stale.exists()
    assert (tmp_path / "release_candidate.json").exists()
    assert (tmp_path / "release_candidate.md").exists()
    assert (tmp_path / "pilot" / "pilot_readiness.json").exists()
    assert (tmp_path / "training" / "sft_training_run.json").exists()
    assert (tmp_path / "training" / "dpo_training_run.json").exists()
    assert (tmp_path / "distillation" / "distillation_eval.json").exists()
    assert (tmp_path / "benchmark" / "benchmark_matrix.json").exists()
    assert (tmp_path / "exchange" / "unit.json").exists()
    assert (tmp_path / "exchange" / "specialist_package.json").exists()
    assert (tmp_path / "exchange" / "smoke" / "specialist_smoke.json").exists()
    assert (tmp_path / "exchange" / "index.json").exists()
    assert payload["training_evidence"]["final_stage"] == "dpo"
    assert payload["artifact_refs"]["adapters"] == [str(tmp_path / "training" / "dpo_lora")]
    assert {gate["name"] for gate in payload["gate_results"]} == {
        "pilot_readiness",
        "training_evidence",
        "distillation_eval",
        "benchmark_matrix",
        "specialist_unit",
        "package_check",
        "smoke_run",
        "registry_index",
    }


def test_release_candidate_default_suite_runs_from_arbitrary_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    work_dir = tmp_path / "release"
    monkeypatch.chdir(cwd)

    code = release_main(["--work-dir", str(work_dir)])

    payload = json.loads((work_dir / "release_candidate.json").read_text(encoding="utf-8"))
    assert code == 0
    assert payload["ok"] is True
    assert payload["command_config"]["suites"] == ["decision_v0"]
    assert Path(payload["evidence_paths"]["benchmark_matrix"]).exists()


def test_release_candidate_degraded_candidate_exits_nonzero(tmp_path: Path) -> None:
    code = release_main(
        [
            "--work-dir",
            str(tmp_path),
            "--candidate-model-id",
            "dummy_bad",
        ]
    )

    payload = json.loads((tmp_path / "release_candidate.json").read_text(encoding="utf-8"))
    assert code == 1
    assert payload["ok"] is False
    distill_gate = next(gate for gate in payload["gate_results"] if gate["name"] == "distillation_eval")
    assert distill_gate["ok"] is False
    assert any("mean_score_drop" in issue for issue in payload["issues"])


def test_release_candidate_run_training_derives_hf_candidate_model_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, str] = {}
    _install_real_training_fakes(tmp_path, monkeypatch, captured)

    payload = run_release_candidate_check(
        work_dir=tmp_path,
        training_base_model=REAL_SMOKE_MODEL,
        run_training=True,
        require_real_training_evidence=True,
    )

    expected = f"hf:{REAL_SMOKE_MODEL}?adapter={tmp_path / 'training' / 'sft_lora'}"
    assert payload["ok"] is True
    assert payload["command_config"]["candidate_model_id"] == expected
    assert payload["training_evidence"]["mode"] == "run"
    assert payload["training_evidence"]["final_stage"] == "sft"
    assert payload["training_evidence"]["adapter_smoke"]["ok"] is True
    assert payload["training_evidence"]["derived_candidate_model_id"] == expected
    assert captured["distill_candidate_model_id"] == expected
    assert captured["matrix_model_ids"].split(",").count(expected) == 1
    assert payload["artifact_refs"]["adapters"] == [str(tmp_path / "training" / "sft_lora")]


def test_release_candidate_run_dpo_selects_dpo_as_final_training_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, str] = {}
    _install_real_training_fakes(tmp_path, monkeypatch, captured)

    payload = run_release_candidate_check(
        work_dir=tmp_path,
        training_base_model=REAL_SMOKE_MODEL,
        run_training=True,
        run_dpo=True,
        require_real_training_evidence=True,
    )

    expected = f"hf:{REAL_SMOKE_MODEL}?adapter={tmp_path / 'training' / 'dpo_lora'}"
    assert payload["ok"] is True
    assert payload["command_config"]["candidate_model_id"] == expected
    assert payload["training_evidence"]["final_stage"] == "dpo"
    assert payload["training_evidence"]["derived_candidate_model_id"] == expected
    assert payload["artifact_refs"]["adapters"] == [str(tmp_path / "training" / "dpo_lora")]


def test_release_candidate_run_training_respects_explicit_candidate_model_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, str] = {}
    _install_real_training_fakes(tmp_path, monkeypatch, captured)

    payload = run_release_candidate_check(
        work_dir=tmp_path,
        training_base_model=REAL_SMOKE_MODEL,
        candidate_model_id="dummy_good",
        run_training=True,
        require_real_training_evidence=True,
    )

    assert payload["ok"] is True
    assert payload["command_config"]["candidate_model_id"] == "dummy_good"
    assert payload["training_evidence"]["derived_candidate_model_id"].startswith(f"hf:{REAL_SMOKE_MODEL}")
    assert captured["distill_candidate_model_id"] == "dummy_good"


def test_release_candidate_run_training_rejects_external_training_reports(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        run_release_candidate_check(
            work_dir=tmp_path,
            run_training=True,
            sft_run=tmp_path / "sft_training_run.json",
        )


def test_release_candidate_run_training_rejects_placeholder_base_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Hugging Face base model"):
        run_release_candidate_check(
            work_dir=tmp_path,
            training_base_model="dummy/base",
            run_training=True,
            require_real_training_evidence=True,
        )


def test_release_candidate_run_training_rejects_padded_placeholder_base_model(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Hugging Face base model"):
        run_release_candidate_check(
            work_dir=tmp_path,
            training_base_model=" dummy/base ",
            run_training=True,
            require_real_training_evidence=True,
        )


def test_release_candidate_cli_rejects_placeholder_base_model_before_run(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as exc:
        release_main(
            [
                "--work-dir",
                str(tmp_path),
                "--run-training",
                "--training-base-model",
                " dummy/base ",
            ]
        )

    assert exc.value.code == 2
    assert not (tmp_path / "release_candidate.json").exists()


def test_release_candidate_run_training_failure_fails_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_real_training_fakes(tmp_path, monkeypatch, {})

    def fail_training(config) -> int:
        return 2

    monkeypatch.setattr("signalforgeai.release.candidate.run_training_job", fail_training)

    payload = run_release_candidate_check(
        work_dir=tmp_path,
        training_base_model=REAL_SMOKE_MODEL,
        run_training=True,
        require_real_training_evidence=True,
    )

    assert payload["ok"] is False
    assert payload["training_evidence"]["ok"] is False
    assert any("SFT training exited nonzero" in issue for issue in payload["issues"])


def _write_preflight(tmp_path: Path) -> Path:
    path = tmp_path / "training_preflight.json"
    logs_root = tmp_path / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    sft_dataset = tmp_path / "pilot.sft.jsonl"
    dpo_dataset = tmp_path / "pilot.dpo.jsonl"
    sft_dataset.write_text('{"version":"sft.v0"}\n', encoding="utf-8")
    dpo_dataset.write_text('{"version":"dpo.v0"}\n', encoding="utf-8")
    payload = {
        "version": "training_preflight.v0",
        "ok": True,
        "base_model": "dummy/base",
        "quality_gate": True,
        "logs_root": str(logs_root),
        "datasets": [
            {
                "role": "sft",
                "kind": "sft",
                "path": str(sft_dataset),
                "rows": 1,
                "ok": True,
                "content_sha256": "a" * 64,
                "split_counts": {"train": 1},
            },
            {
                "role": "dpo",
                "kind": "dpo",
                "path": str(dpo_dataset),
                "rows": 1,
                "ok": True,
                "content_sha256": "b" * 64,
                "split_counts": {"train": 1},
            },
        ],
        "artifact_manifest": {
            "version": "training_artifact.v0",
            "base_model": "dummy/base",
            "dataset_sources": {"sft": str(sft_dataset), "dpo": str(dpo_dataset)},
            "dataset_hashes": {"sft": "a" * 64, "dpo": "b" * 64},
            "split_counts": {"sft": {"train": 1}, "dpo": {"train": 1}},
            "logs_root": str(logs_root),
            "outputs": {
                "sft_out": str(tmp_path / "training" / "sft_lora"),
                "dpo_out": str(tmp_path / "training" / "dpo_lora"),
            },
        },
        "optional_dependencies": {
            "torch": False,
            "datasets": False,
            "transformers": False,
            "trl": False,
            "unsloth": False,
        },
        "issues": [],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_real_sft_run(tmp_path: Path, preflight_path: Path) -> Path:
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    out_dir = tmp_path / "real_sft_lora"
    out_dir.mkdir(parents=True)
    (out_dir / "adapter_model.safetensors").write_text("real-sft-weights", encoding="utf-8")
    payload = build_training_run(
        base_model=REAL_SMOKE_MODEL,
        preflight=preflight,
        preflight_path=preflight_path,
        outputs={"sft_out": str(out_dir), "dpo_out": None, "sft_dir": None},
        optional_dependencies={},
        config={
            "source": "test-real-sft",
            "sft": True,
            "dpo": False,
            "smoke": True,
            "max_steps": 1,
            "quality_gate": True,
        },
        status="succeeded",
        started_at=utc_now(),
        duration_seconds=0.0,
        adapter_smoke={
            "version": ADAPTER_SMOKE_VERSION,
            "ok": True,
            "base_model": REAL_SMOKE_MODEL,
            "adapter_path": str(out_dir),
            "prompt_type": "single_turn_json",
            "latency_ms": 1,
            "details": {"load_label": "test"},
            "reason": "",
        },
    )
    return write_training_run(payload, tmp_path / "real_sft_training_run.json")


def _write_real_sft_run_without_checksums(tmp_path: Path, preflight_path: Path) -> Path:
    path = _write_real_sft_run(tmp_path, preflight_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["file_checksums"] = {}
    payload["output_artifacts"] = []
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _install_real_training_fakes(tmp_path: Path, monkeypatch, captured: dict[str, str]) -> None:
    def fake_pilot_check(*, work_dir, **_kwargs):
        pilot_dir = Path(work_dir)
        pilot_dir.mkdir(parents=True, exist_ok=True)
        _write_preflight(pilot_dir)
        report = pilot_dir / "pilot_readiness.json"
        report.write_text(json.dumps({"ok": True, "issues": []}, indent=2), encoding="utf-8")
        return {"ok": True, "report_json": str(report), "providers": [], "issues": []}

    def fake_training(config) -> int:
        out_dir = Path(config.dpo_out if config.dpo else config.sft_out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "adapter_model.safetensors").write_text("real-weights", encoding="utf-8")
        (out_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        dpo_parent = summarize_dpo_parent_run(config.sft_run) if config.dpo else None
        payload = build_training_run(
            base_model=config.base_model,
            preflight=None,
            preflight_path=config.report_out or None,
            outputs={
                "sft_out": config.sft_out if config.sft else None,
                "dpo_out": config.dpo_out if config.dpo else None,
                "sft_dir": dpo_parent["adapter_refs"][0] if dpo_parent else None,
            },
            optional_dependencies={
                "torch": True,
                "datasets": True,
                "transformers": True,
                "trl": True,
                "unsloth": True,
            },
            config={
                "source": "test-real-training",
                "sft": bool(config.sft),
                "dpo": bool(config.dpo),
                "smoke": bool(config.smoke),
                "max_steps": int(config.max_steps),
                "quality_gate": bool(config.quality_gate),
            },
            dpo_parent_run=dpo_parent,
            status="succeeded",
            started_at=utc_now(),
            duration_seconds=0.0,
            adapter_smoke={
                "version": ADAPTER_SMOKE_VERSION,
                "ok": True,
                "base_model": config.base_model,
                "adapter_path": str(out_dir),
                "prompt_type": "single_turn_json",
                "latency_ms": 1,
                "details": {"load_label": "test"},
                "reason": "",
            },
        )
        write_training_run(payload, config.run_report_out)
        return 0

    def fake_distillation_check(*, recipe_path, work_dir):
        recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
        captured["distill_candidate_model_id"] = recipe["candidate_model_id"]
        suite_name = _suite_name(Path(recipe["suite"]))
        payload = {
            "version": DISTILLATION_EVAL_VERSION,
            "ok": True,
            "recipe": recipe,
            "work_dir": str(work_dir),
            "training_preflight": {"ok": True},
            "baseline": {"suite_name": suite_name, "model_id": recipe["baseline_model_id"]},
            "candidate": {
                "suite_name": suite_name,
                "model_id": recipe["candidate_model_id"],
                "mean_score": 1.0,
                "pass_rate": 1.0,
            },
            "aggregate_deltas": {},
            "gate_results": [],
            "changed_cases": [],
            "regressed_cases": [],
            "improved_cases": [],
            "missing_cases": [],
            "new_cases": [],
            "failure_mode_movements": [],
            "worse_failure_mode_movements": [],
            "artifact_refs": {},
            "issues": [],
        }
        out = Path(work_dir) / "distillation_eval.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (Path(work_dir) / "distillation_eval.md").write_text("# Distillation\n", encoding="utf-8")
        payload["report_json"] = str(out)
        return payload

    def fake_benchmark_matrix(*, config_path, work_dir, suites=None, model_ids=None, **_kwargs):
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        ids = [str(value) for value in (model_ids or config["model_ids"])]
        captured["matrix_model_ids"] = ",".join(ids)
        suite_path = Path((suites or config["suites"])[0])
        suite_name = _suite_name(suite_path)
        rows = [
            {
                "suite": suite_name,
                "suite_path": str(suite_path),
                "model_id": model_id,
                "provider": "hf" if model_id.startswith("hf:") else "dummy",
                "provider_required": model_id.startswith("hf:"),
                "provider_ok": True,
                "provider_skipped": False,
                "skipped": False,
                "ok": True,
                "issue": None,
                "result_path": None,
                "summary_path": None,
                "run_logs_dir": None,
                "metrics": {
                    "cases": 1,
                    "pass_rate": 1.0,
                    "mean_score": 1.0,
                    "cost_per_success_usd": 0.0,
                    "latency_ms_p50": 0.0,
                    "latency_ms_p95": 0.0,
                    "failure_rate": 0.0,
                    "retry_rate": 0.0,
                    "mean_effective": 1.0,
                },
            }
            for model_id in ids
        ]
        out = Path(work_dir) / "benchmark_matrix.json"
        payload = {
            "version": BENCHMARK_MATRIX_VERSION,
            "ok": True,
            "config": config,
            "scorecard": rows,
            "frontiers": {},
            "artifact_refs": {"json": str(out), "markdown": str(Path(work_dir) / "benchmark_matrix.md")},
            "issues": [],
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (Path(work_dir) / "benchmark_matrix.md").write_text("# Benchmark\n", encoding="utf-8")
        return payload

    monkeypatch.setattr("signalforgeai.release.candidate.run_pilot_check", fake_pilot_check)
    monkeypatch.setattr("signalforgeai.release.candidate.run_training_job", fake_training)
    monkeypatch.setattr("signalforgeai.release.candidate.run_distillation_check", fake_distillation_check)
    monkeypatch.setattr("signalforgeai.release.candidate.run_benchmark_matrix", fake_benchmark_matrix)


def _suite_name(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload.get("suite_name") or path.stem)
