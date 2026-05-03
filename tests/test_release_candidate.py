from __future__ import annotations

import json
from pathlib import Path

from tensorfoundry.release.candidate import (
    RELEASE_CANDIDATE_VERSION,
    main as release_main,
    run_release_candidate_check,
    write_release_training_evidence,
)
from tensorfoundry.training.readiness import load_training_run_report


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
    assert payload["mode"] == "external"
    assert payload["final_run_path"] == str(Path(str(initial["dpo_run_path"])).resolve())
    assert not (tmp_path / "override" / "dpo_lora").exists()


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


def _write_preflight(tmp_path: Path) -> Path:
    path = tmp_path / "training_preflight.json"
    payload = {
        "version": "training_preflight.v0",
        "ok": True,
        "base_model": "dummy/base",
        "quality_gate": True,
        "datasets": [
            {
                "role": "sft",
                "kind": "sft",
                "path": str(tmp_path / "pilot.sft.jsonl"),
                "rows": 1,
                "ok": True,
                "content_sha256": "a" * 64,
                "split_counts": {"train": 1},
            },
            {
                "role": "dpo",
                "kind": "dpo",
                "path": str(tmp_path / "pilot.dpo.jsonl"),
                "rows": 1,
                "ok": True,
                "content_sha256": "b" * 64,
                "split_counts": {"train": 1},
            },
        ],
        "artifact_manifest": {
            "version": "training_artifact.v0",
            "base_model": "dummy/base",
            "dataset_hashes": {"sft": "a" * 64, "dpo": "b" * 64},
            "split_counts": {"sft": {"train": 1}, "dpo": {"train": 1}},
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
