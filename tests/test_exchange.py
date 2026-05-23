from __future__ import annotations

import json
from pathlib import Path

from signalforgeai.exchange.cli import main as exchange_main
from signalforgeai.exchange.package import SPECIALIST_PACKAGE_VERSION, run_package_check
from signalforgeai.exchange.registry import build_registry_index
from signalforgeai.exchange.smoke import SPECIALIST_SMOKE_VERSION, run_smoke_check
from signalforgeai.exchange.unit import build_specialist_unit
from signalforgeai.exchange.validation import (
    SPECIALIST_MODEL_UNIT_VERSION,
    sha256_file,
    validate_manifest,
)
from signalforgeai.training.readiness import TRAINING_RUN_VERSION


def test_valid_specialist_unit_passes_release_ready_validation(tmp_path: Path) -> None:
    paths = _write_evidence(tmp_path)
    artifact = tmp_path / "artifacts" / "model.safetensors"
    artifact.parent.mkdir()
    artifact.write_text("weights", encoding="utf-8")

    manifest = _build_unit(tmp_path, paths, artifact_ref="model.safetensors")
    report = validate_manifest(
        manifest,
        artifacts_root=artifact.parent,
        release_ready=True,
    )

    assert report.ok is True
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SPECIALIST_MODEL_UNIT_VERSION
    assert payload["artifacts"]["checksums"]["model.safetensors"] == sha256_file(artifact)


def test_manifest_missing_required_fields_fails_clearly(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps({"schema_version": SPECIALIST_MODEL_UNIT_VERSION}), encoding="utf-8")

    report = validate_manifest(manifest)

    assert report.ok is False
    assert any("schema" in issue and "required" in issue for issue in report.issues)


def test_local_artifact_checksum_mismatch_fails(tmp_path: Path) -> None:
    paths = _write_evidence(tmp_path)
    artifact = tmp_path / "artifacts" / "model.safetensors"
    artifact.parent.mkdir()
    artifact.write_text("weights", encoding="utf-8")
    manifest = _build_unit(tmp_path, paths, artifact_ref="model.safetensors")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifacts"]["checksums"]["model.safetensors"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_manifest(manifest, artifacts_root=artifact.parent)

    assert report.ok is False
    assert any("checksum mismatch" in issue for issue in report.issues)


def test_uri_artifact_refs_validate_without_network(tmp_path: Path) -> None:
    paths = _write_evidence(tmp_path)
    manifest = _build_unit(
        tmp_path,
        paths,
        artifact_ref="hf://signalforgeai/example/model.safetensors",
    )

    report = validate_manifest(manifest, release_ready=True)

    assert report.ok is True
    assert any(row["kind"] == "uri" for row in report.artifacts_checked)


def test_release_ready_fails_without_runnable_artifact_or_evidence_links(tmp_path: Path) -> None:
    manifest = tmp_path / "unit.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": SPECIALIST_MODEL_UNIT_VERSION,
                "id": "unit-example",
                "name": "Unit Example",
                "version": "0.1.0",
                "domain": "pilot",
                "model": {"family": "dummy", "size": "0B", "format": "other"},
                "eval_pack": {
                    "suite_id": "suite",
                    "report_path": str(tmp_path / "missing.md"),
                    "score": 0.9,
                },
                "lineage": {
                    "trace_sources": [str(tmp_path / "logs")],
                    "dataset_sources": [str(tmp_path / "dataset.jsonl")],
                    "distillation_recipe": str(tmp_path / "recipe.json"),
                },
                "hardware_profile": {"target": "cpu", "ram_gb": 0, "latency_ms_p50": 0},
                "failure_modes": [{"id": "known", "description": "Known issue", "mitigation": "Monitor"}],
                "license": {"model_license": "Apache-2.0", "dataset_license": "CC-BY-4.0"},
                "usage_constraints": [],
                "artifacts": {
                    "adapters": [],
                    "safetensors": [],
                    "gguf": [],
                    "ollama": {"modelfile": "", "tag": ""},
                    "checksums": {},
                },
            }
        ),
        encoding="utf-8",
    )

    report = validate_manifest(manifest, release_ready=True)

    assert report.ok is False
    assert any("runnable artifact" in issue for issue in report.issues)
    assert any("evidence links" in issue for issue in report.issues)


def test_build_unit_maps_training_distillation_and_benchmark_evidence(tmp_path: Path) -> None:
    paths = _write_evidence(tmp_path)
    manifest = _build_unit(
        tmp_path,
        paths,
        artifact_ref="hf://signalforgeai/example/model.safetensors",
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["lineage"]["dataset_hashes"]["sft"] == "a" * 64
    assert payload["eval_pack"]["score"] == 0.95
    assert payload["eval_pack"]["latency_ms_p50"] == 42
    assert payload["eval_pack"]["distillation_eval_path"] == str(paths["distillation_eval"].resolve())
    assert payload["eval_pack"]["benchmark_matrix_path"] == str(paths["benchmark_matrix"].resolve())


def test_build_unit_maps_training_run_evidence_and_artifacts(tmp_path: Path) -> None:
    paths = _write_evidence(tmp_path)
    manifest = _build_unit(
        tmp_path,
        paths,
        artifact_ref="",
        training_run_path=paths["training_run"],
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    run_payload = json.loads(paths["training_run"].read_text(encoding="utf-8"))
    adapter_dir = run_payload["artifact_refs"]["adapters"][0]
    adapter_file = run_payload["output_artifacts"][0]["path"]

    assert payload["training_run_evidence"]["version"] == TRAINING_RUN_VERSION
    assert payload["training_run_evidence"]["path"] == str(paths["training_run"].resolve())
    assert payload["artifacts"]["adapters"] == [adapter_dir]
    assert payload["artifacts"]["checksums"][adapter_file] == run_payload["file_checksums"][adapter_file]
    assert validate_manifest(manifest, release_ready=True).ok is True


def test_build_unit_prefers_dpo_adapter_from_training_run(tmp_path: Path) -> None:
    paths = _write_evidence(tmp_path)
    manifest = _build_unit(
        tmp_path,
        paths,
        artifact_ref="",
        training_run_path=paths["training_run_dpo"],
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    run_payload = json.loads(paths["training_run_dpo"].read_text(encoding="utf-8"))

    assert payload["training_run_evidence"]["status"] == "succeeded"
    assert payload["artifacts"]["adapters"] == run_payload["final_adapter_refs"]
    assert payload["artifacts"]["adapters"] == [str(paths["dpo_out"])]


def test_registry_index_rejects_duplicate_unit_versions(tmp_path: Path) -> None:
    paths = _write_evidence(tmp_path)
    first = _build_unit(tmp_path / "first", paths, artifact_ref="hf://example/model.safetensors")
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second = second_dir / "unit.json"
    second.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")

    index = build_registry_index(
        registry_dir=tmp_path,
        out_path=tmp_path / "index.json",
    )

    assert index["ok"] is False
    assert any("duplicate specialist unit" in issue for issue in index["issues"])


def test_exchange_cli_subcommands_return_nonzero_on_validation_failures(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": SPECIALIST_MODEL_UNIT_VERSION}), encoding="utf-8")

    code = exchange_main(["validate", "--manifest", str(bad)])

    assert code == 1


def test_package_check_computes_checksums_and_attaches_evidence(tmp_path: Path) -> None:
    paths = _write_evidence(tmp_path)
    artifact = tmp_path / "artifacts" / "model.safetensors"
    artifact.parent.mkdir()
    artifact.write_text("weights", encoding="utf-8")
    manifest = _build_unit(tmp_path, paths, artifact_ref="model.safetensors")
    evidence = tmp_path / "package.json"

    payload = run_package_check(
        manifest_path=manifest,
        out_path=evidence,
        artifacts_root=artifact.parent,
        package_types=["safetensors"],
        release_ready=True,
        update_manifest=True,
    )

    updated = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["version"] == SPECIALIST_PACKAGE_VERSION
    assert payload["ok"] is True
    assert payload["artifacts"][0]["sha256"] == sha256_file(artifact)
    assert updated["package_evidence"]["path"] == str(evidence)
    assert updated["package_evidence"]["ok"] is True


def test_package_check_fails_missing_local_release_artifact(tmp_path: Path) -> None:
    paths = _write_evidence(tmp_path)
    manifest = _build_unit(
        tmp_path,
        paths,
        artifact_ref="missing.safetensors",
        release_ready=False,
    )

    payload = run_package_check(
        manifest_path=manifest,
        out_path=tmp_path / "package.json",
        artifacts_root=tmp_path / "artifacts",
        package_types=["safetensors"],
        release_ready=True,
    )

    assert payload["ok"] is False
    assert any("local package ref missing" in issue for issue in payload["issues"])


def test_package_check_validates_local_ollama_modelfile(tmp_path: Path) -> None:
    paths = _write_evidence(tmp_path)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    modelfile = artifact_root / "Modelfile"
    modelfile.write_text("PARAMETER temperature 0\n", encoding="utf-8")
    manifest = _build_unit(
        tmp_path,
        paths,
        artifact_ref="hf://signalforgeai/example/model.safetensors",
        ollama_modelfile="Modelfile",
    )

    payload = run_package_check(
        manifest_path=manifest,
        out_path=tmp_path / "package.json",
        artifacts_root=artifact_root,
        package_types=["ollama"],
        release_ready=True,
    )

    assert payload["ok"] is False
    assert any("missing FROM line" in issue for issue in payload["issues"])


def test_smoke_run_writes_evidence_and_attaches_manifest_ref(tmp_path: Path) -> None:
    paths = _write_evidence(tmp_path)
    manifest = _build_unit(
        tmp_path,
        paths,
        artifact_ref="hf://signalforgeai/example/model.safetensors",
    )
    package_payload = run_package_check(
        manifest_path=manifest,
        out_path=tmp_path / "package.json",
        package_types=["safetensors"],
        release_ready=True,
        update_manifest=True,
    )

    payload = run_smoke_check(
        manifest_path=manifest,
        work_dir=tmp_path / "smoke",
        package_evidence_path=tmp_path / "package.json",
        update_manifest=True,
    )
    updated = json.loads(manifest.read_text(encoding="utf-8"))

    assert package_payload["ok"] is True
    assert payload["version"] == SPECIALIST_SMOKE_VERSION
    assert payload["ok"] is True
    assert payload["mode"] == "dummy"
    assert Path(payload["report_json"]).exists()
    assert updated["smoke_run_evidence"]["path"] == payload["report_json"]


def test_registry_index_includes_package_and_smoke_evidence(tmp_path: Path) -> None:
    paths = _write_evidence(tmp_path)
    manifest = _build_unit(
        tmp_path,
        paths,
        artifact_ref="hf://signalforgeai/example/model.safetensors",
    )
    run_package_check(
        manifest_path=manifest,
        out_path=tmp_path / "package.json",
        package_types=["safetensors"],
        release_ready=True,
        update_manifest=True,
    )
    run_smoke_check(
        manifest_path=manifest,
        work_dir=tmp_path / "smoke",
        update_manifest=True,
    )

    index = build_registry_index(
        registry_dir=tmp_path / "registry",
        out_path=tmp_path / "index.json",
        release_ready=True,
    )

    unit = index["units"][0]
    assert index["ok"] is True
    assert unit["package_evidence"]["version"] == SPECIALIST_PACKAGE_VERSION
    assert unit["smoke_run_evidence"]["version"] == SPECIALIST_SMOKE_VERSION


def test_registry_index_includes_training_run_evidence(tmp_path: Path) -> None:
    paths = _write_evidence(tmp_path)
    _build_unit(
        tmp_path,
        paths,
        artifact_ref="",
        training_run_path=paths["training_run"],
    )

    index = build_registry_index(
        registry_dir=tmp_path / "registry",
        out_path=tmp_path / "index.json",
        release_ready=True,
    )

    unit = index["units"][0]
    assert index["ok"] is True
    assert unit["training_run_evidence"]["version"] == TRAINING_RUN_VERSION
    assert unit["evidence"]["training_run_evidence"] == str(paths["training_run"].resolve())


def test_exchange_package_and_smoke_cli(tmp_path: Path) -> None:
    paths = _write_evidence(tmp_path)
    manifest = _build_unit(
        tmp_path,
        paths,
        artifact_ref="hf://signalforgeai/example/model.safetensors",
    )

    package_code = exchange_main(
        [
            "package-check",
            "--manifest",
            str(manifest),
            "--out",
            str(tmp_path / "package.json"),
            "--package-type",
            "safetensors",
            "--release-ready",
            "--update-manifest",
        ]
    )
    smoke_code = exchange_main(
        [
            "smoke-run",
            "--manifest",
            str(manifest),
            "--work-dir",
            str(tmp_path / "smoke"),
            "--update-manifest",
        ]
    )

    assert package_code == 0
    assert smoke_code == 0


def _build_unit(
    tmp_path: Path,
    paths: dict[str, Path],
    *,
    artifact_ref: str,
    training_run_path: Path | None = None,
    ollama_modelfile: str = "hf://signalforgeai/example/Modelfile",
    release_ready: bool = True,
) -> Path:
    out_dir = tmp_path / "registry"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "unit.json"
    build_specialist_unit(
        training_preflight_path=paths["training_preflight"],
        distillation_eval_path=paths["distillation_eval"],
        benchmark_matrix_path=paths["benchmark_matrix"],
        training_run_path=training_run_path,
        out_path=out,
        unit_id="pilot-specialist",
        name="Pilot Specialist",
        version="0.1.0",
        domain="pilot",
        model_family="dummy",
        model_size="0B",
        model_format="safetensors",
        model_license="Apache-2.0",
        dataset_license="CC-BY-4.0",
        usage_constraints=["not for production decisions without review"],
        failure_mode="known_gap",
        failure_description="Dummy specialist only proves exchange plumbing.",
        failure_mitigation="Replace dummy artifact refs before release.",
        failure_severity="low",
        safetensors_refs=[artifact_ref],
        ollama_modelfile=ollama_modelfile,
        ollama_tag="signalforgeai/pilot-specialist:0.1.0",
        artifacts_root=tmp_path / "artifacts",
        release_ready=release_ready,
    )
    return out


def _write_evidence(tmp_path: Path) -> dict[str, Path]:
    logs = tmp_path / "logs"
    datasets = tmp_path / "datasets"
    distill = tmp_path / "distill"
    benchmark = tmp_path / "benchmark"
    for path in (logs, datasets, distill, benchmark):
        path.mkdir(parents=True)
    dataset = datasets / "pilot.sft.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    training_out = tmp_path / "training" / "sft_lora"
    training_out.mkdir(parents=True)
    adapter_file = training_out / "adapter_model.safetensors"
    adapter_file.write_text("weights", encoding="utf-8")
    dpo_out = tmp_path / "training" / "dpo_lora"
    dpo_out.mkdir(parents=True)
    dpo_adapter_file = dpo_out / "adapter_model.safetensors"
    dpo_adapter_file.write_text("dpo-weights", encoding="utf-8")
    recipe = tmp_path / "distillation_recipe.json"
    recipe.write_text("{}", encoding="utf-8")
    training_preflight = tmp_path / "training_preflight.json"
    training_preflight.write_text(
        json.dumps(
            {
                "version": "training_preflight.v0",
                "ok": True,
                "base_model": "dummy/base",
                "logs_root": str(logs),
                "artifact_manifest": {
                    "version": "training_artifact.v0",
                    "base_model": "dummy/base",
                    "dataset_sources": {"sft": str(dataset)},
                    "dataset_hashes": {"sft": "a" * 64},
                    "split_counts": {"sft": {"train": 1}},
                    "datasets": [{"role": "sft", "path": str(dataset)}],
                    "logs_root": str(logs),
                    "outputs": {"sft_out": str(tmp_path / "training" / "sft")},
                },
            }
        ),
        encoding="utf-8",
    )
    training_run = tmp_path / "training_run.json"
    training_run.write_text(
        json.dumps(
            {
                "version": "training_run.v0",
                "ok": True,
                "status": "succeeded",
                "base_model": "dummy/base",
                "preflight_path": str(training_preflight),
                "preflight_ok": True,
                "datasets": [{"role": "sft", "path": str(dataset)}],
                "dataset_hashes": {"sft": "a" * 64},
                "split_counts": {"sft": {"train": 1}},
                "outputs": {"sft_out": str(training_out), "dpo_out": None, "sft_dir": None},
                "artifact_refs": {
                    "adapters": [str(training_out)],
                    "safetensors": [],
                    "gguf": [],
                    "ollama": {"modelfile": "", "tag": ""},
                },
                "output_artifacts": [
                    {
                        "role": "sft_out",
                        "path": str(adapter_file),
                        "sha256": sha256_file(adapter_file),
                        "bytes": adapter_file.stat().st_size,
                    }
                ],
                "file_checksums": {str(adapter_file): sha256_file(adapter_file)},
                "issues": [],
            }
        ),
        encoding="utf-8",
    )
    training_run_dpo = tmp_path / "training_run_dpo.json"
    training_run_dpo.write_text(
        json.dumps(
            {
                "version": "training_run.v0",
                "ok": True,
                "status": "succeeded",
                "base_model": "dummy/base",
                "training_stage": "dpo",
                "preflight_path": str(training_preflight),
                "preflight_ok": True,
                "dpo_parent_run": {
                    "path": str(training_run),
                    "ok": True,
                    "adapter_refs": [str(training_out)],
                },
                "datasets": [{"role": "dpo", "path": str(dataset)}],
                "dataset_hashes": {"dpo": "b" * 64},
                "split_counts": {"dpo": {"train": 1}},
                "outputs": {"sft_out": None, "dpo_out": str(dpo_out), "sft_dir": str(training_out)},
                "artifact_refs": {
                    "adapters": [str(dpo_out)],
                    "safetensors": [],
                    "gguf": [],
                    "ollama": {"modelfile": "", "tag": ""},
                },
                "artifact_refs_by_role": {
                    "sft_out": [],
                    "dpo_out": [str(dpo_out)],
                },
                "final_adapter_refs": [str(dpo_out)],
                "output_artifacts": [
                    {
                        "role": "dpo_out",
                        "path": str(dpo_adapter_file),
                        "sha256": sha256_file(dpo_adapter_file),
                        "bytes": dpo_adapter_file.stat().st_size,
                    }
                ],
                "file_checksums": {str(dpo_adapter_file): sha256_file(dpo_adapter_file)},
                "issues": [],
            }
        ),
        encoding="utf-8",
    )
    distillation_eval = distill / "distillation_eval.json"
    distillation_eval.with_suffix(".md").write_text("# Distill\n", encoding="utf-8")
    distillation_eval.write_text(
        json.dumps(
            {
                "version": "distillation_eval.v0",
                "ok": True,
                "recipe": {
                    "suite": "decision_v0",
                    "candidate_model_id": "dummy_good",
                    "source_path": str(recipe),
                },
                "candidate": {
                    "suite_name": "benchmark_v0_decision",
                    "model_id": "dummy_good",
                    "mean_score": 0.95,
                    "pass_rate": 1.0,
                },
                "artifact_refs": {
                    "baseline_logs": str(logs / "baseline"),
                    "candidate_logs": str(logs / "candidate"),
                },
            }
        ),
        encoding="utf-8",
    )
    (logs / "baseline").mkdir()
    (logs / "candidate").mkdir()
    benchmark_matrix = benchmark / "benchmark_matrix.json"
    (benchmark / "logs").mkdir()
    benchmark_matrix.write_text(
        json.dumps(
            {
                "version": "benchmark_matrix.v0",
                "ok": True,
                "artifact_refs": {"logs_dir": str(benchmark / "logs")},
                "scorecard": [
                    {
                        "suite": "benchmark_v0_decision",
                        "model_id": "dummy_good",
                        "run_logs_dir": str(benchmark / "logs" / "run-1"),
                        "metrics": {
                            "latency_ms_p50": 42,
                            "latency_ms_p95": 90,
                            "cost_per_success_usd": 0.0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (benchmark / "logs" / "run-1").mkdir()
    return {
        "training_preflight": training_preflight,
        "training_run": training_run,
        "training_run_dpo": training_run_dpo,
        "dpo_out": dpo_out,
        "distillation_eval": distillation_eval,
        "benchmark_matrix": benchmark_matrix,
    }
