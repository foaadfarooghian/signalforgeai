from __future__ import annotations

import json
from pathlib import Path

from tensorfoundry.exchange.cli import main as exchange_main
from tensorfoundry.exchange.registry import build_registry_index
from tensorfoundry.exchange.unit import build_specialist_unit
from tensorfoundry.exchange.validation import (
    SPECIALIST_MODEL_UNIT_VERSION,
    sha256_file,
    validate_manifest,
)


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
        artifact_ref="hf://tensorfoundry/example/model.safetensors",
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
        artifact_ref="hf://tensorfoundry/example/model.safetensors",
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["lineage"]["dataset_hashes"]["sft"] == "a" * 64
    assert payload["eval_pack"]["score"] == 0.95
    assert payload["eval_pack"]["latency_ms_p50"] == 42
    assert payload["eval_pack"]["distillation_eval_path"] == str(paths["distillation_eval"].resolve())
    assert payload["eval_pack"]["benchmark_matrix_path"] == str(paths["benchmark_matrix"].resolve())


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


def _build_unit(tmp_path: Path, paths: dict[str, Path], *, artifact_ref: str) -> Path:
    out_dir = tmp_path / "registry"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "unit.json"
    build_specialist_unit(
        training_preflight_path=paths["training_preflight"],
        distillation_eval_path=paths["distillation_eval"],
        benchmark_matrix_path=paths["benchmark_matrix"],
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
        ollama_modelfile="hf://tensorfoundry/example/Modelfile",
        ollama_tag="tensorfoundry/pilot-specialist:0.1.0",
        artifacts_root=tmp_path / "artifacts",
        release_ready=True,
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
        "distillation_eval": distillation_eval,
        "benchmark_matrix": benchmark_matrix,
    }
