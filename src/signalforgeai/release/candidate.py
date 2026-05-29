"""Offline-first release-candidate evidence gate for SignalForge AI."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from signalforgeai.distillation.recipe import DISTILLATION_RECIPE_VERSION
from signalforgeai.distillation.check import run_distillation_check
from signalforgeai.evaluation.matrix import (
    BENCHMARK_MATRIX_VERSION,
    BenchmarkMatrixConfig,
    MetricWeights,
    resolve_suite_path,
    run_benchmark_matrix,
)
from signalforgeai.exchange.package import run_package_check
from signalforgeai.exchange.registry import build_registry_index
from signalforgeai.exchange.smoke import run_smoke_check
from signalforgeai.exchange.unit import build_specialist_unit
from signalforgeai.learning.learn import (
    DEFAULT_REAL_SFT_SMOKE_MODEL,
    PLACEHOLDER_TRAINING_BASE_MODELS,
    TrainingRunConfig,
    run_training as run_training_job,
)
from signalforgeai.models.registry import check_provider_for_model
from signalforgeai.pilot_check import DEFAULT_SUITE, run_pilot_check
from signalforgeai.training.readiness import (
    build_training_run,
    load_training_preflight_report,
    load_training_run_report,
    summarize_dpo_parent_run,
    utc_now,
    write_training_run,
)


RELEASE_CANDIDATE_VERSION = "release_candidate.v0"

_DEFAULT_HOSTED_MODEL_ID = "openai:gpt-5-mini"
_DEFAULT_LOCAL_MODEL_ID = "ollama:ministral-3:8b"
_GENERATED_PATHS = (
    "pilot",
    "training",
    "distillation",
    "benchmark",
    "exchange",
    "release_candidate.json",
    "release_candidate.md",
    "distillation_recipe.json",
    "benchmark_matrix_config.json",
)


def run_release_candidate_check(
    *,
    work_dir: str | Path,
    mode: str = "dummy",
    unit_id: str = "pilot-specialist",
    name: str = "Pilot Specialist",
    version: str = "0.7.0",
    domain: str = "pilot",
    baseline_model_id: str = "dummy_good",
    candidate_model_id: str | None = None,
    training_base_model: str = "dummy/base",
    suites: Optional[Sequence[str]] = None,
    comparison_model_ids: Optional[Sequence[str]] = None,
    matrix_config: str | Path | None = None,
    sft_run: str | Path | None = None,
    dpo_run: str | Path | None = None,
    run_training: bool = False,
    run_dpo: bool = False,
    training_max_steps: int = 1,
    final_training_stage: str = "auto",
    require_real_training_evidence: bool = False,
    require_provider: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Run the full offline release-candidate evidence chain."""
    if run_training and (sft_run is not None or dpo_run is not None):
        raise ValueError("--run-training cannot be combined with --sft-run or --dpo-run")
    if run_dpo and not run_training:
        raise ValueError("--run-dpo requires --run-training")
    if run_training and final_training_stage == "dpo" and not run_dpo:
        raise ValueError("--final-training-stage dpo requires --run-dpo")
    if run_training and run_dpo and final_training_stage == "sft":
        raise ValueError("--final-training-stage sft cannot be combined with --run-dpo")
    if run_training and _is_placeholder_training_base_model(training_base_model):
        raise ValueError(_invalid_training_base_model_issue())
    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _reset_generated_outputs(root)

    requested_suites = list(suites or [DEFAULT_SUITE])
    resolved_suites = [_resolve_suite(value, root) for value in requested_suites]
    required_providers = set(require_provider or [])
    benchmark_required_providers = set(required_providers)
    explicit_candidate_model_id = candidate_model_id is not None
    effective_candidate_model_id = candidate_model_id or "dummy_good"
    gates: List[Dict[str, Any]] = []
    issues: List[str] = []
    warnings: List[str] = []
    paths = _paths(root)
    command_config = {
        "mode": mode,
        "suites": requested_suites,
        "resolved_suites": [str(path) for path in resolved_suites],
        "baseline_model_id": baseline_model_id,
        "candidate_model_id": effective_candidate_model_id,
        "requested_candidate_model_id": candidate_model_id,
        "comparison_model_ids": list(comparison_model_ids or []),
        "matrix_config": str(matrix_config) if matrix_config is not None else None,
        "sft_run": str(sft_run) if sft_run is not None else None,
        "dpo_run": str(dpo_run) if dpo_run is not None else None,
        "run_training": bool(run_training),
        "run_dpo": bool(run_dpo),
        "training_max_steps": int(training_max_steps),
        "final_training_stage": final_training_stage,
        "require_real_training_evidence": bool(require_real_training_evidence),
        "require_provider": sorted(required_providers),
        "training_base_model": training_base_model,
    }
    artifacts: Dict[str, Any] = {}
    provider_status: Dict[str, Any] = {}
    checksums: Dict[str, str] = {}
    evidence_paths: Dict[str, str] = {}

    pilot_payload: Dict[str, Any] = {}
    preflight_payload: Dict[str, Any] = {}
    training_payload: Dict[str, Any] = {}
    distill_payload: Dict[str, Any] = {}
    matrix_payload: Dict[str, Any] = {}
    unit_manifest: Dict[str, Any] = {}
    package_payload: Dict[str, Any] = {}
    smoke_payload: Dict[str, Any] = {}
    index_payload: Dict[str, Any] = {}

    try:
        pilot_payload = run_pilot_check(
            work_dir=paths["pilot"],
            suites=[str(path) for path in resolved_suites],
            require_provider=required_providers,
            hosted_model_id=_DEFAULT_HOSTED_MODEL_ID,
            local_model_id=_DEFAULT_LOCAL_MODEL_ID,
            training_preflight=True,
            training_base_model=training_base_model,
        )
        _add_gate(gates, "pilot_readiness", pilot_payload.get("ok"), pilot_payload.get("report_json"), pilot_payload.get("issues"))
        provider_status["pilot"] = pilot_payload.get("providers", [])
        evidence_paths["pilot_readiness"] = str(pilot_payload.get("report_json") or "")
        if not pilot_payload.get("ok"):
            issues.extend(f"pilot_readiness: {issue}" for issue in _list(pilot_payload.get("issues")))
    except Exception as exc:
        _add_gate(gates, "pilot_readiness", False, None, [str(exc)])
        issues.append(f"pilot_readiness: {exc}")

    preflight_path = paths["pilot"] / "training_preflight.json"
    try:
        preflight_payload = load_training_preflight_report(preflight_path)
        evidence_paths["training_preflight"] = str(preflight_path)
        training_payload = write_release_training_evidence(
            preflight=preflight_payload,
            preflight_path=preflight_path,
            training_dir=paths["training"],
            base_model=training_base_model,
            sft_run=sft_run,
            dpo_run=dpo_run,
            run_training=run_training,
            run_dpo=run_dpo,
            training_max_steps=training_max_steps,
            final_training_stage=final_training_stage,
            require_real_training_evidence=bool(require_real_training_evidence or run_training),
        )
        derived_candidate = _candidate_model_id_from_training(
            training_payload,
            base_model=training_base_model,
        )
        if derived_candidate:
            training_payload["derived_candidate_model_id"] = derived_candidate
            command_config["derived_candidate_model_id"] = derived_candidate
            if run_training and not explicit_candidate_model_id:
                effective_candidate_model_id = derived_candidate
                command_config["candidate_model_id"] = effective_candidate_model_id
        if run_training:
            candidate_check = check_provider_for_model(
                effective_candidate_model_id,
                required=True,
            )
            provider_status["candidate"] = _provider_check_payload(candidate_check)
            benchmark_required_providers.add(candidate_check.provider)
            if not candidate_check.ok:
                issues.append(
                    "candidate_provider: "
                    + (candidate_check.reason or f"provider unavailable for {effective_candidate_model_id}")
                )
        _add_gate(gates, "training_evidence", training_payload.get("ok"), training_payload.get("final_run_path"), training_payload.get("issues"))
        if not training_payload.get("ok"):
            issues.extend(f"training_evidence: {issue}" for issue in _list(training_payload.get("issues")))
        evidence_paths["training_run"] = str(training_payload.get("final_run_path") or "")
    except Exception as exc:
        _add_gate(gates, "training_evidence", False, None, [str(exc)])
        issues.append(f"training_evidence: {exc}")

    recipe_path = paths["recipe"]
    try:
        _write_distillation_recipe(
            recipe_path,
            unit_id=unit_id,
            domain=domain,
            suite_path=resolved_suites[0],
            baseline_model_id=baseline_model_id,
            candidate_model_id=effective_candidate_model_id,
            training_preflight_path=preflight_path,
        )
        distill_payload = run_distillation_check(
            recipe_path=recipe_path,
            work_dir=paths["distillation"],
        )
        _add_gate(gates, "distillation_eval", distill_payload.get("ok"), distill_payload.get("report_json"), distill_payload.get("issues"))
        evidence_paths["distillation_eval"] = str(distill_payload.get("report_json") or "")
        if not distill_payload.get("ok"):
            issues.extend(f"distillation_eval: {issue}" for issue in _list(distill_payload.get("issues")))
    except Exception as exc:
        _add_gate(gates, "distillation_eval", False, None, [str(exc)])
        issues.append(f"distillation_eval: {exc}")

    try:
        matrix_config_path = Path(matrix_config).resolve() if matrix_config is not None else _write_matrix_config(
            paths["matrix_config"],
            unit_id=unit_id,
            suites=resolved_suites,
            model_ids=_matrix_model_ids(
                baseline_model_id=baseline_model_id,
                candidate_model_id=effective_candidate_model_id,
                comparison_model_ids=comparison_model_ids or [],
            ),
            require_providers=benchmark_required_providers,
        )
        matrix_payload = run_benchmark_matrix(
            config_path=matrix_config_path,
            work_dir=paths["benchmark"],
            suites=[str(path) for path in resolved_suites] if suites else None,
            model_ids=[baseline_model_id, effective_candidate_model_id, *(comparison_model_ids or [])]
            if matrix_config is not None
            else None,
            require_providers=list(benchmark_required_providers) or None,
        )
        matrix_issues = list(_list(matrix_payload.get("issues")))
        if not _matrix_contains_candidate(matrix_payload, effective_candidate_model_id):
            matrix_issues.append(f"benchmark matrix missing candidate model row: {effective_candidate_model_id}")
        matrix_ok = bool(matrix_payload.get("ok") and not matrix_issues)
        _add_gate(gates, "benchmark_matrix", matrix_ok, matrix_payload.get("artifact_refs", {}).get("json"), matrix_issues)
        provider_status["benchmark_matrix"] = _matrix_provider_status(matrix_payload)
        evidence_paths["benchmark_matrix"] = str(matrix_payload.get("artifact_refs", {}).get("json") or "")
        if matrix_issues:
            issues.extend(f"benchmark_matrix: {issue}" for issue in matrix_issues)
    except Exception as exc:
        _add_gate(gates, "benchmark_matrix", False, None, [str(exc)])
        issues.append(f"benchmark_matrix: {exc}")

    try:
        final_training_run = str(training_payload.get("final_run_path") or "")
        real_training = bool(training_payload.get("real_training_evidence"))
        unit_manifest = build_specialist_unit(
            training_preflight_path=preflight_path,
            training_run_path=final_training_run,
            distillation_eval_path=paths["distillation"] / "distillation_eval.json",
            benchmark_matrix_path=paths["benchmark"] / "benchmark_matrix.json",
            out_path=paths["unit"],
            unit_id=unit_id,
            name=name,
            version=version,
            domain=domain,
            summary=(
                "Release-candidate evidence bundle with external training evidence."
                if real_training
                else "Offline release-candidate evidence bundle."
            ),
            model_family="dummy",
            model_size="0B",
            model_format="safetensors",
            base_model=training_base_model,
            model_license="Apache-2.0",
            dataset_license="CC-BY-4.0",
            usage_constraints=["not for production decisions without review"],
            failure_mode="release_candidate" if real_training else "offline_mock",
            failure_description=(
                "External training evidence is linked, but the candidate still requires domain review."
                if real_training
                else "Offline mock artifacts prove release evidence plumbing only."
            ),
            failure_mitigation=(
                "Review linked eval, benchmark, package, and smoke evidence before production use."
                if real_training
                else "Replace mock artifacts with real trained model refs before production release."
            ),
            release_ready=True,
        )
        _add_gate(gates, "specialist_unit", True, paths["unit"], [])
        evidence_paths["specialist_unit"] = str(paths["unit"])
    except Exception as exc:
        _add_gate(gates, "specialist_unit", False, paths["unit"], [str(exc)])
        issues.append(f"specialist_unit: {exc}")

    try:
        package_payload = run_package_check(
            manifest_path=paths["unit"],
            out_path=paths["package"],
            package_types=["auto"],
            release_ready=True,
            update_manifest=True,
        )
        _add_gate(gates, "package_check", package_payload.get("ok"), paths["package"], package_payload.get("issues"))
        warnings.extend(f"package_check: {warning}" for warning in _list(package_payload.get("warnings")))
        evidence_paths["package_evidence"] = str(paths["package"])
        if not package_payload.get("ok"):
            issues.extend(f"package_check: {issue}" for issue in _list(package_payload.get("issues")))
    except Exception as exc:
        _add_gate(gates, "package_check", False, paths["package"], [str(exc)])
        issues.append(f"package_check: {exc}")

    try:
        smoke_payload = run_smoke_check(
            manifest_path=paths["unit"],
            work_dir=paths["smoke"],
            mode="dummy",
            model_id=effective_candidate_model_id
            if effective_candidate_model_id.startswith("dummy")
            else "dummy_good",
            update_manifest=True,
        )
        _add_gate(gates, "smoke_run", smoke_payload.get("ok"), smoke_payload.get("report_json"), smoke_payload.get("issues"))
        provider_status["smoke"] = smoke_payload.get("provider", {})
        warnings.extend(f"smoke_run: {warning}" for warning in _list(smoke_payload.get("warnings")))
        evidence_paths["smoke_run"] = str(smoke_payload.get("report_json") or "")
        if not smoke_payload.get("ok"):
            issues.extend(f"smoke_run: {issue}" for issue in _list(smoke_payload.get("issues")))
    except Exception as exc:
        _add_gate(gates, "smoke_run", False, paths["smoke"] / "specialist_smoke.json", [str(exc)])
        issues.append(f"smoke_run: {exc}")

    try:
        index_payload = build_registry_index(
            registry_dir=paths["exchange"],
            out_path=paths["index"],
            release_ready=True,
        )
        _add_gate(gates, "registry_index", index_payload.get("ok"), paths["index"], index_payload.get("issues"))
        evidence_paths["registry_index"] = str(paths["index"])
        if not index_payload.get("ok"):
            issues.extend(f"registry_index: {issue}" for issue in _list(index_payload.get("issues")))
    except Exception as exc:
        _add_gate(gates, "registry_index", False, paths["index"], [str(exc)])
        issues.append(f"registry_index: {exc}")

    if paths["unit"].exists():
        unit_manifest = _load_json(paths["unit"])
        artifacts = _dict(unit_manifest.get("artifacts"))
        checksums = {str(k): str(v) for k, v in _dict(artifacts.get("checksums")).items()}

    ok = all(bool(gate.get("ok")) for gate in gates) and not issues
    payload: Dict[str, Any] = {
        "version": RELEASE_CANDIDATE_VERSION,
        "ok": ok,
        "created_at": _utc_now(),
        "identity": {
            "id": unit_id,
            "name": name,
            "version": version,
            "domain": domain,
        },
        "mode": mode,
        "work_dir": str(root),
        "command_config": command_config,
        "gate_results": gates,
        "evidence_paths": evidence_paths,
        "artifact_refs": artifacts,
        "checksums": checksums,
        "provider_status": provider_status,
        "training_evidence": _training_summary(training_payload),
        "issues": issues,
        "warnings": warnings,
    }
    _write_release_reports(payload, json_path=paths["release_json"], markdown_path=paths["release_md"])
    payload["report_json"] = str(paths["release_json"])
    payload["report_md"] = str(paths["release_md"])
    return payload


def write_release_training_evidence(
    *,
    preflight: Mapping[str, Any],
    preflight_path: str | Path,
    training_dir: str | Path,
    base_model: str,
    sft_run: str | Path | None = None,
    dpo_run: str | Path | None = None,
    run_training: bool = False,
    run_dpo: bool = False,
    training_max_steps: int = 1,
    final_training_stage: str = "auto",
    require_real_training_evidence: bool = False,
) -> Dict[str, Any]:
    """Write or load release training evidence and return the selected final run."""
    if final_training_stage not in {"auto", "sft", "dpo"}:
        raise ValueError(f"unsupported final training stage: {final_training_stage}")
    if run_training and (sft_run is not None or dpo_run is not None):
        raise ValueError("--run-training cannot be combined with --sft-run or --dpo-run")
    if run_dpo and not run_training:
        raise ValueError("--run-dpo requires --run-training")
    if run_training and final_training_stage == "dpo" and not run_dpo:
        raise ValueError("--final-training-stage dpo requires --run-dpo")
    if run_training and run_dpo and final_training_stage == "sft":
        raise ValueError("--final-training-stage sft cannot be combined with --run-dpo")
    if dpo_run is not None and final_training_stage == "sft":
        raise ValueError("--final-training-stage sft cannot be combined with --dpo-run")
    out_dir = Path(training_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    issues: List[str] = []
    runs: Dict[str, str] = {}
    mode = "external" if sft_run or dpo_run else "mock"

    if run_training:
        if _is_placeholder_training_base_model(base_model):
            return {
                "ok": False,
                "mode": "run",
                "final_stage": "dpo" if run_dpo else "sft",
                "final_run_path": str(
                    out_dir / ("dpo_training_run.json" if run_dpo else "sft_training_run.json")
                ),
                "sft_run_path": str(out_dir / "sft_training_run.json"),
                "dpo_run_path": str(out_dir / "dpo_training_run.json") if run_dpo else None,
                "runs": {},
                "real_training_evidence": False,
                "parent_real_training_evidence": False,
                "require_real_training_evidence": bool(require_real_training_evidence),
                "adapter_smoke": {},
                "issues": [_invalid_training_base_model_issue()],
            }
        return _write_real_training_evidence(
            preflight=preflight,
            preflight_path=preflight_path,
            out_dir=out_dir,
            base_model=base_model,
            run_dpo=run_dpo,
            training_max_steps=training_max_steps,
            require_real_training_evidence=require_real_training_evidence,
        )

    if dpo_run is not None:
        dpo_path = Path(dpo_run).resolve()
        payload = load_training_run_report(dpo_path)
        try:
            _require_adapter_evidence(payload, str(dpo_path))
        except ValueError as exc:
            issues.append(str(exc))
        parent_payload, parent_path, parent_issues = _load_dpo_parent_training_run(
            payload,
            explicit_sft_run=sft_run,
        )
        issues.extend(parent_issues)
        parent_real = _is_real_training_run(parent_payload) if parent_payload else None
        real = _is_real_training_run(payload)
        if require_real_training_evidence:
            issues.extend(
                _real_dpo_evidence_issues(
                    dpo_payload=payload,
                    dpo_path=dpo_path,
                    parent_payload=parent_payload,
                    parent_path=parent_path,
                )
            )
        runs["dpo"] = str(dpo_path)
        if parent_path:
            runs["sft"] = str(parent_path)
        return {
            "ok": not issues,
            "mode": "external_dpo",
            "final_stage": str(payload.get("training_stage") or "unknown"),
            "final_run_path": str(dpo_path),
            "sft_run_path": str(parent_path) if parent_path else None,
            "dpo_run_path": str(dpo_path),
            "runs": runs,
            "real_training_evidence": real,
            "parent_real_training_evidence": parent_real,
            "require_real_training_evidence": bool(require_real_training_evidence),
            "adapter_smoke": _dict(payload.get("adapter_smoke")),
            "issues": issues,
        }

    if sft_run is not None:
        sft_payload = load_training_run_report(sft_run)
        _require_adapter_evidence(sft_payload, str(sft_run))
        parent = summarize_dpo_parent_run(sft_run)
        sft_path = Path(sft_run).resolve()
        runs["sft"] = str(sft_path)
        parent_real = _is_real_training_run(sft_payload)
        if final_training_stage == "sft":
            if require_real_training_evidence and not parent_real:
                issues.append(_real_training_required_issue())
            return {
                "ok": not issues,
                "mode": "external_sft",
                "final_stage": str(sft_payload.get("training_stage") or "sft"),
                "final_run_path": str(sft_path),
                "sft_run_path": str(sft_path),
                "dpo_run_path": None,
                "runs": runs,
                "real_training_evidence": parent_real,
                "parent_real_training_evidence": parent_real,
                "require_real_training_evidence": bool(require_real_training_evidence),
                "adapter_smoke": _dict(sft_payload.get("adapter_smoke")),
                "issues": issues,
            }
    else:
        sft_path = out_dir / "sft_training_run.json"
        parent_payload = _write_mock_sft_run(
            preflight=preflight,
            preflight_path=preflight_path,
            out_path=sft_path,
            output_dir=out_dir / "sft_lora",
            base_model=base_model,
        )
        parent = summarize_dpo_parent_run(sft_path)
        runs["sft"] = str(sft_path)
        parent_real = False
        if not parent_payload.get("ok"):
            issues.extend(str(issue) for issue in _list(parent_payload.get("issues")))
        if final_training_stage == "sft":
            if require_real_training_evidence:
                issues.append(_real_training_required_issue())
            return {
                "ok": not issues,
                "mode": mode,
                "final_stage": "sft",
                "final_run_path": str(sft_path),
                "sft_run_path": str(sft_path),
                "dpo_run_path": None,
                "runs": runs,
                "real_training_evidence": False,
                "parent_real_training_evidence": False,
                "require_real_training_evidence": bool(require_real_training_evidence),
                "adapter_smoke": _dict(parent_payload.get("adapter_smoke")),
                "issues": issues,
            }

    dpo_path = out_dir / "dpo_training_run.json"
    dpo_payload = _write_mock_dpo_run(
        preflight=preflight,
        preflight_path=preflight_path,
        out_path=dpo_path,
        output_dir=out_dir / "dpo_lora",
        parent=parent,
        base_model=base_model,
        sft_run_path=sft_path,
    )
    runs["dpo"] = str(dpo_path)
    if not dpo_payload.get("ok"):
        issues.extend(str(issue) for issue in _list(dpo_payload.get("issues")))
    if require_real_training_evidence:
        issues.append(_real_training_required_issue())

    return {
        "ok": not issues,
        "mode": mode,
        "final_stage": "dpo",
        "final_run_path": str(dpo_path),
        "sft_run_path": runs.get("sft"),
        "dpo_run_path": str(dpo_path),
        "runs": runs,
        "real_training_evidence": False,
        "parent_real_training_evidence": parent_real,
        "require_real_training_evidence": bool(require_real_training_evidence),
        "adapter_smoke": _dict(dpo_payload.get("adapter_smoke")),
        "issues": issues,
    }


def _write_real_training_evidence(
    *,
    preflight: Mapping[str, Any],
    preflight_path: str | Path,
    out_dir: Path,
    base_model: str,
    run_dpo: bool,
    training_max_steps: int,
    require_real_training_evidence: bool,
) -> Dict[str, Any]:
    """Run real opt-in training and return release training evidence."""
    issues: List[str] = []
    runs: Dict[str, str] = {}
    logs_root = _preflight_logs_root(preflight)
    sft_dataset = _preflight_dataset_path(preflight, "sft")
    dpo_dataset = _preflight_dataset_path(preflight, "dpo")
    sft_report = out_dir / "sft_training_run.json"
    dpo_report = out_dir / "dpo_training_run.json"
    sft_out = out_dir / "sft_lora"
    dpo_out = out_dir / "dpo_lora"
    sft_payload: Dict[str, Any] = {}
    dpo_payload: Dict[str, Any] = {}

    if not sft_dataset:
        issues.append("training preflight has no SFT dataset source")
    else:
        sft_code = run_training_job(
            TrainingRunConfig(
                base_model=base_model,
                sft=True,
                sft_data=sft_dataset,
                sft_out=str(sft_out),
                dry_run=False,
                smoke=True,
                max_steps=int(training_max_steps),
                quality_gate=True,
                logs_root=logs_root,
                report_out=str(out_dir / "sft_training_preflight.json"),
                run_report_out=str(sft_report),
            )
        )
        runs["sft"] = str(sft_report)
        if sft_code != 0:
            issues.append(f"SFT training exited nonzero: {sft_code}")
        try:
            sft_payload = load_training_run_report(sft_report)
            _require_adapter_evidence(sft_payload, str(sft_report))
        except Exception as exc:
            issues.append(str(exc))

    parent_real = _is_real_training_run(sft_payload) if sft_payload else False
    if require_real_training_evidence and sft_payload and not parent_real:
        issues.append(_real_training_required_issue())

    if not run_dpo:
        return {
            "ok": not issues,
            "mode": "run",
            "final_stage": "sft",
            "final_run_path": str(sft_report),
            "sft_run_path": str(sft_report),
            "dpo_run_path": None,
            "runs": runs,
            "real_training_evidence": bool(sft_payload and parent_real),
            "parent_real_training_evidence": parent_real,
            "require_real_training_evidence": bool(require_real_training_evidence),
            "adapter_smoke": _dict(sft_payload.get("adapter_smoke")),
            "issues": issues,
        }

    if issues:
        return {
            "ok": False,
            "mode": "run",
            "final_stage": "dpo",
            "final_run_path": str(dpo_report),
            "sft_run_path": str(sft_report),
            "dpo_run_path": str(dpo_report),
            "runs": runs,
            "real_training_evidence": False,
            "parent_real_training_evidence": parent_real,
            "require_real_training_evidence": bool(require_real_training_evidence),
            "adapter_smoke": _dict(sft_payload.get("adapter_smoke")),
            "issues": issues,
        }

    if not dpo_dataset:
        issues.append("training preflight has no DPO dataset source")
    else:
        dpo_code = run_training_job(
            TrainingRunConfig(
                base_model=base_model,
                dpo=True,
                dpo_data=dpo_dataset,
                dpo_out=str(dpo_out),
                sft_run=str(sft_report),
                dry_run=False,
                smoke=True,
                max_steps=int(training_max_steps),
                quality_gate=True,
                logs_root=logs_root,
                report_out=str(out_dir / "dpo_training_preflight.json"),
                run_report_out=str(dpo_report),
            )
        )
        runs["dpo"] = str(dpo_report)
        if dpo_code != 0:
            issues.append(f"DPO training exited nonzero: {dpo_code}")
        try:
            dpo_payload = load_training_run_report(dpo_report)
            _require_adapter_evidence(dpo_payload, str(dpo_report))
        except Exception as exc:
            issues.append(str(exc))

    real = _is_real_training_run(dpo_payload) if dpo_payload else False
    if require_real_training_evidence and dpo_payload and not real:
        issues.append(_real_training_required_issue())

    return {
        "ok": not issues,
        "mode": "run",
        "final_stage": "dpo",
        "final_run_path": str(dpo_report),
        "sft_run_path": str(sft_report),
        "dpo_run_path": str(dpo_report),
        "runs": runs,
        "real_training_evidence": bool(dpo_payload and real),
        "parent_real_training_evidence": parent_real,
        "require_real_training_evidence": bool(require_real_training_evidence),
        "adapter_smoke": _dict(dpo_payload.get("adapter_smoke")),
        "issues": issues,
    }


def _write_mock_sft_run(
    *,
    preflight: Mapping[str, Any],
    preflight_path: str | Path,
    out_path: Path,
    output_dir: Path,
    base_model: str,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "adapter_model.safetensors").write_text(
        "signalforgeai mock sft adapter\n",
        encoding="utf-8",
    )
    started_at = utc_now()
    payload = build_training_run(
        base_model=base_model,
        preflight=preflight,
        preflight_path=preflight_path,
        outputs={"sft_out": str(output_dir), "dpo_out": None, "sft_dir": None},
        optional_dependencies=_optional_dependencies(preflight),
        config={
            "source": "signalforgeai-release-candidate-check",
            "training_evidence_mode": "mock",
            "sft": True,
            "dpo": False,
            "smoke": True,
            "max_steps": 1,
            "quality_gate": True,
        },
        status="succeeded",
        started_at=started_at,
        duration_seconds=0.0,
    )
    write_training_run(payload, out_path)
    return payload


def _write_mock_dpo_run(
    *,
    preflight: Mapping[str, Any],
    preflight_path: str | Path,
    out_path: Path,
    output_dir: Path,
    parent: Mapping[str, Any],
    base_model: str,
    sft_run_path: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "adapter_model.safetensors").write_text(
        "signalforgeai mock dpo adapter\n",
        encoding="utf-8",
    )
    started_at = utc_now()
    payload = build_training_run(
        base_model=base_model,
        preflight=preflight,
        preflight_path=preflight_path,
        outputs={
            "sft_out": None,
            "dpo_out": str(output_dir),
            "sft_dir": _first_adapter_ref(parent),
        },
        optional_dependencies=_optional_dependencies(preflight),
        config={
            "source": "signalforgeai-release-candidate-check",
            "training_evidence_mode": "mock",
            "sft": False,
            "dpo": True,
            "sft_run": str(sft_run_path),
            "smoke": True,
            "max_steps": 1,
            "quality_gate": True,
        },
        dpo_parent_run=parent,
        status="succeeded",
        started_at=started_at,
        duration_seconds=0.0,
    )
    write_training_run(payload, out_path)
    return payload


def _reset_generated_outputs(work_dir: Path) -> None:
    for value in _GENERATED_PATHS:
        path = work_dir / value
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _paths(root: Path) -> Dict[str, Path]:
    exchange = root / "exchange"
    return {
        "pilot": root / "pilot",
        "training": root / "training",
        "distillation": root / "distillation",
        "benchmark": root / "benchmark",
        "exchange": exchange,
        "unit": exchange / "unit.json",
        "package": exchange / "specialist_package.json",
        "smoke": exchange / "smoke",
        "index": exchange / "index.json",
        "release_json": root / "release_candidate.json",
        "release_md": root / "release_candidate.md",
        "recipe": root / "distillation_recipe.json",
        "matrix_config": root / "benchmark_matrix_config.json",
    }


def _resolve_suite(value: str, work_dir: Path) -> Path:
    config = BenchmarkMatrixConfig(
        version=BENCHMARK_MATRIX_VERSION,
        id="release-candidate-suite-resolution",
        suites=[value],
        model_ids=["dummy_good"],
        metric_weights=MetricWeights(),
        require_providers=[],
        source_path=str(work_dir / "suite_resolution.json"),
    )
    return resolve_suite_path(config, value).resolve()


def _write_distillation_recipe(
    path: Path,
    *,
    unit_id: str,
    domain: str,
    suite_path: Path,
    baseline_model_id: str,
    candidate_model_id: str,
    training_preflight_path: Path,
) -> Path:
    payload = {
        "version": DISTILLATION_RECIPE_VERSION,
        "id": f"{unit_id}-release-candidate",
        "domain": domain,
        "suite": str(suite_path),
        "baseline_model_id": baseline_model_id,
        "candidate_model_id": candidate_model_id,
        "training_preflight_path": str(training_preflight_path),
        "thresholds": {
            "max_pass_rate_drop": 0.0,
            "max_mean_score_drop": 0.0,
            "allow_worse_failure_modes": False,
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_matrix_config(
    path: Path,
    *,
    unit_id: str,
    suites: Sequence[Path],
    model_ids: Sequence[str],
    require_providers: Set[str],
) -> Path:
    payload = {
        "version": BENCHMARK_MATRIX_VERSION,
        "id": f"{unit_id}-release-candidate",
        "suites": [str(path) for path in suites],
        "model_ids": list(model_ids),
        "metric_weights": {"lambda_cost": 0.0, "mu_latency": 0.0},
        "require_providers": sorted(require_providers),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _matrix_model_ids(
    *,
    baseline_model_id: str,
    candidate_model_id: str,
    comparison_model_ids: Sequence[str],
) -> List[str]:
    return _unique([baseline_model_id, candidate_model_id, *comparison_model_ids])


def _matrix_contains_candidate(payload: Mapping[str, Any], candidate_model_id: str) -> bool:
    for row in payload.get("scorecard", []):
        if isinstance(row, dict) and row.get("model_id") == candidate_model_id:
            return True
    return False


def _matrix_provider_status(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in payload.get("scorecard", []):
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "suite": row.get("suite"),
                "model_id": row.get("model_id"),
                "provider": row.get("provider"),
                "provider_ok": row.get("provider_ok"),
                "provider_required": row.get("provider_required"),
                "provider_skipped": row.get("provider_skipped"),
                "skipped": row.get("skipped"),
                "issue": row.get("issue"),
            }
        )
    return out


def _add_gate(
    gates: List[Dict[str, Any]],
    name: str,
    ok: Any,
    path: Any,
    issues: Any,
) -> None:
    gates.append(
        {
            "name": name,
            "ok": bool(ok),
            "path": str(path) if path else "",
            "issues": _list(issues),
        }
    )


def _write_release_reports(
    payload: Mapping[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(format_release_candidate_markdown(payload), encoding="utf-8")


def format_release_candidate_markdown(payload: Mapping[str, Any]) -> str:
    identity = _dict(payload.get("identity"))
    lines = [
        "# SignalForge AI Release Candidate",
        "",
        f"- OK: `{str(payload.get('ok')).lower()}`",
        f"- Unit: `{identity.get('id')}@{identity.get('version')}`",
        f"- Domain: `{identity.get('domain')}`",
        f"- Work dir: `{payload.get('work_dir')}`",
        "",
        "## Gates",
        "",
        "| gate | ok | path | issues |",
        "|---|---:|---|---:|",
    ]
    for gate in payload.get("gate_results", []):
        if not isinstance(gate, dict):
            continue
        lines.append(
            f"| {gate.get('name')} | {str(gate.get('ok')).lower()} | "
            f"`{gate.get('path') or ''}` | {len(_list(gate.get('issues')))} |"
        )

    evidence = _dict(payload.get("evidence_paths"))
    if evidence:
        lines.extend(["", "## Evidence", ""])
        for key, path in sorted(evidence.items()):
            lines.append(f"- {key}: `{path}`")

    training = _dict(payload.get("training_evidence"))
    if training:
        lines.extend(["", "## Training Evidence", ""])
        lines.append(f"- Final stage: `{training.get('final_stage')}`")
        lines.append(f"- Real training evidence: `{str(training.get('real_training_evidence')).lower()}`")
        parent_real = training.get("parent_real_training_evidence")
        if parent_real is not None:
            lines.append(f"- Parent real training evidence: `{str(parent_real).lower()}`")
        lines.append(f"- Final adapter refs: `{training.get('adapter_ref_count', 0)}`")
        lines.append(f"- File checksums: `{training.get('file_checksum_count', 0)}`")

    if payload.get("issues"):
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in _list(payload.get("issues")))
    if payload.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in _list(payload.get("warnings")))
    return "\n".join(lines).rstrip() + "\n"


def _training_summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = {
        "ok": bool(payload.get("ok")),
        "mode": payload.get("mode"),
        "final_stage": payload.get("final_stage"),
        "final_run_path": payload.get("final_run_path"),
        "sft_run_path": payload.get("sft_run_path"),
        "dpo_run_path": payload.get("dpo_run_path"),
        "derived_candidate_model_id": payload.get("derived_candidate_model_id"),
        "real_training_evidence": bool(payload.get("real_training_evidence")),
        "parent_real_training_evidence": payload.get("parent_real_training_evidence"),
        "require_real_training_evidence": bool(payload.get("require_real_training_evidence")),
        "adapter_smoke": _dict(payload.get("adapter_smoke")),
        "runs": _dict(payload.get("runs")),
        "issues": _list(payload.get("issues")),
    }
    summary.update(_training_run_details(str(payload.get("final_run_path") or "")))
    return summary


def _candidate_model_id_from_training(payload: Mapping[str, Any], *, base_model: str) -> str:
    run_path = str(payload.get("final_run_path") or "")
    if not run_path:
        return ""
    try:
        run = load_training_run_report(run_path)
    except Exception:
        return ""
    refs = _adapter_refs(run)
    if not refs:
        return ""
    return f"hf:{base_model}?adapter={refs[0]}"


def _preflight_dataset_path(preflight: Mapping[str, Any], role: str) -> str:
    datasets = preflight.get("datasets")
    if not isinstance(datasets, list):
        return ""
    for row in datasets:
        if not isinstance(row, Mapping) or row.get("role") != role:
            continue
        value = row.get("source") or row.get("path")
        if value:
            return str(value)
    return ""


def _preflight_logs_root(preflight: Mapping[str, Any]) -> str:
    value = preflight.get("logs_root")
    return str(value) if value else ""


def _provider_check_payload(check: Any) -> Dict[str, Any]:
    return {
        "model_id": check.model_id,
        "provider": check.provider,
        "ok": bool(check.ok),
        "required": bool(check.required),
        "skipped": bool(check.skipped),
        "reason": check.reason,
        "details": check.details or {},
    }


def _optional_dependencies(preflight: Mapping[str, Any]) -> Dict[str, bool]:
    value = preflight.get("optional_dependencies")
    if not isinstance(value, dict):
        return {}
    return {str(key): bool(item) for key, item in value.items()}


def _require_adapter_evidence(payload: Mapping[str, Any], path: str) -> None:
    if not _adapter_refs(payload):
        raise ValueError(f"training run has no adapter refs: {path}")
    if not _file_checksums(payload):
        raise ValueError(f"training run has no file checksums: {path}")


def _is_real_training_run(payload: Mapping[str, Any]) -> bool:
    config = payload.get("command_config")
    adapter_smoke = payload.get("adapter_smoke")
    smoke_ok = isinstance(adapter_smoke, Mapping) and adapter_smoke.get("ok") is True
    return (
        payload.get("ok") is True
        and str(payload.get("status") or "") == "succeeded"
        and bool(_adapter_refs(payload))
        and bool(_file_checksums(payload))
        and smoke_ok
        and not (
            isinstance(config, Mapping)
            and str(config.get("training_evidence_mode") or "").lower() == "mock"
        )
    )


def _real_training_required_issue() -> str:
    return (
        "real training evidence required; use --run-training, pass --dpo-run, or "
        "pass --sft-run with --final-training-stage sft from a non-mock training_run.v0 "
        "that includes successful adapter_smoke evidence"
    )


def _real_dpo_required_issue() -> str:
    return (
        "real DPO evidence required; final DPO training_run.v0 must be non-mock, "
        "successful, and include adapter refs, file checksums, and successful "
        "adapter_smoke evidence"
    )


def _real_dpo_parent_required_issue() -> str:
    return (
        "real DPO evidence requires a non-mock successful parent SFT training_run.v0 "
        "with adapter refs, file checksums, and successful adapter_smoke evidence"
    )


def _invalid_training_base_model_issue() -> str:
    return (
        "real training requires a Hugging Face base model; pass --training-base-model "
        f"{DEFAULT_REAL_SFT_SMOKE_MODEL} or another valid HF model id"
    )


def _real_dpo_evidence_issues(
    *,
    dpo_payload: Mapping[str, Any],
    dpo_path: Path,
    parent_payload: Mapping[str, Any],
    parent_path: Path | None,
) -> List[str]:
    issues: List[str] = []
    if str(dpo_payload.get("training_stage") or "") != "dpo":
        issues.append(f"real DPO evidence requires training_stage 'dpo': {dpo_path}")
    if not _is_real_training_run(dpo_payload):
        issues.append(_real_dpo_required_issue())
    if not parent_payload:
        issues.append(
            "real DPO evidence requires a successful parent SFT training_run.v0"
            + (f": {parent_path}" if parent_path else "")
        )
        return issues
    if str(parent_payload.get("training_stage") or "") != "sft":
        issues.append(f"real DPO parent must have training_stage 'sft': {parent_path}")
    if not _is_real_training_run(parent_payload):
        issues.append(_real_dpo_parent_required_issue())
    return issues


def _load_dpo_parent_training_run(
    dpo_payload: Mapping[str, Any],
    *,
    explicit_sft_run: str | Path | None,
) -> tuple[Dict[str, Any], Path | None, List[str]]:
    parent_path = _dpo_parent_path(dpo_payload, explicit_sft_run=explicit_sft_run)
    if parent_path is None:
        return {}, None, ["DPO training run has no parent SFT run path"]
    try:
        parent_payload = load_training_run_report(parent_path)
    except Exception as exc:
        return {}, parent_path, [f"DPO parent SFT run is not valid: {exc}"]
    try:
        _require_adapter_evidence(parent_payload, str(parent_path))
    except ValueError as exc:
        return parent_payload, parent_path, [f"DPO parent SFT run is not valid: {exc}"]
    return parent_payload, parent_path, []


def _dpo_parent_path(
    dpo_payload: Mapping[str, Any],
    *,
    explicit_sft_run: str | Path | None,
) -> Path | None:
    if explicit_sft_run is not None:
        return Path(explicit_sft_run).resolve()
    parent = dpo_payload.get("dpo_parent_run")
    if isinstance(parent, Mapping):
        value = parent.get("path")
        if value:
            return Path(str(value)).resolve()
    return None


def _training_run_details(path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "training_stage": None,
        "training_status": None,
        "final_adapter_refs": [],
        "adapter_ref_count": 0,
        "file_checksum_count": 0,
        "dpo_parent_run": {},
    }
    if not path:
        return out
    try:
        payload = _load_json(Path(path))
    except Exception:
        return out
    refs = _adapter_refs(payload)
    checksums = _file_checksums(payload)
    out.update(
        {
            "training_stage": payload.get("training_stage"),
            "training_status": payload.get("status"),
            "final_adapter_refs": refs,
            "adapter_ref_count": len(refs),
            "file_checksum_count": len(checksums),
            "dpo_parent_run": _dict(payload.get("dpo_parent_run")),
        }
    )
    return out


def _is_placeholder_training_base_model(base_model: str) -> bool:
    return str(base_model or "").strip() in PLACEHOLDER_TRAINING_BASE_MODELS


def _adapter_refs(payload: Mapping[str, Any]) -> List[str]:
    final_refs = payload.get("final_adapter_refs")
    if isinstance(final_refs, list) and final_refs:
        return [str(value) for value in final_refs if str(value)]
    artifacts = payload.get("artifact_refs")
    if isinstance(artifacts, dict):
        adapters = artifacts.get("adapters")
        if isinstance(adapters, list):
            return [str(value) for value in adapters if str(value)]
    return []


def _file_checksums(payload: Mapping[str, Any]) -> Dict[str, str]:
    checksums = payload.get("file_checksums")
    if not isinstance(checksums, Mapping):
        return {}
    out: Dict[str, str] = {}
    for key, value in checksums.items():
        if key is None or value is None:
            continue
        key_text = str(key).strip()
        value_text = str(value).strip()
        if key_text and value_text:
            out[key_text] = value_text
    return out


def _first_adapter_ref(parent: Mapping[str, Any]) -> str:
    refs = parent.get("adapter_refs")
    if isinstance(refs, list) and refs:
        return str(refs[0])
    return ""


def _unique(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        item = str(value)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run SignalForge AI release-candidate evidence checks."
    )
    parser.add_argument("--work-dir", required=True, type=str, help="Output work directory")
    parser.add_argument("--mode", default="dummy", choices=["dummy"], help="Release-candidate mode")
    parser.add_argument("--id", default="pilot-specialist", help="Specialist unit id")
    parser.add_argument("--name", default="Pilot Specialist", help="Specialist unit display name")
    parser.add_argument("--version", default="0.7.0", help="Specialist unit version")
    parser.add_argument("--domain", default="pilot", help="Specialist unit domain")
    parser.add_argument("--baseline-model-id", default="dummy_good", help="Baseline/teacher model id")
    parser.add_argument(
        "--candidate-model-id",
        default=None,
        help="Candidate specialist model id; defaults to dummy_good, or the trained HF adapter in --run-training mode",
    )
    parser.add_argument("--training-base-model", default="dummy/base", help="Training base model")
    parser.add_argument("--suite", action="append", default=[], help="Suite alias/path; may be repeated")
    parser.add_argument(
        "--comparison-model-id",
        action="append",
        default=[],
        help="Additional model id for benchmark matrix evidence",
    )
    parser.add_argument("--matrix-config", type=str, default=None, help="Optional benchmark_matrix.v0 config")
    parser.add_argument("--sft-run", type=str, default=None, help="Existing successful SFT training_run.v0")
    parser.add_argument("--dpo-run", type=str, default=None, help="Existing successful DPO training_run.v0")
    parser.add_argument(
        "--run-training",
        action="store_true",
        help="Run opt-in real SFT training from generated pilot datasets",
    )
    parser.add_argument(
        "--run-dpo",
        action="store_true",
        help="After --run-training SFT, run opt-in DPO and package the DPO adapter",
    )
    parser.add_argument(
        "--training-max-steps",
        type=int,
        default=1,
        help="Max training steps for opt-in real training runs",
    )
    parser.add_argument(
        "--final-training-stage",
        choices=["auto", "sft", "dpo"],
        default="auto",
        help="Select which training run is packaged as final evidence",
    )
    parser.add_argument(
        "--require-real-training-evidence",
        action="store_true",
        help="Fail if the final training run was generated by mock release evidence",
    )
    parser.add_argument(
        "--require-provider",
        action="append",
        default=[],
        choices=["hosted", "local", "all"],
        help="Fail if selected provider path is unavailable",
    )
    args = parser.parse_args(argv)
    if args.run_training and (args.sft_run or args.dpo_run):
        parser.error("--run-training cannot be combined with --sft-run or --dpo-run")
    if args.run_dpo and not args.run_training:
        parser.error("--run-dpo requires --run-training")
    if args.run_training and args.final_training_stage == "dpo" and not args.run_dpo:
        parser.error("--final-training-stage dpo requires --run-dpo")
    if args.run_training and args.run_dpo and args.final_training_stage == "sft":
        parser.error("--final-training-stage sft cannot be combined with --run-dpo")
    if args.run_training and _is_placeholder_training_base_model(args.training_base_model):
        parser.error(_invalid_training_base_model_issue())

    payload = run_release_candidate_check(
        work_dir=args.work_dir,
        mode=args.mode,
        unit_id=args.id,
        name=args.name,
        version=args.version,
        domain=args.domain,
        baseline_model_id=args.baseline_model_id,
        candidate_model_id=args.candidate_model_id,
        training_base_model=args.training_base_model,
        suites=args.suite or None,
        comparison_model_ids=args.comparison_model_id,
        matrix_config=args.matrix_config,
        sft_run=args.sft_run,
        dpo_run=args.dpo_run,
        run_training=bool(args.run_training),
        run_dpo=bool(args.run_dpo),
        training_max_steps=int(args.training_max_steps),
        final_training_stage=args.final_training_stage,
        require_real_training_evidence=bool(args.require_real_training_evidence),
        require_provider=args.require_provider,
    )
    print(f"Release candidate: {'OK' if payload['ok'] else 'FAILED'}")
    print(f"Report JSON: {payload['report_json']}")
    print(f"Report Markdown: {payload['report_md']}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
