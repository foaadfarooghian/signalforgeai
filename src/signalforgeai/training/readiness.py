"""Training preflight and evidence contracts for SignalForge AI."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from signalforgeai.export.validate import DatasetValidationResult


TRAINING_PREFLIGHT_VERSION = "training_preflight.v0"
TRAINING_ARTIFACT_VERSION = "training_artifact.v0"
TRAINING_SMOKE_VERSION = "training_smoke.v0"
TRAINING_RUN_VERSION = "training_run.v0"
ADAPTER_SMOKE_VERSION = "adapter_smoke.v0"
TRAINING_DEPENDENCIES = ("torch", "datasets", "transformers", "trl", "unsloth")
TRAINING_OUTPUT_SUFFIXES = (".safetensors", ".bin", ".json", ".model", ".txt")


@dataclass(frozen=True)
class TrainingDatasetCheck:
    """Validation result for one training dataset input."""

    role: str
    path: str | Path
    result: DatasetValidationResult


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp for evidence artifacts."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def check_optional_training_dependencies() -> Dict[str, bool]:
    """Check whether optional training dependencies are importable."""
    return {name: find_spec(name) is not None for name in TRAINING_DEPENDENCIES}


def build_training_preflight(
    *,
    base_model: str,
    dataset_checks: Sequence[TrainingDatasetCheck],
    outputs: Mapping[str, Optional[str]],
    quality_gate: bool,
    logs_root: Optional[str],
    optional_dependencies: Mapping[str, bool],
    smoke_requested: bool,
    max_steps: int,
    dry_run: bool,
    config: Mapping[str, Any],
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a `training_preflight.v0` report payload."""
    created_at = created_at or utc_now()
    datasets = [_dataset_payload(check) for check in dataset_checks]
    deps = dict(optional_dependencies)
    missing_dependencies = sorted(name for name, ok in deps.items() if not ok)
    dependencies_ok = not missing_dependencies
    dataset_ok = all(bool(check.result.ok) for check in dataset_checks)
    smoke_ok = dependencies_ok if smoke_requested else True
    issues = _collect_issues(dataset_checks)
    if smoke_requested and missing_dependencies:
        joined = ", ".join(missing_dependencies)
        issues.append(f"missing optional training dependencies: {joined}")

    artifact_manifest = build_training_artifact(
        base_model=base_model,
        dataset_checks=dataset_checks,
        outputs=outputs,
        quality_gate=quality_gate,
        logs_root=logs_root,
        optional_dependencies=deps,
        config=config,
        created_at=created_at,
    )
    smoke = build_training_smoke(
        requested=smoke_requested,
        dependencies_ok=dependencies_ok,
        missing_dependencies=missing_dependencies,
        max_steps=max_steps,
        dry_run=dry_run,
        created_at=created_at,
    )

    payload: Dict[str, Any] = {
        "version": TRAINING_PREFLIGHT_VERSION,
        "ok": bool(dataset_ok and smoke_ok),
        "created_at": created_at,
        "dry_run": bool(dry_run),
        "quality_gate": bool(quality_gate),
        "logs_root": logs_root,
        "base_model": base_model,
        "datasets": datasets,
        "optional_dependencies": deps,
        "dependencies_ok": dependencies_ok,
        "outputs": dict(outputs),
        "smoke": smoke,
        "max_steps": int(max_steps),
        "artifact_manifest": artifact_manifest,
        "issues": issues,
    }
    if smoke_requested:
        payload["smoke_evidence"] = smoke
    return payload


def build_training_artifact(
    *,
    base_model: str,
    dataset_checks: Sequence[TrainingDatasetCheck],
    outputs: Mapping[str, Optional[str]],
    quality_gate: bool,
    logs_root: Optional[str],
    optional_dependencies: Mapping[str, bool],
    config: Mapping[str, Any],
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the minimal `training_artifact.v0` evidence manifest."""
    created_at = created_at or utc_now()
    datasets = [_dataset_artifact_payload(check) for check in dataset_checks]
    return {
        "version": TRAINING_ARTIFACT_VERSION,
        "created_at": created_at,
        "base_model": base_model,
        "dataset_sources": {check.role: str(check.path) for check in dataset_checks},
        "dataset_hashes": {
            check.role: check.result.content_sha256 for check in dataset_checks
        },
        "split_counts": {
            check.role: dict(check.result.split_counts) for check in dataset_checks
        },
        "datasets": datasets,
        "outputs": dict(outputs),
        "quality_gate": bool(quality_gate),
        "logs_root": logs_root,
        "command_config": dict(config),
        "dependency_checks": dict(optional_dependencies),
    }


def build_training_smoke(
    *,
    requested: bool,
    dependencies_ok: bool,
    missing_dependencies: Sequence[str],
    max_steps: int,
    dry_run: bool,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a `training_smoke.v0` evidence block."""
    created_at = created_at or utc_now()
    status = "not_requested"
    if requested and dependencies_ok:
        status = "planned" if dry_run else "ready"
    elif requested:
        status = "blocked_missing_dependencies"
    return {
        "version": TRAINING_SMOKE_VERSION,
        "created_at": created_at,
        "requested": bool(requested),
        "dry_run": bool(dry_run),
        "max_steps": int(max_steps),
        "dependencies_ok": bool(dependencies_ok),
        "missing_dependencies": list(missing_dependencies),
        "would_launch_training": bool(requested and dependencies_ok and not dry_run),
        "status": status,
    }


def write_training_preflight(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write a training preflight report JSON file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_training_preflight_report(path: str | Path) -> Dict[str, Any]:
    """Load and validate a successful `training_preflight.v0` report."""
    payload = _load_json_object(path, label="training preflight")
    if payload.get("version") != TRAINING_PREFLIGHT_VERSION:
        raise ValueError(
            f"unsupported training preflight version: {payload.get('version')!r}"
        )
    if payload.get("ok") is not True:
        raise ValueError(f"training preflight is not ok: {path}")
    return payload


def load_training_run_report(path: str | Path) -> Dict[str, Any]:
    """Load and validate a successful `training_run.v0` report."""
    payload = _load_json_object(path, label="training run")
    if payload.get("version") != TRAINING_RUN_VERSION:
        raise ValueError(f"unsupported training run version: {payload.get('version')!r}")
    if payload.get("ok") is not True:
        raise ValueError(f"training run is not ok: {path}")
    return payload


def write_training_run(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write a training run report JSON file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def build_training_run(
    *,
    base_model: str,
    preflight: Mapping[str, Any] | None,
    preflight_path: str | Path | None,
    outputs: Mapping[str, Optional[str]],
    optional_dependencies: Mapping[str, bool],
    config: Mapping[str, Any],
    dpo_parent_run: Mapping[str, Any] | None = None,
    status: str,
    started_at: str,
    duration_seconds: float,
    issues: Sequence[str] = (),
    adapter_smoke: Mapping[str, Any] | None = None,
    completed_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build `training_run.v0` evidence for an attempted training execution."""
    completed_at = completed_at or utc_now()
    preflight_ok = preflight is None or preflight.get("ok") is True
    artifact_scan = scan_training_output_artifacts(outputs)
    run_issues = list(issues)
    if not preflight_ok:
        run_issues.append("training preflight is not ok")
    if status == "succeeded" and not artifact_scan["artifact_refs"]["adapters"]:
        run_issues.append("no training output artifacts found")
    if (
        status == "succeeded"
        and artifact_scan["artifact_refs"]["adapters"]
        and not artifact_scan["file_checksums"]
    ):
        run_issues.append("no training output file checksums found")
    if adapter_smoke is not None and adapter_smoke.get("ok") is not True:
        reason = str(adapter_smoke.get("reason") or "adapter smoke failed")
        run_issues.append(f"adapter smoke failed: {reason}")
    stage = _training_stage(config)
    deps = dict(optional_dependencies)
    missing_dependencies = sorted(name for name, ok in deps.items() if not ok)
    return {
        "version": TRAINING_RUN_VERSION,
        "ok": bool(status == "succeeded" and preflight_ok and not run_issues),
        "status": status,
        "created_at": completed_at,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": round(float(duration_seconds), 6),
        "base_model": base_model,
        "training_stage": stage,
        "preflight_path": str(preflight_path) if preflight_path is not None else None,
        "preflight_ok": bool(preflight_ok),
        "dpo_parent_run": dict(dpo_parent_run) if isinstance(dpo_parent_run, Mapping) else None,
        "datasets": _preflight_datasets(preflight),
        "dataset_hashes": _preflight_dataset_hashes(preflight),
        "split_counts": _preflight_split_counts(preflight),
        "outputs": dict(outputs),
        "artifact_refs": artifact_scan["artifact_refs"],
        "artifact_refs_by_role": artifact_scan["artifact_refs_by_role"],
        "final_adapter_refs": artifact_scan["final_adapter_refs"],
        "output_artifacts": artifact_scan["output_artifacts"],
        "file_checksums": artifact_scan["file_checksums"],
        "missing_outputs": artifact_scan["missing_outputs"],
        "optional_dependencies": deps,
        "dependencies_ok": not missing_dependencies,
        "missing_dependencies": missing_dependencies,
        "smoke_bounded": bool(config.get("smoke") and int(config.get("max_steps") or 0) == 1),
        "adapter_smoke": dict(adapter_smoke) if isinstance(adapter_smoke, Mapping) else None,
        "command_config": dict(config),
        "issues": run_issues,
    }


def scan_training_output_artifacts(
    outputs: Mapping[str, Optional[str]],
) -> Dict[str, Any]:
    """Scan training output dirs for runnable refs and checksum-worthy files."""
    refs_by_role: Dict[str, list[str]] = {"sft_out": [], "dpo_out": []}
    output_artifacts: list[Dict[str, Any]] = []
    file_checksums: Dict[str, str] = {}
    missing_outputs: list[str] = []
    for role in ("sft_out", "dpo_out"):
        value = outputs.get(role)
        if not value:
            continue
        root = Path(str(value))
        if not root.exists():
            missing_outputs.append(str(root))
            continue
        refs_by_role[role].append(str(root))
        if root.is_file():
            _record_output_file(root, role, output_artifacts, file_checksums)
            continue
        for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
            if file_path.suffix.lower() in TRAINING_OUTPUT_SUFFIXES:
                _record_output_file(file_path, role, output_artifacts, file_checksums)
    final_adapter_refs = refs_by_role["dpo_out"] or refs_by_role["sft_out"]
    artifact_refs: Dict[str, Any] = {
        "adapters": list(final_adapter_refs),
        "safetensors": [],
        "gguf": [],
        "ollama": {"modelfile": "", "tag": ""},
    }
    return {
        "artifact_refs": artifact_refs,
        "artifact_refs_by_role": refs_by_role,
        "final_adapter_refs": list(final_adapter_refs),
        "output_artifacts": output_artifacts,
        "file_checksums": file_checksums,
        "missing_outputs": missing_outputs,
    }


def summarize_dpo_parent_run(path: str | Path) -> Dict[str, Any]:
    """Load a successful SFT run report and summarize it as DPO parent evidence."""
    payload = load_training_run_report(path)
    adapters = _adapter_refs(payload)
    if not adapters:
        raise ValueError(f"DPO parent SFT run has no adapter refs: {path}")
    checksums = payload.get("file_checksums")
    return {
        "path": str(path),
        "version": TRAINING_RUN_VERSION,
        "ok": True,
        "status": str(payload.get("status") or ""),
        "base_model": str(payload.get("base_model") or ""),
        "training_stage": str(payload.get("training_stage") or ""),
        "adapter_refs": adapters,
        "file_checksums": dict(checksums) if isinstance(checksums, Mapping) else {},
    }


def _dataset_payload(check: TrainingDatasetCheck) -> Dict[str, Any]:
    payload = check.result.to_dict()
    payload["role"] = check.role
    payload["source"] = str(check.path)
    return payload


def _dataset_artifact_payload(check: TrainingDatasetCheck) -> Dict[str, Any]:
    result = check.result
    return {
        "role": check.role,
        "kind": result.kind,
        "path": str(check.path),
        "rows": result.rows,
        "ok": result.ok,
        "content_sha256": result.content_sha256,
        "split_counts": dict(result.split_counts),
        "duplicate_count": result.duplicate_count,
        "provenance_checked": result.provenance_checked,
    }


def _collect_issues(dataset_checks: Sequence[TrainingDatasetCheck]) -> list[str]:
    issues: list[str] = []
    for check in dataset_checks:
        for issue in check.result.issues:
            issues.append(f"{check.role}: {issue}")
        for issue in check.result.quality_issues:
            issues.append(f"{check.role}: {issue}")
    return issues


def _training_stage(config: Mapping[str, Any]) -> str:
    if config.get("sft") and config.get("dpo"):
        return "sft_dpo"
    if config.get("dpo"):
        return "dpo"
    if config.get("sft"):
        return "sft"
    return "unknown"


def _adapter_refs(payload: Mapping[str, Any]) -> list[str]:
    final_refs = payload.get("final_adapter_refs")
    if isinstance(final_refs, list) and final_refs:
        return [str(value) for value in final_refs if str(value)]
    artifact_refs = payload.get("artifact_refs")
    if isinstance(artifact_refs, Mapping):
        adapters = artifact_refs.get("adapters")
        if isinstance(adapters, list):
            return [str(value) for value in adapters if str(value)]
    return []


def _record_output_file(
    path: Path,
    role: str,
    output_artifacts: list[Dict[str, Any]],
    file_checksums: Dict[str, str],
) -> None:
    digest = _sha256_file(path)
    file_checksums[str(path)] = digest
    output_artifacts.append(
        {
            "role": role,
            "path": str(path),
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
    )


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json_object(path: str | Path, *, label: str) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _preflight_artifact(preflight: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(preflight, Mapping):
        return {}
    artifact = preflight.get("artifact_manifest")
    return artifact if isinstance(artifact, Mapping) else {}


def _preflight_datasets(preflight: Mapping[str, Any] | None) -> list[Dict[str, Any]]:
    artifact = _preflight_artifact(preflight)
    datasets = artifact.get("datasets")
    if isinstance(datasets, list):
        return [dict(row) for row in datasets if isinstance(row, Mapping)]
    return []


def _preflight_dataset_hashes(preflight: Mapping[str, Any] | None) -> Dict[str, Any]:
    hashes = _preflight_artifact(preflight).get("dataset_hashes")
    return dict(hashes) if isinstance(hashes, Mapping) else {}


def _preflight_split_counts(preflight: Mapping[str, Any] | None) -> Dict[str, Any]:
    splits = _preflight_artifact(preflight).get("split_counts")
    return dict(splits) if isinstance(splits, Mapping) else {}


def monotonic_seconds() -> float:
    """Expose a monotonic clock for CLI timing and tests."""
    return time.perf_counter()
