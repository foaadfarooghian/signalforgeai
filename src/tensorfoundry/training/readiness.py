"""Training preflight and evidence contracts for TensorFoundry."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from tensorfoundry.export.validate import DatasetValidationResult


TRAINING_PREFLIGHT_VERSION = "training_preflight.v0"
TRAINING_ARTIFACT_VERSION = "training_artifact.v0"
TRAINING_SMOKE_VERSION = "training_smoke.v0"
TRAINING_DEPENDENCIES = ("torch", "datasets", "transformers", "trl", "unsloth")


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
