"""Validation for specialist model exchange unit manifests."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping
from urllib.parse import urlparse

from jsonschema import Draft202012Validator


SPECIALIST_MODEL_UNIT_VERSION = "specialist_model_unit.v0"
SPECIALIST_VALIDATION_VERSION = "specialist_model_unit_validation.v0"


@dataclass(frozen=True)
class ValidationReport:
    """Validation result for one specialist model unit manifest."""

    version: str
    ok: bool
    manifest_path: str
    schema_ok: bool
    release_ready: bool
    issues: List[str]
    warnings: List[str]
    artifacts_checked: List[Dict[str, Any]]
    evidence_checked: List[Dict[str, Any]]
    computed_checksums: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_manifest(
    manifest_path: str | Path,
    *,
    artifacts_root: str | Path | None = None,
    release_ready: bool = False,
    report_out: str | Path | None = None,
) -> ValidationReport:
    """Validate a `specialist_model_unit.v0` manifest and optional release gates."""
    path = Path(manifest_path)
    issues: List[str] = []
    warnings: List[str] = []
    artifacts_checked: List[Dict[str, Any]] = []
    evidence_checked: List[Dict[str, Any]] = []
    computed_checksums: Dict[str, str] = {}

    try:
        manifest = _load_json(path)
    except ValueError as exc:
        report = ValidationReport(
            version=SPECIALIST_VALIDATION_VERSION,
            ok=False,
            manifest_path=str(path),
            schema_ok=False,
            release_ready=release_ready,
            issues=[str(exc)],
            warnings=[],
            artifacts_checked=[],
            evidence_checked=[],
            computed_checksums={},
        )
        _maybe_write_report(report, report_out)
        return report

    schema = load_unit_schema()
    schema_issues = _schema_issues(manifest, schema)
    issues.extend(schema_issues)
    schema_ok = not schema_issues
    base_dir = Path(artifacts_root) if artifacts_root is not None else path.parent

    if isinstance(manifest, dict):
        checksums_raw = _nested_dict(manifest, "artifacts", "checksums")
        checksums = {str(k): str(v) for k, v in checksums_raw.items()}
        for ref in _artifact_refs(manifest):
            check = _check_ref(
                ref,
                base_dir=base_dir,
                checksums=checksums,
                missing_is_issue=release_ready,
            )
            artifacts_checked.append(check)
            if check.get("sha256"):
                computed_checksums[str(ref)] = str(check["sha256"])
            if check.get("issue"):
                issues.append(str(check["issue"]))
            if check.get("warning"):
                warnings.append(str(check["warning"]))

        for ref in _evidence_refs(manifest):
            check = _check_ref(
                ref,
                base_dir=path.parent,
                checksums={},
                missing_is_issue=True,
                checksum_required=False,
            )
            evidence_checked.append(check)
            if check.get("issue"):
                issues.append(str(check["issue"]))
            if check.get("warning"):
                warnings.append(str(check["warning"]))

        if release_ready:
            if not _usage_constraints(manifest):
                issues.append("release_ready: usage_constraints must not be empty")
            if not _failure_modes(manifest):
                issues.append("release_ready: failure_modes must not be empty")
            if not _has_runnable_artifact(manifest):
                issues.append("release_ready: at least one runnable artifact ref is required")
            if not _has_evidence_links(manifest):
                issues.append("release_ready: training, distillation, and benchmark evidence links are required")

    report = ValidationReport(
        version=SPECIALIST_VALIDATION_VERSION,
        ok=not issues,
        manifest_path=str(path),
        schema_ok=schema_ok,
        release_ready=release_ready,
        issues=issues,
        warnings=warnings,
        artifacts_checked=artifacts_checked,
        evidence_checked=evidence_checked,
        computed_checksums=computed_checksums,
    )
    _maybe_write_report(report, report_out)
    return report


def load_unit_schema() -> Dict[str, Any]:
    """Load the packaged `specialist_model_unit.v0` JSON schema."""
    schema_file = (
        resources.files("tensorfoundry.exchange.schemas")
        / "specialist_model_unit.schema.json"
    )
    return json.loads(schema_file.read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 hex digest for a local file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_uri_ref(value: str) -> bool:
    """Return true when a ref is an external URI-style artifact reference."""
    parsed = urlparse(value)
    return bool(parsed.scheme and (parsed.netloc or parsed.scheme in {"hf", "s3", "gs", "oci"}))


def parse_checksum_args(values: Iterable[str]) -> Dict[str, str]:
    """Parse repeatable `REF=SHA256` CLI values."""
    out: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"checksum must be REF=SHA256: {value}")
        ref, digest = value.split("=", 1)
        ref = ref.strip()
        digest = digest.strip().lower()
        if not ref or not digest:
            raise ValueError(f"checksum must be REF=SHA256: {value}")
        out[ref] = digest
    return out


def _schema_issues(manifest: Mapping[str, Any], schema: Mapping[str, Any]) -> List[str]:
    validator = Draft202012Validator(schema)
    issues: List[str] = []
    for error in sorted(validator.iter_errors(manifest), key=lambda e: list(e.path)):
        location = ".".join(str(part) for part in error.path) or "<root>"
        issues.append(f"schema:{location}: {error.message}")
    return issues


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ValueError(f"manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return payload


def _check_ref(
    ref: str,
    *,
    base_dir: Path,
    checksums: Mapping[str, str],
    missing_is_issue: bool,
    checksum_required: bool = True,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ref": ref}
    if not ref:
        out["issue"] = "empty reference"
        return out
    if is_uri_ref(ref):
        out.update({"kind": "uri", "exists": None, "ok": True})
        return out

    path = Path(ref)
    if not path.is_absolute():
        path = base_dir / path
    out.update({"kind": "local", "path": str(path), "exists": path.exists()})
    if not path.exists():
        message = f"local ref missing: {ref}"
        if missing_is_issue:
            out["issue"] = message
            out["ok"] = False
        else:
            out["warning"] = message
            out["ok"] = True
        return out
    if path.is_dir():
        out["ok"] = True
        return out

    digest = sha256_file(path)
    expected = checksums.get(ref) or checksums.get(str(path))
    out["sha256"] = digest
    if expected:
        out["expected_sha256"] = expected
        if expected.lower() != digest:
            out["issue"] = f"checksum mismatch for {ref}: expected {expected}, got {digest}"
            out["ok"] = False
            return out
    elif checksum_required:
        out["computed"] = True
    out["ok"] = True
    return out


def _artifact_refs(manifest: Mapping[str, Any]) -> List[str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return []
    refs: List[str] = []
    for key in ("adapters", "safetensors", "gguf"):
        values = artifacts.get(key)
        if isinstance(values, list):
            refs.extend(str(value) for value in values if str(value))
    ollama = artifacts.get("ollama")
    if isinstance(ollama, dict) and str(ollama.get("modelfile") or ""):
        refs.append(str(ollama["modelfile"]))
    return refs


def _evidence_refs(manifest: Mapping[str, Any]) -> List[str]:
    refs: List[str] = []
    eval_pack = manifest.get("eval_pack")
    if isinstance(eval_pack, dict):
        refs.extend(
            str(eval_pack[key])
            for key in (
                "report_path",
                "distillation_eval_path",
                "benchmark_matrix_path",
            )
            if str(eval_pack.get(key) or "")
        )
    lineage = manifest.get("lineage")
    if isinstance(lineage, dict):
        for key in ("training_preflight", "distillation_recipe"):
            if str(lineage.get(key) or ""):
                refs.append(str(lineage[key]))
        for key in ("trace_sources", "dataset_sources"):
            values = lineage.get(key)
            if isinstance(values, list):
                refs.extend(str(value) for value in values if str(value))
    return refs


def _has_runnable_artifact(manifest: Mapping[str, Any]) -> bool:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return False
    for key in ("adapters", "safetensors", "gguf"):
        values = artifacts.get(key)
        if isinstance(values, list) and any(str(value) for value in values):
            return True
    ollama = artifacts.get("ollama")
    return isinstance(ollama, dict) and bool(str(ollama.get("tag") or ""))


def _has_evidence_links(manifest: Mapping[str, Any]) -> bool:
    eval_pack = manifest.get("eval_pack")
    lineage = manifest.get("lineage")
    if not isinstance(eval_pack, dict) or not isinstance(lineage, dict):
        return False
    return bool(
        eval_pack.get("distillation_eval_path")
        and eval_pack.get("benchmark_matrix_path")
        and lineage.get("training_preflight")
    )


def _usage_constraints(manifest: Mapping[str, Any]) -> List[str]:
    values = manifest.get("usage_constraints")
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value)]


def _failure_modes(manifest: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    values = manifest.get("failure_modes")
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, dict)]


def _nested_dict(payload: Mapping[str, Any], *keys: str) -> Dict[str, Any]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return dict(current) if isinstance(current, dict) else {}


def _maybe_write_report(report: ValidationReport, report_out: str | Path | None) -> None:
    if report_out is None:
        return
    out = Path(report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
