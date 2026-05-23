"""Specialist package evidence checks for exchange units."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

from signalforgeai.exchange.validation import is_uri_ref, sha256_file, validate_manifest


SPECIALIST_PACKAGE_VERSION = "specialist_package.v0"
PACKAGE_TYPES = ("adapters", "safetensors", "gguf", "ollama")
PACKAGE_ALIASES = {"adapter": "adapters", "adapters": "adapters"}


def run_package_check(
    *,
    manifest_path: str | Path,
    out_path: str | Path,
    artifacts_root: str | Path | None = None,
    package_types: Optional[Iterable[str]] = None,
    release_ready: bool = False,
    update_manifest: bool = False,
) -> Dict[str, Any]:
    """Validate packaged artifact refs and write `specialist_package.v0` evidence."""
    manifest_file = Path(manifest_path)
    out = Path(out_path)
    manifest = _load_manifest(manifest_file)
    base_dir = Path(artifacts_root) if artifacts_root is not None else manifest_file.parent
    issues: List[str] = []
    warnings: List[str] = []

    validation = validate_manifest(manifest_file, artifacts_root=artifacts_root, release_ready=False)
    issues.extend(validation.issues)
    warnings.extend(validation.warnings)

    requested = _resolve_package_types(manifest, package_types)
    if not requested:
        issues.append("package_check: no package artifact refs found")

    artifacts = _dict(manifest.get("artifacts"))
    checksums = {str(k): str(v) for k, v in _dict(artifacts.get("checksums")).items()}
    artifact_checks: List[Dict[str, Any]] = []
    for package_type in requested:
        if package_type == "ollama":
            artifact_checks.extend(
                _check_ollama_package(
                    artifacts,
                    base_dir=base_dir,
                    checksums=checksums,
                    release_ready=release_ready,
                )
            )
            continue
        refs = artifacts.get(package_type)
        if not isinstance(refs, list) or not refs:
            issues.append(f"package_check: missing {package_type} refs")
            continue
        for ref in refs:
            check = _check_artifact_ref(
                str(ref),
                role=package_type,
                base_dir=base_dir,
                checksums=checksums,
                release_ready=release_ready,
            )
            artifact_checks.append(check)

    for check in artifact_checks:
        if check.get("issue"):
            issues.append(str(check["issue"]))
        if check.get("warning"):
            warnings.append(str(check["warning"]))

    payload = {
        "version": SPECIALIST_PACKAGE_VERSION,
        "ok": not issues,
        "created_at": _utc_now(),
        "manifest_path": str(manifest_file),
        "artifacts_root": str(base_dir),
        "release_ready": bool(release_ready),
        "package_types": sorted(requested),
        "artifacts": artifact_checks,
        "issues": issues,
        "warnings": warnings,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if update_manifest:
        _attach_package_evidence(manifest_file, out, payload)
    return payload


def _resolve_package_types(
    manifest: Mapping[str, Any],
    package_types: Optional[Iterable[str]],
) -> Set[str]:
    if package_types:
        requested: Set[str] = set()
        for value in package_types:
            normalized = PACKAGE_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())
            if normalized == "auto":
                requested.update(_present_package_types(manifest))
            elif normalized in PACKAGE_TYPES:
                requested.add(normalized)
            else:
                raise ValueError(f"unsupported package type: {value}")
        return requested
    return _present_package_types(manifest)


def _present_package_types(manifest: Mapping[str, Any]) -> Set[str]:
    artifacts = _dict(manifest.get("artifacts"))
    out: Set[str] = set()
    for key in ("adapters", "safetensors", "gguf"):
        refs = artifacts.get(key)
        if isinstance(refs, list) and any(str(ref) for ref in refs):
            out.add(key)
    ollama = _dict(artifacts.get("ollama"))
    if str(ollama.get("modelfile") or "") or str(ollama.get("tag") or ""):
        out.add("ollama")
    return out


def _check_ollama_package(
    artifacts: Mapping[str, Any],
    *,
    base_dir: Path,
    checksums: Mapping[str, str],
    release_ready: bool,
) -> List[Dict[str, Any]]:
    ollama = _dict(artifacts.get("ollama"))
    checks: List[Dict[str, Any]] = []
    tag = str(ollama.get("tag") or "")
    modelfile = str(ollama.get("modelfile") or "")
    if not tag:
        checks.append({"role": "ollama_tag", "ok": False, "issue": "package_check: missing ollama tag"})
    else:
        checks.append({"role": "ollama_tag", "ref": tag, "kind": "tag", "ok": True})
    if not modelfile:
        checks.append(
            {"role": "ollama_modelfile", "ok": False, "issue": "package_check: missing ollama modelfile"}
        )
        return checks
    check = _check_artifact_ref(
        modelfile,
        role="ollama_modelfile",
        base_dir=base_dir,
        checksums=checksums,
        release_ready=release_ready,
    )
    if check.get("kind") == "local" and check.get("exists") is True:
        path = Path(str(check["path"]))
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            if "FROM " not in text.upper():
                check["issue"] = f"invalid Ollama Modelfile: missing FROM line: {modelfile}"
                check["ok"] = False
    checks.append(check)
    return checks


def _check_artifact_ref(
    ref: str,
    *,
    role: str,
    base_dir: Path,
    checksums: Mapping[str, str],
    release_ready: bool,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"role": role, "ref": ref}
    if is_uri_ref(ref):
        out.update({"kind": "uri", "exists": None, "ok": True})
        return out
    path = Path(ref)
    if not path.is_absolute():
        path = base_dir / path
    out.update({"kind": "local", "path": str(path), "exists": path.exists()})
    if not path.exists():
        key = "issue" if release_ready else "warning"
        out[key] = f"local package ref missing: {ref}"
        out["ok"] = not release_ready
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
    out["ok"] = True
    return out


def _attach_package_evidence(manifest_path: Path, evidence_path: Path, payload: Mapping[str, Any]) -> None:
    manifest = _load_manifest(manifest_path)
    manifest["package_evidence"] = {
        "version": SPECIALIST_PACKAGE_VERSION,
        "path": str(evidence_path.resolve()),
        "ok": bool(payload.get("ok")),
        "package_types": list(payload.get("package_types", [])),
        "artifact_count": len(payload.get("artifacts", [])) if isinstance(payload.get("artifacts"), list) else 0,
        "issues": list(payload.get("issues", [])) if isinstance(payload.get("issues"), list) else [],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_manifest(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return payload


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
