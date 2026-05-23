"""Consumer smoke-run evidence for specialist exchange units."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from signalforgeai.exchange.package import SPECIALIST_PACKAGE_VERSION
from signalforgeai.exchange.validation import validate_manifest
from signalforgeai.models.registry import check_provider_for_model, get_provider_for_model


SPECIALIST_SMOKE_VERSION = "specialist_smoke.v0"


def run_smoke_check(
    *,
    manifest_path: str | Path,
    work_dir: str | Path,
    mode: str = "dummy",
    model_id: Optional[str] = None,
    package_evidence_path: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    require_provider: bool = False,
    update_manifest: bool = False,
) -> Dict[str, Any]:
    """Run an offline/default consumer smoke check and write evidence."""
    manifest_file = Path(manifest_path)
    out_dir = Path(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_file)
    issues: list[str] = []
    warnings: list[str] = []

    validation = validate_manifest(manifest_file, artifacts_root=artifacts_root, release_ready=True)
    issues.extend(validation.issues)
    warnings.extend(validation.warnings)

    package_evidence = _load_package_evidence(
        manifest,
        manifest_path=manifest_file,
        explicit_path=package_evidence_path,
    )
    if package_evidence is None:
        warnings.append("smoke_run: package evidence not supplied")
    elif package_evidence.get("version") != SPECIALIST_PACKAGE_VERSION:
        issues.append(f"smoke_run: unsupported package evidence version {package_evidence.get('version')!r}")
    elif package_evidence.get("ok") is not True:
        issues.append("smoke_run: package evidence is not ok")

    resolved_model_id = model_id or _default_model_id(manifest, mode)
    provider_check = check_provider_for_model(
        resolved_model_id,
        required=bool(require_provider or mode == "dummy"),
    )
    provider_payload = asdict(provider_check)
    generated_text = ""
    skipped = bool(provider_check.skipped)
    if provider_check.ok and mode == "dummy":
        provider = get_provider_for_model(resolved_model_id)
        result = provider.generate(
            prompt="SignalForge AI specialist smoke run",
            model_id=resolved_model_id,
            task_type="smoke",
        )
        generated_text = result.text[:200]
    elif not provider_check.ok and not provider_check.skipped:
        issues.append(provider_check.reason or f"provider unavailable: {resolved_model_id}")

    payload = {
        "version": SPECIALIST_SMOKE_VERSION,
        "ok": not issues,
        "created_at": _utc_now(),
        "manifest_path": str(manifest_file),
        "artifacts_root": str(artifacts_root) if artifacts_root is not None else None,
        "mode": mode,
        "model_id": resolved_model_id,
        "provider": provider_payload,
        "skipped": skipped,
        "package_evidence_path": _package_evidence_ref(manifest, package_evidence_path),
        "generated_text_preview": generated_text,
        "issues": issues,
        "warnings": warnings,
    }
    json_path = out_dir / "specialist_smoke.json"
    md_path = out_dir / "specialist_smoke.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_format_smoke_markdown(payload), encoding="utf-8")
    payload["report_json"] = str(json_path)
    payload["report_md"] = str(md_path)
    if update_manifest:
        _attach_smoke_evidence(manifest_file, json_path, payload)
    return payload


def _default_model_id(manifest: Mapping[str, Any], mode: str) -> str:
    if mode == "dummy":
        return "dummy_good"
    artifacts = _dict(manifest.get("artifacts"))
    if mode == "ollama":
        tag = str(_dict(artifacts.get("ollama")).get("tag") or "")
        return tag if tag.startswith("ollama:") else f"ollama:{tag or 'unknown'}"
    base = str(_dict(manifest.get("model")).get("base_model") or "")
    return base if base.startswith("hf:") else f"hf:{base or 'unknown'}"


def _load_package_evidence(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    explicit_path: str | Path | None,
) -> Optional[Dict[str, Any]]:
    embedded_path = _dict(manifest.get("package_evidence")).get("path")
    path_value = explicit_path or embedded_path
    if not path_value:
        return None
    path = Path(path_value)
    if explicit_path is None and not path.is_absolute():
        path = manifest_path.parent / path
    if not path.exists():
        return {"version": SPECIALIST_PACKAGE_VERSION, "ok": False, "issues": [f"package evidence missing: {path}"]}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"version": "", "ok": False}


def _package_evidence_ref(
    manifest: Mapping[str, Any],
    explicit_path: str | Path | None,
) -> Optional[str]:
    if explicit_path is not None:
        return str(explicit_path)
    value = _dict(manifest.get("package_evidence")).get("path")
    return str(value) if value else None


def _attach_smoke_evidence(manifest_path: Path, smoke_path: Path, payload: Mapping[str, Any]) -> None:
    manifest = _load_manifest(manifest_path)
    manifest["smoke_run_evidence"] = {
        "version": SPECIALIST_SMOKE_VERSION,
        "path": str(smoke_path.resolve()),
        "ok": bool(payload.get("ok")),
        "mode": str(payload.get("mode") or ""),
        "model_id": str(payload.get("model_id") or ""),
        "issues": list(payload.get("issues", [])) if isinstance(payload.get("issues"), list) else [],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _format_smoke_markdown(payload: Mapping[str, Any]) -> str:
    provider = _dict(payload.get("provider"))
    lines = [
        "# SignalForge AI Specialist Smoke Run",
        "",
        f"- OK: `{str(payload.get('ok')).lower()}`",
        f"- Mode: `{payload.get('mode')}`",
        f"- Model: `{payload.get('model_id')}`",
        f"- Provider: `{provider.get('provider')}`",
        f"- Skipped: `{str(payload.get('skipped')).lower()}`",
    ]
    if payload.get("issues"):
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in payload["issues"])
    if payload.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in payload["warnings"])
    return "\n".join(lines).rstrip() + "\n"


def _load_manifest(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return payload


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
