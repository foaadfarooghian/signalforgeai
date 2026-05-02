"""File-based specialist model exchange registry index generation."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from tensorfoundry.exchange.validation import SPECIALIST_MODEL_UNIT_VERSION, validate_manifest


SPECIALIST_REGISTRY_INDEX_VERSION = "specialist_registry_index.v0"


def build_registry_index(
    *,
    registry_dir: str | Path,
    out_path: str | Path,
    release_ready: bool = False,
) -> Dict[str, Any]:
    """Scan unit manifests and write a `specialist_registry_index.v0` file."""
    root = Path(registry_dir)
    out = Path(out_path)
    issues: List[str] = []
    units: List[Dict[str, Any]] = []
    seen: Dict[Tuple[str, str], str] = {}

    if not root.exists():
        issues.append(f"registry dir not found: {root}")
    else:
        for manifest_path in sorted(root.rglob("*.json")):
            if manifest_path.resolve() == out.resolve():
                continue
            payload = _try_load_json(manifest_path)
            if payload.get("schema_version") != SPECIALIST_MODEL_UNIT_VERSION:
                continue
            report = validate_manifest(manifest_path, release_ready=release_ready)
            if not report.ok:
                issues.extend(f"{manifest_path}: {issue}" for issue in report.issues)
            unit_id = str(payload.get("id") or "")
            version = str(payload.get("version") or "")
            key = (unit_id, version)
            if key in seen:
                issues.append(
                    f"duplicate specialist unit id/version: {unit_id}@{version} "
                    f"({seen[key]} and {manifest_path})"
                )
            else:
                seen[key] = str(manifest_path)
            units.append(_unit_index_entry(payload, manifest_path, report.ok))

    payload = {
        "version": SPECIALIST_REGISTRY_INDEX_VERSION,
        "ok": not issues,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "registry_dir": str(root),
        "release_ready": bool(release_ready),
        "units": sorted(units, key=lambda row: (row["id"], row["version"])),
        "issues": issues,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _unit_index_entry(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    valid: bool,
) -> Dict[str, Any]:
    eval_pack = _dict(manifest.get("eval_pack"))
    model = _dict(manifest.get("model"))
    artifacts = _dict(manifest.get("artifacts"))
    lineage = _dict(manifest.get("lineage"))
    training_run_evidence = _dict(manifest.get("training_run_evidence"))
    package_evidence = _dict(manifest.get("package_evidence"))
    smoke_evidence = _dict(manifest.get("smoke_run_evidence"))
    return {
        "id": manifest.get("id"),
        "version": manifest.get("version"),
        "name": manifest.get("name"),
        "domain": manifest.get("domain"),
        "schema_version": manifest.get("schema_version"),
        "manifest_path": str(manifest_path),
        "valid": bool(valid),
        "model": model,
        "score": eval_pack.get("score"),
        "reliability": eval_pack.get("reliability"),
        "latency_ms_p50": eval_pack.get("latency_ms_p50"),
        "cost_per_success_usd": eval_pack.get("cost_per_success_usd"),
        "artifact_formats": _artifact_formats(artifacts),
        "evidence": {
            "report_path": eval_pack.get("report_path"),
            "distillation_eval_path": eval_pack.get("distillation_eval_path"),
            "benchmark_matrix_path": eval_pack.get("benchmark_matrix_path"),
            "training_preflight": lineage.get("training_preflight"),
            "training_run_evidence": training_run_evidence.get("path"),
            "package_evidence": package_evidence.get("path"),
            "smoke_run_evidence": smoke_evidence.get("path"),
        },
        "training_run_evidence": training_run_evidence,
        "package_evidence": package_evidence,
        "smoke_run_evidence": smoke_evidence,
        "checksums": _dict(artifacts.get("checksums")),
    }


def _artifact_formats(artifacts: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for key in ("adapters", "safetensors", "gguf"):
        values = artifacts.get(key)
        if isinstance(values, list) and values:
            out.append(key)
    ollama = artifacts.get("ollama")
    if isinstance(ollama, dict) and str(ollama.get("tag") or ""):
        out.append("ollama")
    return out


def _try_load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}
