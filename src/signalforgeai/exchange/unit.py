"""Build specialist model unit manifests from release evidence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from signalforgeai.distillation.check import DISTILLATION_EVAL_VERSION
from signalforgeai.evaluation.matrix import BENCHMARK_MATRIX_VERSION
from signalforgeai.exchange.validation import (
    SPECIALIST_MODEL_UNIT_VERSION,
    is_uri_ref,
    parse_checksum_args,
    sha256_file,
    validate_manifest,
)
from signalforgeai.training.readiness import TRAINING_PREFLIGHT_VERSION, TRAINING_RUN_VERSION


def build_specialist_unit(
    *,
    training_preflight_path: str | Path,
    distillation_eval_path: str | Path,
    benchmark_matrix_path: str | Path,
    out_path: str | Path,
    training_run_path: str | Path | None = None,
    unit_id: str,
    name: str,
    version: str,
    domain: str,
    model_family: str,
    model_size: str,
    model_format: str,
    model_license: str,
    dataset_license: str,
    usage_constraints: Iterable[str],
    failure_mode: str,
    failure_description: str,
    failure_mitigation: str,
    failure_severity: Optional[str] = None,
    base_model: Optional[str] = None,
    quantization: Optional[str] = None,
    summary: Optional[str] = None,
    adapter_refs: Optional[Iterable[str]] = None,
    safetensors_refs: Optional[Iterable[str]] = None,
    gguf_refs: Optional[Iterable[str]] = None,
    ollama_modelfile: str = "",
    ollama_tag: str = "",
    checksum_args: Optional[Iterable[str]] = None,
    artifacts_root: str | Path | None = None,
    hardware_target: str = "cpu",
    ram_gb: float = 0.0,
    vram_gb: Optional[float] = None,
    throughput_tokens_per_s: Optional[float] = None,
    release_ready: bool = False,
) -> Dict[str, Any]:
    """Build, write, and validate a `specialist_model_unit.v0` manifest."""
    training_path = Path(training_preflight_path).resolve()
    distill_path = Path(distillation_eval_path).resolve()
    benchmark_path = Path(benchmark_matrix_path).resolve()
    run_path = Path(training_run_path).resolve() if training_run_path is not None else None
    out = Path(out_path)
    training = _load_evidence(training_path, TRAINING_PREFLIGHT_VERSION)
    training_run = _load_evidence(run_path, TRAINING_RUN_VERSION) if run_path is not None else {}
    distill = _load_evidence(distill_path, DISTILLATION_EVAL_VERSION)
    benchmark = _load_evidence(benchmark_path, BENCHMARK_MATRIX_VERSION)
    usage_constraints_list = [str(value) for value in usage_constraints if str(value)]
    if not usage_constraints_list:
        raise ValueError("at least one usage constraint is required")

    training_artifact = _dict(training.get("artifact_manifest"))
    candidate = _dict(distill.get("candidate"))
    recipe = _dict(distill.get("recipe"))
    benchmark_row = _matching_benchmark_row(
        benchmark,
        suite_name=str(candidate.get("suite_name") or ""),
        model_id=str(recipe.get("candidate_model_id") or candidate.get("model_id") or ""),
    )
    benchmark_metrics = _dict(benchmark_row.get("metrics") if benchmark_row else {})

    run_artifacts = _training_run_artifacts(training_run)
    checksums = {str(k): str(v) for k, v in _dict(training_run.get("file_checksums")).items()}
    checksums.update(parse_checksum_args(checksum_args or []))
    artifact_refs: Dict[str, Any] = {
        "adapters": _explicit_or_training_refs(adapter_refs, run_artifacts, "adapters"),
        "safetensors": _explicit_or_training_refs(safetensors_refs, run_artifacts, "safetensors"),
        "gguf": _explicit_or_training_refs(gguf_refs, run_artifacts, "gguf"),
    }
    artifact_refs["ollama"] = {
        "modelfile": str(ollama_modelfile or ""),
        "tag": str(ollama_tag or ""),
    }
    checksums.update(_compute_missing_local_checksums(artifact_refs, checksums, artifacts_root))

    latency_p50 = _number(benchmark_metrics.get("latency_ms_p50"), default=0.0)
    latency_p95 = _optional_number(benchmark_metrics.get("latency_ms_p95"))
    score = _clamp01(_number(candidate.get("mean_score"), default=0.0))
    reliability = _clamp01(_number(candidate.get("pass_rate"), default=0.0))
    cost_per_success = _optional_number(benchmark_metrics.get("cost_per_success_usd"))
    dataset_sources = _list_from_mapping_or_rows(
        training_artifact.get("dataset_sources"),
        training_artifact.get("datasets"),
        "path",
    )
    trace_sources = _trace_sources(training, training_artifact, distill, benchmark)
    failure = {
        "id": failure_mode,
        "description": failure_description,
        "mitigation": failure_mitigation,
    }
    if failure_severity:
        failure["severity"] = failure_severity

    model = {
        "family": model_family,
        "size": model_size,
        "format": model_format,
    }
    resolved_base = base_model or str(training_artifact.get("base_model") or training.get("base_model") or "")
    if resolved_base:
        model["base_model"] = resolved_base
    if quantization:
        model["quantization"] = quantization

    manifest: Dict[str, Any] = {
        "schema_version": SPECIALIST_MODEL_UNIT_VERSION,
        "id": unit_id,
        "name": name,
        "version": version,
        "domain": domain,
        "model": model,
        "eval_pack": {
            "suite_id": str(candidate.get("suite_name") or recipe.get("suite") or "unknown"),
            "report_path": _distillation_report_path(distill_path),
            "score": score,
            "reliability": reliability,
            "latency_ms_p50": latency_p50,
            "benchmark_run_ids": _benchmark_run_ids(benchmark_row),
            "distillation_eval_path": str(distill_path),
            "benchmark_matrix_path": str(benchmark_path),
        },
        "lineage": {
            "trace_sources": trace_sources,
            "dataset_sources": dataset_sources,
            "distillation_recipe": str(recipe.get("source_path") or recipe.get("training_preflight_path") or ""),
            "training_preflight": str(training_path),
            "dataset_hashes": _dict(training_artifact.get("dataset_hashes")),
            "split_counts": _dict(training_artifact.get("split_counts")),
        },
        "hardware_profile": {
            "target": hardware_target,
            "ram_gb": float(ram_gb),
            "latency_ms_p50": latency_p50,
        },
        "failure_modes": [failure],
        "license": {
            "model_license": model_license,
            "dataset_license": dataset_license,
        },
        "usage_constraints": usage_constraints_list,
        "artifacts": {
            **artifact_refs,
            "checksums": checksums,
        },
    }
    if summary:
        manifest["summary"] = summary
    if cost_per_success is not None:
        manifest["eval_pack"]["cost_per_success_usd"] = cost_per_success
    if latency_p95 is not None:
        manifest["hardware_profile"]["latency_ms_p95"] = latency_p95
    if vram_gb is not None:
        manifest["hardware_profile"]["vram_gb"] = float(vram_gb)
    if throughput_tokens_per_s is not None:
        manifest["hardware_profile"]["throughput_tokens_per_s"] = float(throughput_tokens_per_s)
    if run_path is not None:
        manifest["training_run_evidence"] = {
            "version": TRAINING_RUN_VERSION,
            "path": str(run_path),
            "ok": bool(training_run.get("ok")),
            "status": str(training_run.get("status") or ""),
            "artifact_count": len(training_run.get("output_artifacts", []))
            if isinstance(training_run.get("output_artifacts"), list)
            else 0,
            "issues": list(training_run.get("issues", []))
            if isinstance(training_run.get("issues"), list)
            else [],
        }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report = validate_manifest(out, artifacts_root=artifacts_root, release_ready=release_ready)
    if not report.ok:
        raise ValueError("; ".join(report.issues))
    return manifest


def _load_evidence(path: Path, expected_version: str) -> Dict[str, Any]:
    if not path.exists():
        raise ValueError(f"evidence not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evidence must be a JSON object: {path}")
    if payload.get("version") != expected_version:
        raise ValueError(
            f"unsupported evidence version for {path}: {payload.get('version')!r}; "
            f"expected {expected_version!r}"
        )
    if payload.get("ok") is not True:
        raise ValueError(f"evidence is not ok: {path}")
    return payload


def _training_run_artifacts(training_run: Mapping[str, Any]) -> Dict[str, Any]:
    refs = training_run.get("artifact_refs")
    return dict(refs) if isinstance(refs, dict) else {}


def _explicit_or_training_refs(
    values: Optional[Iterable[str]],
    training_artifacts: Mapping[str, Any],
    key: str,
) -> List[str]:
    explicit = [str(value) for value in values or [] if str(value)]
    if explicit:
        return explicit
    fallback = training_artifacts.get(key)
    if isinstance(fallback, list):
        return [str(value) for value in fallback if str(value)]
    return []


def _matching_benchmark_row(
    benchmark: Mapping[str, Any],
    *,
    suite_name: str,
    model_id: str,
) -> Dict[str, Any]:
    rows = benchmark.get("scorecard")
    if not isinstance(rows, list):
        return {}
    candidates = [row for row in rows if isinstance(row, dict) and row.get("model_id") == model_id]
    for row in candidates:
        if row.get("suite") == suite_name:
            return row
    return candidates[0] if candidates else {}


def _compute_missing_local_checksums(
    artifacts: Mapping[str, Any],
    checksums: Mapping[str, str],
    artifacts_root: str | Path | None,
) -> Dict[str, str]:
    base_dir = Path(artifacts_root) if artifacts_root is not None else Path.cwd()
    out: Dict[str, str] = {}
    for ref in _artifact_refs(artifacts):
        if ref in checksums or is_uri_ref(ref):
            continue
        path = Path(ref)
        if not path.is_absolute():
            path = base_dir / path
        if path.exists() and path.is_file():
            out[ref] = sha256_file(path)
    return out


def _artifact_refs(artifacts: Mapping[str, Any]) -> List[str]:
    refs: List[str] = []
    for key in ("adapters", "safetensors", "gguf"):
        values = artifacts.get(key)
        if isinstance(values, list):
            refs.extend(str(value) for value in values if str(value))
    ollama = artifacts.get("ollama")
    if isinstance(ollama, dict) and str(ollama.get("modelfile") or ""):
        refs.append(str(ollama["modelfile"]))
    return refs


def _list_from_mapping_or_rows(value: Any, rows: Any, row_key: str) -> List[str]:
    if isinstance(value, dict):
        out = [str(v) for v in value.values() if str(v)]
        if out:
            return out
    if isinstance(rows, list):
        out = [
            str(row.get(row_key))
            for row in rows
            if isinstance(row, dict) and str(row.get(row_key) or "")
        ]
        if out:
            return out
    return ["unknown"]


def _trace_sources(
    training: Mapping[str, Any],
    training_artifact: Mapping[str, Any],
    distill: Mapping[str, Any],
    benchmark: Mapping[str, Any],
) -> List[str]:
    refs: List[str] = []
    for value in (
        training_artifact.get("logs_root"),
        training.get("logs_root"),
        _dict(distill.get("artifact_refs")).get("baseline_logs"),
        _dict(distill.get("artifact_refs")).get("candidate_logs"),
        _dict(benchmark.get("artifact_refs")).get("logs_dir"),
    ):
        if str(value or ""):
            refs.append(str(value))
    return refs or ["unknown"]


def _distillation_report_path(distill_path: Path) -> str:
    sibling = distill_path.with_suffix(".md")
    return str(sibling if sibling.exists() else distill_path)


def _benchmark_run_ids(row: Mapping[str, Any]) -> List[str]:
    run_logs_dir = row.get("run_logs_dir")
    if str(run_logs_dir or ""):
        return [str(run_logs_dir)]
    return []


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _number(value: Any, *, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _optional_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
