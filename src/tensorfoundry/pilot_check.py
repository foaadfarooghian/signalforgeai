"""Production-pilot readiness check for TensorFoundry."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from tensorfoundry.evaluation.harness import run_suite
from tensorfoundry.export.dataset import export_sft
from tensorfoundry.export.preferences import export_preferences
from tensorfoundry.export.repairs import export_repairs
from tensorfoundry.export.validate import DatasetValidationResult, validate_dataset_jsonl, write_dataset_manifest
from tensorfoundry.learning.curriculum import export_curriculum
from tensorfoundry.logging.validate import validate_jsonl_file
from tensorfoundry.models.registry import ProviderCheck, check_provider_for_model


DEFAULT_SUITE = "src/tensorfoundry/evaluation/benchmarks/v0/suites/decision_v0.json"


def _iter_jsonl_files(root: Path) -> Iterable[Path]:
    return sorted(root.rglob("*.jsonl"))


def _validate_artifacts(logs_dir: Path) -> List[str]:
    issues: List[str] = []
    for path in _iter_jsonl_files(logs_dir):
        for issue in validate_jsonl_file(path):
            issues.append(f"{path}: {issue}")
    return issues


def _provider_checks(
    *,
    require_provider: Set[str],
    hosted_model_id: str,
    local_model_id: str,
) -> List[ProviderCheck]:
    checks: List[ProviderCheck] = [
        check_provider_for_model("dummy_good", required=True),
    ]
    include_hosted = bool(require_provider & {"hosted", "all"})
    include_local = bool(require_provider & {"local", "all"})
    if include_hosted:
        checks.append(check_provider_for_model(hosted_model_id, required=True))
    else:
        optional = check_provider_for_model(hosted_model_id, required=False)
        checks.append(optional)
    if include_local:
        checks.append(check_provider_for_model(local_model_id, required=True))
    else:
        optional = check_provider_for_model(local_model_id, required=False)
        checks.append(optional)
    return checks


def _as_dict(check: ProviderCheck) -> Dict[str, Any]:
    return asdict(check)


def _write_report(payload: Dict[str, Any], report_md: Path) -> None:
    lines = [
        "# TensorFoundry Pilot Readiness",
        "",
        f"- OK: `{str(payload['ok']).lower()}`",
        f"- Work dir: `{payload['work_dir']}`",
        f"- Suites run: `{len(payload['suites'])}`",
        f"- Dataset manifest: `{payload['dataset_manifest']}`",
        "",
        "## Providers",
        "",
        "| provider | model | ok | required | skipped | reason |",
        "|---|---|---:|---:|---:|---|",
    ]
    for c in payload["providers"]:
        lines.append(
            f"| {c['provider']} | `{c['model_id']}` | {str(c['ok']).lower()} | "
            f"{str(c['required']).lower()} | {str(c['skipped']).lower()} | {c.get('reason') or ''} |"
        )
    lines.extend(["", "## Datasets", "", "| kind | rows | ok | path |", "|---|---:|---:|---|"])
    for d in payload["datasets"]:
        lines.append(f"| {d['kind']} | {d['rows']} | {str(d['ok']).lower()} | `{d['path']}` |")
    if payload["issues"]:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in payload["issues"])
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pilot_check(
    *,
    work_dir: Path,
    suites: List[str],
    require_provider: Set[str],
    hosted_model_id: str,
    local_model_id: str,
) -> Dict[str, Any]:
    """Run the offline pilot loop and return a readiness payload."""
    work_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = work_dir / "logs"
    results_dir = work_dir / "results"
    datasets_dir = work_dir / "datasets"

    provider_checks = _provider_checks(
        require_provider=require_provider,
        hosted_model_id=hosted_model_id,
        local_model_id=local_model_id,
    )

    old_model = os.environ.get("TENSORFOUNDRY_MODEL_ID")
    old_provider = os.environ.get("TENSORFOUNDRY_PROVIDER")
    suite_results = []
    try:
        os.environ["TENSORFOUNDRY_PROVIDER"] = "dummy"
        for model_id in ("dummy_good", "dummy_bad"):
            os.environ["TENSORFOUNDRY_MODEL_ID"] = model_id
            for suite in suites:
                result = run_suite(suite_path=suite, output_dir=results_dir, logs_dir=logs_dir)
                suite_results.append(asdict(result))
    finally:
        if old_model is None:
            os.environ.pop("TENSORFOUNDRY_MODEL_ID", None)
        else:
            os.environ["TENSORFOUNDRY_MODEL_ID"] = old_model
        if old_provider is None:
            os.environ.pop("TENSORFOUNDRY_PROVIDER", None)
        else:
            os.environ["TENSORFOUNDRY_PROVIDER"] = old_provider

    issues = _validate_artifacts(logs_dir)

    sft_out = datasets_dir / "pilot.sft.jsonl"
    prefs_out = datasets_dir / "pilot.prefs.jsonl"
    repairs_out = datasets_dir / "pilot.repairs.jsonl"
    curriculum_out = datasets_dir / "pilot.curriculum.jsonl"
    export_sft(
        logs_root=logs_dir,
        out_path=sft_out,
        suite=None,
        min_score=0.7,
        success_only=True,
        limit=None,
        include_prefixes=[],
        exclude_prefixes=[],
        exclude_model_ids=set(),
        step="critic_check",
        use_step_prompt=False,
        min_response_chars=10,
    )
    export_preferences(
        logs_root=logs_dir,
        out_path=prefs_out,
        suite=None,
        min_score=0.0,
        success_only=False,
        include_prefixes=[],
        exclude_prefixes=[],
        exclude_model_ids=set(),
        lambda_cost=0.0,
        mu_latency=0.0,
        max_abs_score_gap=1.0,
        limit=None,
        prompt_normalize="none",
        prompt_source="auto",
        output_format="prefs",
        deterministic=True,
    )
    export_repairs(
        logs_root=logs_dir,
        out_path=repairs_out,
        suite=None,
        min_success_score=0.7,
        max_failure_score=0.8,
        include_prefixes=[],
        exclude_prefixes=[],
        exclude_model_ids=set(),
        limit=None,
        prompt_normalize="none",
        prompt_source="auto",
        output_format="repairs",
    )
    export_curriculum(
        logs_root=logs_dir,
        out_path=curriculum_out,
        suite=None,
        min_score=0.0,
        success_only=False,
        include_prefixes=[],
        exclude_prefixes=[],
        exclude_model_ids=set(),
        step="critic_check",
        use_step_prompt=False,
        min_response_chars=1,
        easy_min_score=0.8,
        escalation_max_score=0.5,
        buckets=None,
        limit=None,
    )

    dataset_results: List[DatasetValidationResult] = [
        validate_dataset_jsonl(sft_out, kind="sft"),
        validate_dataset_jsonl(prefs_out, kind="prefs"),
        validate_dataset_jsonl(repairs_out, kind="repairs"),
        validate_dataset_jsonl(curriculum_out, kind="curriculum"),
    ]
    manifest_path = write_dataset_manifest(dataset_results, datasets_dir / "manifest.json")
    provider_ok = all(c.ok or c.skipped for c in provider_checks)
    suites_ok = all(s["failed"] == 0 for s in suite_results)
    datasets_ok = all(r.ok for r in dataset_results)
    ok = provider_ok and suites_ok and datasets_ok and not issues
    payload: Dict[str, Any] = {
        "version": "pilot_readiness.v0",
        "ok": ok,
        "work_dir": str(work_dir),
        "logs_dir": str(logs_dir),
        "results_dir": str(results_dir),
        "dataset_manifest": str(manifest_path),
        "providers": [_as_dict(c) for c in provider_checks],
        "suites": suite_results,
        "datasets": [r.to_dict() for r in dataset_results],
        "issues": issues,
    }
    report_json = work_dir / "pilot_readiness.json"
    report_md = work_dir / "pilot_readiness.md"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(payload, report_md)
    payload["report_json"] = str(report_json)
    payload["report_md"] = str(report_md)
    return payload


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run TensorFoundry production-pilot readiness checks.")
    parser.add_argument("--mode", type=str, default="dummy", choices=["dummy"], help="Pilot mode")
    parser.add_argument("--work-dir", type=str, default="results/pilot_check", help="Output work directory")
    parser.add_argument("--suite", action="append", default=[], help="Suite path; may be repeated")
    parser.add_argument(
        "--require-provider",
        action="append",
        default=[],
        choices=["hosted", "local", "all"],
        help="Fail if the selected provider path is unavailable",
    )
    parser.add_argument(
        "--hosted-model-id",
        type=str,
        default=os.getenv("TENSORFOUNDRY_HOSTED_MODEL_ID", "openai:gpt-5-mini"),
    )
    parser.add_argument(
        "--local-model-id",
        type=str,
        default=os.getenv("TENSORFOUNDRY_LOCAL_MODEL_ID", "ollama:ministral-3:8b"),
    )
    args = parser.parse_args(argv)
    payload = run_pilot_check(
        work_dir=Path(args.work_dir),
        suites=args.suite or [DEFAULT_SUITE],
        require_provider=set(args.require_provider),
        hosted_model_id=args.hosted_model_id,
        local_model_id=args.local_model_id,
    )
    print(f"Pilot readiness: {'OK' if payload['ok'] else 'FAILED'}")
    print(f"Report: {payload['report_md']}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
