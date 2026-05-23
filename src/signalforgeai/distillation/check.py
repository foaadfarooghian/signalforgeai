"""Distillation evaluation gate CLI and report generation."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from signalforgeai.distillation.recipe import (
    DistillationThresholds,
    apply_recipe_overrides,
    load_distillation_recipe,
    resolve_recipe_relative_path,
)
from signalforgeai.evaluation.diagnosis import failure_mode_severity, normalize_failure_mode
from signalforgeai.evaluation.harness import SuiteResult, run_suite
from signalforgeai.training.readiness import TRAINING_PREFLIGHT_VERSION


DISTILLATION_EVAL_VERSION = "distillation_eval.v0"
CaseKey = Tuple[str, str]


def run_distillation_check(
    *,
    recipe_path: Path,
    work_dir: Path,
    suite: Optional[str] = None,
    baseline_model_id: Optional[str] = None,
    candidate_model_id: Optional[str] = None,
    max_pass_rate_drop: Optional[float] = None,
    max_mean_score_drop: Optional[float] = None,
    allow_worse_failure_modes: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run a recipe-driven distillation eval gate and write reports."""
    recipe = apply_recipe_overrides(
        load_distillation_recipe(recipe_path),
        suite=suite,
        baseline_model_id=baseline_model_id,
        candidate_model_id=candidate_model_id,
        max_pass_rate_drop=max_pass_rate_drop,
        max_mean_score_drop=max_mean_score_drop,
        allow_worse_failure_modes=allow_worse_failure_modes,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    _reset_generated_outputs(work_dir)

    issues: List[str] = []
    try:
        training_preflight = load_training_preflight(
            resolve_recipe_relative_path(recipe, recipe.training_preflight_path)
        )
        training_summary = summarize_training_preflight(training_preflight)
    except ValueError as exc:
        training_preflight = None
        training_summary = {
            "ok": False,
            "path": recipe.training_preflight_path,
            "issues": [str(exc)],
        }
        issues.append(str(exc))

    baseline_suite: Optional[SuiteResult] = None
    candidate_suite: Optional[SuiteResult] = None
    comparison = empty_comparison(recipe.thresholds)
    if training_preflight is not None:
        baseline_suite = _run_model_suite(
            model_id=recipe.baseline_model_id,
            suite_path=recipe.suite,
            output_dir=work_dir / "results" / "baseline",
            logs_dir=work_dir / "logs" / "baseline",
        )
        candidate_suite = _run_model_suite(
            model_id=recipe.candidate_model_id,
            suite_path=recipe.suite,
            output_dir=work_dir / "results" / "candidate",
            logs_dir=work_dir / "logs" / "candidate",
        )
        comparison = compare_distillation_results(
            asdict(baseline_suite),
            asdict(candidate_suite),
            thresholds=recipe.thresholds,
        )
        issues.extend(str(issue) for issue in comparison["issues"])

    payload: Dict[str, Any] = {
        "version": DISTILLATION_EVAL_VERSION,
        "ok": bool(training_summary["ok"] and comparison["ok"] and not issues),
        "recipe": recipe.to_dict(),
        "work_dir": str(work_dir),
        "training_preflight": training_summary,
        "baseline": (
            summarize_suite(asdict(baseline_suite), work_dir=work_dir)
            if baseline_suite is not None
            else None
        ),
        "candidate": (
            summarize_suite(asdict(candidate_suite), work_dir=work_dir)
            if candidate_suite is not None
            else None
        ),
        "aggregate_deltas": comparison["aggregate_deltas"],
        "gate_results": comparison["gate_results"],
        "changed_cases": comparison["changed_cases"],
        "regressed_cases": comparison["regressed_cases"],
        "improved_cases": comparison["improved_cases"],
        "missing_cases": comparison["missing_cases"],
        "new_cases": comparison["new_cases"],
        "failure_mode_movements": comparison["failure_mode_movements"],
        "worse_failure_mode_movements": comparison["worse_failure_mode_movements"],
        "artifact_refs": _artifact_refs(work_dir),
        "issues": issues,
    }
    report_json = work_dir / "distillation_eval.json"
    report_md = work_dir / "distillation_eval.md"
    write_distillation_reports(payload, json_path=report_json, markdown_path=report_md)
    payload["report_json"] = str(report_json)
    payload["report_md"] = str(report_md)
    return payload


def load_training_preflight(path: Path) -> Dict[str, Any]:
    """Load and validate a `training_preflight.v0` report."""
    if not path.exists():
        raise ValueError(f"training preflight not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"training preflight must be an object: {path}")
    if payload.get("version") != TRAINING_PREFLIGHT_VERSION:
        raise ValueError(
            f"unsupported training preflight version: {payload.get('version')!r}"
        )
    if payload.get("ok") is not True:
        raise ValueError(f"training preflight is not ok: {path}")
    return payload


def summarize_training_preflight(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the training evidence fields needed by distillation reports."""
    artifact = payload.get("artifact_manifest")
    artifact_manifest = artifact if isinstance(artifact, dict) else {}
    datasets_raw = payload.get("datasets")
    datasets = datasets_raw if isinstance(datasets_raw, list) else []
    return {
        "version": payload.get("version"),
        "ok": bool(payload.get("ok")),
        "base_model": payload.get("base_model"),
        "quality_gate": bool(payload.get("quality_gate")),
        "datasets": [
            {
                "role": row.get("role"),
                "kind": row.get("kind"),
                "path": row.get("path"),
                "rows": row.get("rows"),
                "ok": row.get("ok"),
                "content_sha256": row.get("content_sha256"),
                "split_counts": row.get("split_counts"),
            }
            for row in datasets
            if isinstance(row, dict)
        ],
        "artifact_manifest": {
            "version": artifact_manifest.get("version"),
            "base_model": artifact_manifest.get("base_model"),
            "dataset_sources": artifact_manifest.get("dataset_sources", {}),
            "dataset_hashes": artifact_manifest.get("dataset_hashes", {}),
            "split_counts": artifact_manifest.get("split_counts", {}),
            "outputs": artifact_manifest.get("outputs", {}),
        },
    }


def compare_distillation_results(
    baseline_suite: Dict[str, Any],
    candidate_suite: Dict[str, Any],
    *,
    thresholds: DistillationThresholds,
) -> Dict[str, Any]:
    """Compare baseline and candidate suite results by suite/case identity."""
    baseline_cases = _flatten_cases(baseline_suite)
    candidate_cases = _flatten_cases(candidate_suite)
    baseline_summary = _summarize_cases(baseline_cases)
    candidate_summary = _summarize_cases(candidate_cases)
    baseline_summary.update(_reward_metrics(Path(str(baseline_suite.get("run_logs_dir", "")))))
    candidate_summary.update(_reward_metrics(Path(str(candidate_suite.get("run_logs_dir", "")))))

    common = sorted(set(baseline_cases) & set(candidate_cases))
    missing = sorted(set(baseline_cases) - set(candidate_cases))
    new = sorted(set(candidate_cases) - set(baseline_cases))

    changed_cases: List[Dict[str, Any]] = []
    regressed_cases: List[Dict[str, Any]] = []
    improved_cases: List[Dict[str, Any]] = []
    movements: List[Dict[str, Any]] = []
    worse_movements: List[Dict[str, Any]] = []

    for key in common:
        before = baseline_cases[key]
        after = candidate_cases[key]
        score_delta = round(float(after["score"]) - float(before["score"]), 10)
        before_mode = normalize_failure_mode(before.get("failure_mode"), passed=before["passed"])
        after_mode = normalize_failure_mode(after.get("failure_mode"), passed=after["passed"])
        before_severity = failure_mode_severity(before_mode, passed=before["passed"])
        after_severity = failure_mode_severity(after_mode, passed=after["passed"])
        severity_delta = after_severity - before_severity

        changed = (
            bool(before["passed"]) != bool(after["passed"])
            or score_delta != 0.0
            or before_mode != after_mode
        )
        delta = _case_delta(key, before, after, score_delta)
        if changed:
            changed_cases.append(delta)
        if before["passed"] and not after["passed"] or score_delta < 0 or severity_delta > 0:
            regressed_cases.append(delta)
        if not before["passed"] and after["passed"] or score_delta > 0 or severity_delta < 0:
            improved_cases.append(delta)
        if before_mode != after_mode or severity_delta != 0:
            movement = {
                **_case_key_dict(key),
                "baseline_failure_mode": before_mode,
                "candidate_failure_mode": after_mode,
                "baseline_severity": before_severity,
                "candidate_severity": after_severity,
                "severity_delta": severity_delta,
            }
            movements.append(movement)
            if severity_delta > 0:
                worse_movements.append(movement)

    missing_cases = [_case_payload(key, baseline_cases[key]) for key in missing]
    new_cases = [_case_payload(key, candidate_cases[key]) for key in new]
    deltas = _aggregate_deltas(baseline_summary, candidate_summary)
    gate_results = _gate_results(
        thresholds=thresholds,
        deltas=deltas,
        missing_cases=missing_cases,
        worse_movements=worse_movements,
    )
    issues = [str(g["issue"]) for g in gate_results if not g["ok"] and g.get("issue")]
    return {
        "ok": not issues,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "aggregate_deltas": deltas,
        "gate_results": gate_results,
        "changed_cases": changed_cases,
        "regressed_cases": regressed_cases,
        "improved_cases": improved_cases,
        "missing_cases": missing_cases,
        "new_cases": new_cases,
        "failure_mode_movements": movements,
        "worse_failure_mode_movements": worse_movements,
        "issues": issues,
    }


def empty_comparison(thresholds: DistillationThresholds) -> Dict[str, Any]:
    """Return an empty comparison object for early input failures."""
    return {
        "ok": False,
        "aggregate_deltas": {},
        "gate_results": [
            {
                "name": "training_preflight",
                "ok": False,
                "skipped": False,
                "issue": "training preflight did not pass",
            }
        ],
        "changed_cases": [],
        "regressed_cases": [],
        "improved_cases": [],
        "missing_cases": [],
        "new_cases": [],
        "failure_mode_movements": [],
        "worse_failure_mode_movements": [],
        "issues": ["training preflight did not pass"],
        "thresholds": thresholds.to_dict(),
    }


def summarize_suite(suite: Dict[str, Any], *, work_dir: Path) -> Dict[str, Any]:
    """Summarize a suite result and its reward metrics for report embedding."""
    cases = _flatten_cases(suite)
    summary = _summarize_cases(cases)
    metrics = _reward_metrics(Path(str(suite.get("run_logs_dir", ""))))
    return {
        **summary,
        **metrics,
        "suite_name": suite.get("suite_name"),
        "agent": suite.get("agent"),
        "model_id": suite.get("model_id"),
        "run_id": suite.get("run_id"),
        "run_logs_dir": suite.get("run_logs_dir"),
        "work_dir": str(work_dir),
    }


def write_distillation_reports(
    payload: Dict[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    """Write JSON and Markdown distillation reports."""
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(format_distillation_markdown(payload), encoding="utf-8")


def format_distillation_markdown(payload: Dict[str, Any]) -> str:
    """Render a `distillation_eval.v0` payload as Markdown."""
    recipe_raw = payload.get("recipe")
    recipe: Dict[str, Any] = recipe_raw if isinstance(recipe_raw, dict) else {}
    deltas = payload.get("aggregate_deltas")
    delta = deltas if isinstance(deltas, dict) else {}
    lines = [
        "# SignalForge AI Distillation Eval",
        "",
        f"- OK: `{str(payload.get('ok')).lower()}`",
        f"- Recipe: `{recipe.get('id')}`",
        f"- Domain: `{recipe.get('domain')}`",
        f"- Suite: `{recipe.get('suite')}`",
        f"- Baseline: `{recipe.get('baseline_model_id')}`",
        f"- Candidate: `{recipe.get('candidate_model_id')}`",
        f"- Pass-rate delta: `{float(delta.get('pass_rate_delta', 0.0)):+.2%}`",
        f"- Mean-score delta: `{float(delta.get('mean_score_delta', 0.0)):+.4f}`",
        "",
        "## Gates",
        "",
        "| gate | ok | value | threshold | skipped |",
        "|---|---:|---:|---:|---:|",
    ]
    for gate in payload.get("gate_results", []):
        if not isinstance(gate, dict):
            continue
        lines.append(
            f"| {gate.get('name')} | {str(gate.get('ok')).lower()} | "
            f"{_fmt_value(gate.get('value'))} | {_fmt_value(gate.get('threshold'))} | "
            f"{str(gate.get('skipped', False)).lower()} |"
        )

    _append_case_table(lines, "Regressed Cases", payload.get("regressed_cases", []))
    _append_case_table(lines, "Improved Cases", payload.get("improved_cases", []))
    _append_movement_table(lines, payload.get("worse_failure_mode_movements", []))

    if payload.get("issues"):
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in payload["issues"])
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a SignalForge AI distillation eval gate.")
    parser.add_argument("--recipe", type=str, required=True, help="distillation_recipe.v0 path")
    parser.add_argument("--work-dir", type=str, required=True, help="Output work directory")
    parser.add_argument("--suite", type=str, default="", help="Override recipe suite path")
    parser.add_argument("--baseline-model-id", type=str, default="", help="Override baseline model id")
    parser.add_argument("--candidate-model-id", type=str, default="", help="Override candidate model id")
    parser.add_argument("--max-pass-rate-drop", type=float, default=None)
    parser.add_argument("--max-mean-score-drop", type=float, default=None)
    parser.add_argument("--allow-worse-failure-modes", action="store_true")
    args = parser.parse_args(argv)

    try:
        payload = run_distillation_check(
            recipe_path=Path(args.recipe),
            work_dir=Path(args.work_dir),
            suite=args.suite or None,
            baseline_model_id=args.baseline_model_id or None,
            candidate_model_id=args.candidate_model_id or None,
            max_pass_rate_drop=args.max_pass_rate_drop,
            max_mean_score_drop=args.max_mean_score_drop,
            allow_worse_failure_modes=True if args.allow_worse_failure_modes else None,
        )
    except ValueError as exc:
        print(f"Distillation eval: FAILED ({exc})")
        return 2
    print(f"Distillation eval: {'OK' if payload['ok'] else 'FAILED'}")
    print(f"Report: {payload['report_md']}")
    return 0 if payload["ok"] else 1


def _run_model_suite(
    *,
    model_id: str,
    suite_path: str,
    output_dir: Path,
    logs_dir: Path,
) -> SuiteResult:
    old_model = os.environ.get("SIGNALFORGEAI_MODEL_ID")
    old_provider = os.environ.get("SIGNALFORGEAI_PROVIDER")
    try:
        os.environ["SIGNALFORGEAI_MODEL_ID"] = model_id
        if model_id.startswith("dummy"):
            os.environ["SIGNALFORGEAI_PROVIDER"] = "dummy"
        result = run_suite(
            suite_path=suite_path,
            output_dir=output_dir,
            logs_dir=logs_dir,
        )
        return result
    finally:
        if old_model is None:
            os.environ.pop("SIGNALFORGEAI_MODEL_ID", None)
        else:
            os.environ["SIGNALFORGEAI_MODEL_ID"] = old_model
        if old_provider is None:
            os.environ.pop("SIGNALFORGEAI_PROVIDER", None)
        else:
            os.environ["SIGNALFORGEAI_PROVIDER"] = old_provider


def _reset_generated_outputs(work_dir: Path) -> None:
    for name in ("logs", "results"):
        path = work_dir / name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
    for name in ("distillation_eval.json", "distillation_eval.md"):
        path = work_dir / name
        if path.exists() or path.is_symlink():
            path.unlink()


def _flatten_cases(suite: Dict[str, Any]) -> Dict[CaseKey, Dict[str, Any]]:
    suite_name = str(suite.get("suite_name", "unknown"))
    out: Dict[CaseKey, Dict[str, Any]] = {}
    results = suite.get("results", [])
    if not isinstance(results, list):
        return out
    for row in results:
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("case_id", "unknown"))
        passed = bool(row.get("passed"))
        out[(suite_name, case_id)] = {
            "suite_name": suite_name,
            "case_id": case_id,
            "passed": passed,
            "score": _as_float(row.get("score")),
            "failure_mode": normalize_failure_mode(row.get("failure_mode"), passed=passed),
            "terminal_status": row.get("terminal_status"),
            "terminal_reason": row.get("terminal_reason"),
            "trace_path": row.get("trace_path") or _row_artifact_refs(row).get("trace") or "",
            "artifact_refs": _row_artifact_refs(row),
        }
    return out


def _summarize_cases(cases: Dict[CaseKey, Dict[str, Any]]) -> Dict[str, Any]:
    total = len(cases)
    passed = sum(1 for row in cases.values() if bool(row["passed"]))
    failed = total - passed
    mean_score = sum(float(row["score"]) for row in cases.values()) / total if total else 0.0
    return {
        "cases": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / total if total else 0.0,
        "mean_score": mean_score,
    }


def _aggregate_deltas(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    pass_rate_delta = round(float(candidate["pass_rate"]) - float(baseline["pass_rate"]), 10)
    mean_score_delta = round(float(candidate["mean_score"]) - float(baseline["mean_score"]), 10)
    return {
        "pass_rate_delta": pass_rate_delta,
        "pass_rate_drop": max(0.0, -pass_rate_delta),
        "mean_score_delta": mean_score_delta,
        "mean_score_drop": max(0.0, -mean_score_delta),
        "cost_per_success_improvement": _metric_improvement(
            baseline,
            candidate,
            "cost_per_success_usd",
        ),
        "latency_ms_p50_improvement": _metric_improvement(
            baseline,
            candidate,
            "latency_ms_p50",
        ),
    }


def _metric_improvement(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    key: str,
) -> Optional[float]:
    before = baseline.get(key)
    after = candidate.get(key)
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return None
    return round(float(before) - float(after), 10)


def _gate_results(
    *,
    thresholds: DistillationThresholds,
    deltas: Dict[str, Any],
    missing_cases: List[Dict[str, Any]],
    worse_movements: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    gates = [
        _gate(
            "pass_rate_drop",
            float(deltas["pass_rate_drop"]),
            thresholds.max_pass_rate_drop,
            float(deltas["pass_rate_drop"]) <= thresholds.max_pass_rate_drop,
        ),
        _gate(
            "mean_score_drop",
            float(deltas["mean_score_drop"]),
            thresholds.max_mean_score_drop,
            float(deltas["mean_score_drop"]) <= thresholds.max_mean_score_drop,
        ),
        _gate(
            "missing_cases",
            len(missing_cases),
            0,
            not missing_cases,
        ),
    ]
    if thresholds.allow_worse_failure_modes:
        gates.append(
            _gate(
                "worse_failure_modes",
                len(worse_movements),
                0,
                True,
                skipped=True,
            )
        )
    else:
        gates.append(
            _gate(
                "worse_failure_modes",
                len(worse_movements),
                0,
                not worse_movements,
            )
        )
    gates.append(
        _optional_improvement_gate(
            "cost_per_success_improvement",
            deltas.get("cost_per_success_improvement"),
            thresholds.min_cost_per_success_improvement,
        )
    )
    gates.append(
        _optional_improvement_gate(
            "latency_ms_p50_improvement",
            deltas.get("latency_ms_p50_improvement"),
            thresholds.min_latency_ms_p50_improvement,
        )
    )
    return gates


def _gate(
    name: str,
    value: Any,
    threshold: Any,
    ok: bool,
    *,
    skipped: bool = False,
) -> Dict[str, Any]:
    issue = None if ok else f"{name} {value} exceeds threshold {threshold}"
    return {
        "name": name,
        "ok": bool(ok),
        "value": value,
        "threshold": threshold,
        "skipped": skipped,
        "issue": issue,
    }


def _optional_improvement_gate(
    name: str,
    value: Any,
    threshold: Optional[float],
) -> Dict[str, Any]:
    if threshold is None:
        return _gate(name, value, None, True, skipped=True)
    if not isinstance(value, (int, float)):
        return {
            "name": name,
            "ok": True,
            "value": value,
            "threshold": threshold,
            "skipped": True,
            "issue": None,
            "reason": "metric unavailable",
        }
    ok = float(value) >= threshold
    return {
        "name": name,
        "ok": ok,
        "value": value,
        "threshold": threshold,
        "skipped": False,
        "issue": None if ok else f"{name} {value} is below threshold {threshold}",
    }


def _case_delta(
    key: CaseKey,
    before: Dict[str, Any],
    after: Dict[str, Any],
    score_delta: float,
) -> Dict[str, Any]:
    return {
        **_case_key_dict(key),
        "baseline": _case_state(before),
        "candidate": _case_state(after),
        "score_delta": score_delta,
    }


def _case_payload(key: CaseKey, case: Dict[str, Any]) -> Dict[str, Any]:
    return {**_case_key_dict(key), **_case_state(case)}


def _case_state(case: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "passed": bool(case.get("passed")),
        "score": _as_float(case.get("score")),
        "failure_mode": str(case.get("failure_mode", "none")),
        "terminal_status": case.get("terminal_status"),
        "terminal_reason": case.get("terminal_reason"),
        "trace_path": case.get("trace_path", ""),
    }


def _case_key_dict(key: CaseKey) -> Dict[str, str]:
    suite_name, case_id = key
    return {"suite_name": suite_name, "case_id": case_id}


def _reward_metrics(logs_dir: Path) -> Dict[str, Any]:
    if str(logs_dir) in {"", "."} or not logs_dir.exists():
        return {
            "mean_cost_usd": None,
            "cost_per_success_usd": None,
            "latency_ms_p50": None,
            "latency_ms_p95": None,
        }
    rows = list(_iter_reward_rows(logs_dir))
    costs: List[float] = []
    latencies: List[int] = []
    for row in rows:
        cost = row.get("cost_usd")
        if isinstance(cost, (int, float)):
            costs.append(float(cost))
        latency = row.get("latency_ms")
        if isinstance(latency, int):
            latencies.append(latency)
    successes = sum(1 for row in rows if row.get("success") is True)
    return {
        "mean_cost_usd": sum(costs) / len(costs) if costs else None,
        "cost_per_success_usd": sum(costs) / successes if costs and successes else None,
        "latency_ms_p50": _p50(latencies),
        "latency_ms_p95": _p95(latencies),
    }


def _iter_reward_rows(logs_dir: Path) -> Iterable[Dict[str, Any]]:
    for reward_file in sorted(logs_dir.rglob("reward.jsonl")):
        for line in reward_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict) and row.get("version") == "reward.v0":
                yield row


def _p50(values: List[int]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[len(ordered) // 2])


def _p95(values: List[int]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return float(ordered[index])


def _row_artifact_refs(row: Dict[str, Any]) -> Dict[str, str]:
    refs = row.get("artifact_refs")
    if not isinstance(refs, dict):
        return {}
    return {str(k): str(v) for k, v in refs.items()}


def _artifact_refs(work_dir: Path) -> Dict[str, str]:
    return {
        "baseline_logs": str(work_dir / "logs" / "baseline"),
        "candidate_logs": str(work_dir / "logs" / "candidate"),
        "baseline_results": str(work_dir / "results" / "baseline"),
        "candidate_results": str(work_dir / "results" / "candidate"),
    }


def _append_case_table(lines: List[str], title: str, rows: Any) -> None:
    if not isinstance(rows, list) or not rows:
        return
    lines.extend(
        [
            "",
            f"## {title}",
            "",
            "| suite | case | baseline | candidate | score_delta | trace |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for row in rows[:20]:
        if not isinstance(row, dict):
            continue
        baseline = row.get("baseline", {})
        candidate = row.get("candidate", row)
        lines.append(
            f"| {row.get('suite_name')} | `{row.get('case_id')}` | "
            f"{_fmt_case_status(baseline)} | {_fmt_case_status(candidate)} | "
            f"{float(row.get('score_delta', 0.0)):+.4f} | "
            f"`{candidate.get('trace_path', '')}` |"
        )


def _append_movement_table(lines: List[str], rows: Any) -> None:
    if not isinstance(rows, list) or not rows:
        return
    lines.extend(
        [
            "",
            "## Worse Failure-Mode Movements",
            "",
            "| suite | case | baseline | candidate | severity_delta |",
            "|---|---|---|---|---:|",
        ]
    )
    for row in rows[:20]:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"| {row.get('suite_name')} | `{row.get('case_id')}` | "
            f"{row.get('baseline_failure_mode')} | {row.get('candidate_failure_mode')} | "
            f"{int(row.get('severity_delta', 0))} |"
        )


def _fmt_case_status(case: Any) -> str:
    if not isinstance(case, dict):
        return ""
    return (
        f"{'pass' if bool(case.get('passed')) else 'fail'} "
        f"{float(case.get('score') or 0.0):.3f} "
        f"{case.get('failure_mode') or 'none'}"
    )


def _fmt_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _as_float(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _as_int(value: Any) -> Optional[int]:
    return int(value) if isinstance(value, int) else None


if __name__ == "__main__":
    raise SystemExit(main())
