"""Regression comparison for production-pilot readiness artifacts."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tensorfoundry.evaluation.diagnosis import failure_mode_severity, normalize_failure_mode

CaseKey = Tuple[str, str, str]


@dataclass(frozen=True)
class RegressionPolicy:
    """Policy thresholds for readiness regression checks."""

    max_pass_rate_drop: float = 0.0
    max_mean_score_drop: float = 0.0
    allow_new_failing_cases: bool = False
    allow_worse_failure_modes: bool = False


def load_pilot_readiness_artifact(path: Path | str) -> Dict[str, Any]:
    """Load pilot_readiness.json from a file path or containing directory."""
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / "pilot_readiness.json"
    with resolved.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"pilot readiness artifact must be a JSON object: {resolved}")
    return data


def compare_pilot_readiness(
    baseline: Dict[str, Any],
    current: Dict[str, Any],
    *,
    policy: Optional[RegressionPolicy] = None,
) -> Dict[str, Any]:
    """Compare two pilot readiness payloads and return an eval_regression.v0 payload."""
    policy = policy or RegressionPolicy()
    baseline_cases = _flatten_cases(baseline)
    current_cases = _flatten_cases(current)
    baseline_summary = _summarize_cases(baseline_cases)
    current_summary = _summarize_cases(current_cases)

    baseline_keys = set(baseline_cases)
    current_keys = set(current_cases)
    common_keys = sorted(baseline_keys & current_keys)
    new_keys = sorted(current_keys - baseline_keys)
    missing_keys = sorted(baseline_keys - current_keys)

    changed_cases: List[Dict[str, Any]] = []
    regressed_cases: List[Dict[str, Any]] = []
    improved_cases: List[Dict[str, Any]] = []
    movements: List[Dict[str, Any]] = []
    worse_movements: List[Dict[str, Any]] = []

    for key in common_keys:
        before = baseline_cases[key]
        after = current_cases[key]
        score_delta = round(float(after["score"]) - float(before["score"]), 10)
        before_passed = bool(before["passed"])
        after_passed = bool(after["passed"])
        before_mode = normalize_failure_mode(before.get("failure_mode"), passed=before_passed)
        after_mode = normalize_failure_mode(after.get("failure_mode"), passed=after_passed)
        before_severity = failure_mode_severity(before_mode, passed=before_passed)
        after_severity = failure_mode_severity(after_mode, passed=after_passed)
        severity_delta = after_severity - before_severity

        changed = (
            before_passed != after_passed
            or score_delta != 0.0
            or before_mode != after_mode
        )
        if changed:
            changed_cases.append(_case_delta(key, before, after, score_delta))
        if before_passed and not after_passed or score_delta < 0.0 or severity_delta > 0:
            regressed_cases.append(_case_delta(key, before, after, score_delta))
        if not before_passed and after_passed or score_delta > 0.0 or severity_delta < 0:
            improved_cases.append(_case_delta(key, before, after, score_delta))
        if before_mode != after_mode or severity_delta != 0:
            movement = {
                **_case_key_dict(key),
                "baseline_failure_mode": before_mode,
                "current_failure_mode": after_mode,
                "baseline_severity": before_severity,
                "current_severity": after_severity,
                "severity_delta": severity_delta,
            }
            movements.append(movement)
            if severity_delta > 0:
                worse_movements.append(movement)

    new_cases = [_case_payload(key, current_cases[key]) for key in new_keys]
    missing_cases = [_case_payload(key, baseline_cases[key]) for key in missing_keys]
    new_failing_cases = [c for c in new_cases if not bool(c["passed"])]

    pass_rate_delta = round(
        float(current_summary["pass_rate"]) - float(baseline_summary["pass_rate"]),
        10,
    )
    mean_score_delta = round(
        float(current_summary["mean_score"]) - float(baseline_summary["mean_score"]),
        10,
    )
    pass_rate_drop = max(0.0, -pass_rate_delta)
    mean_score_drop = max(0.0, -mean_score_delta)

    issues: List[str] = []
    if pass_rate_drop > policy.max_pass_rate_drop:
        issues.append(
            "pass_rate_drop "
            f"{pass_rate_drop:.6f} exceeds policy {policy.max_pass_rate_drop:.6f}"
        )
    if mean_score_drop > policy.max_mean_score_drop:
        issues.append(
            "mean_score_drop "
            f"{mean_score_drop:.6f} exceeds policy {policy.max_mean_score_drop:.6f}"
        )
    if missing_cases:
        issues.append(f"{len(missing_cases)} baseline case(s) missing from current run")
    if new_failing_cases and not policy.allow_new_failing_cases:
        issues.append(f"{len(new_failing_cases)} new failing case(s) found")
    if worse_movements and not policy.allow_worse_failure_modes:
        issues.append(f"{len(worse_movements)} worse failure-mode movement(s) found")

    return {
        "version": "eval_regression.v0",
        "ok": not issues,
        "policy": asdict(policy),
        "baseline": _artifact_meta(baseline),
        "current": _artifact_meta(current),
        "baseline_summary": baseline_summary,
        "current_summary": current_summary,
        "aggregate_deltas": {
            "pass_rate_delta": pass_rate_delta,
            "pass_rate_drop": pass_rate_drop,
            "mean_score_delta": mean_score_delta,
            "mean_score_drop": mean_score_drop,
        },
        "changed_cases": changed_cases,
        "regressed_cases": regressed_cases,
        "improved_cases": improved_cases,
        "new_cases": new_cases,
        "new_failing_cases": new_failing_cases,
        "missing_cases": missing_cases,
        "failure_mode_movements": movements,
        "worse_failure_mode_movements": worse_movements,
        "issues": issues,
    }


def write_regression_reports(
    payload: Dict[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> Dict[str, str]:
    """Write JSON and Markdown regression reports."""
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(format_regression_markdown(payload), encoding="utf-8")
    return {"report_json": str(json_path), "report_md": str(markdown_path)}


def format_regression_markdown(payload: Dict[str, Any]) -> str:
    """Render an eval_regression.v0 payload as Markdown."""
    deltas = payload["aggregate_deltas"]
    lines = [
        "# TensorFoundry Evaluation Regression",
        "",
        f"- OK: `{str(payload['ok']).lower()}`",
        f"- Pass-rate delta: `{deltas['pass_rate_delta']:+.2%}`",
        f"- Mean-score delta: `{deltas['mean_score_delta']:+.4f}`",
        f"- Changed cases: `{len(payload['changed_cases'])}`",
        f"- Regressed cases: `{len(payload['regressed_cases'])}`",
        f"- Improved cases: `{len(payload['improved_cases'])}`",
        f"- New cases: `{len(payload['new_cases'])}`",
        f"- Missing cases: `{len(payload['missing_cases'])}`",
        "",
    ]
    if payload["issues"]:
        lines.extend(["## Issues", ""])
        lines.extend(f"- {issue}" for issue in payload["issues"])
        lines.append("")

    _append_case_table(lines, "Regressed Cases", payload["regressed_cases"])
    _append_case_table(lines, "Improved Cases", payload["improved_cases"])
    _append_case_table(lines, "New Failing Cases", payload["new_failing_cases"])
    _append_movement_table(lines, payload["worse_failure_mode_movements"])
    return "\n".join(lines).rstrip() + "\n"


def _flatten_cases(payload: Dict[str, Any]) -> Dict[CaseKey, Dict[str, Any]]:
    suites = payload.get("suites", [])
    out: Dict[CaseKey, Dict[str, Any]] = {}
    if not isinstance(suites, list):
        return out
    for suite in suites:
        if not isinstance(suite, dict):
            continue
        suite_name = str(suite.get("suite_name", "unknown"))
        model_id = str(suite.get("model_id") or "unknown")
        results = suite.get("results", [])
        if not isinstance(results, list):
            continue
        for case in results:
            if not isinstance(case, dict):
                continue
            case_id = str(case.get("case_id", "unknown"))
            key = (suite_name, model_id, case_id)
            passed = bool(case.get("passed"))
            out[key] = {
                "suite_name": suite_name,
                "model_id": model_id,
                "case_id": case_id,
                "passed": passed,
                "score": _as_float(case.get("score")),
                "failure_mode": normalize_failure_mode(case.get("failure_mode"), passed=passed),
                "terminal_status": case.get("terminal_status"),
                "terminal_reason": case.get("terminal_reason"),
                "trace_path": case.get("trace_path")
                or _artifact_refs(case).get("trace")
                or "",
                "artifact_refs": _artifact_refs(case),
            }
    return out


def _summarize_cases(cases: Dict[CaseKey, Dict[str, Any]]) -> Dict[str, Any]:
    total = len(cases)
    passed = sum(1 for c in cases.values() if c["passed"])
    failed = total - passed
    mean_score = sum(float(c["score"]) for c in cases.values()) / total if total else 0.0
    modes = Counter(str(c["failure_mode"]) for c in cases.values() if not c["passed"])
    return {
        "cases": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / total if total else 0.0,
        "mean_score": mean_score,
        "failure_modes": dict(sorted(modes.items())),
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
        "current": _case_state(after),
        "score_delta": score_delta,
    }


def _case_payload(key: CaseKey, case: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **_case_key_dict(key),
        **_case_state(case),
    }


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
    suite_name, model_id, case_id = key
    return {"suite_name": suite_name, "model_id": model_id, "case_id": case_id}


def _artifact_refs(case: Dict[str, Any]) -> Dict[str, str]:
    refs = case.get("artifact_refs")
    if not isinstance(refs, dict):
        return {}
    return {str(k): str(v) for k, v in refs.items()}


def _artifact_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "version": payload.get("version"),
        "ok": payload.get("ok"),
        "work_dir": payload.get("work_dir"),
        "report_json": payload.get("report_json"),
        "report_md": payload.get("report_md"),
    }


def _append_case_table(lines: List[str], title: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    lines.extend(
        [
            f"## {title}",
            "",
            "| suite | model | case | baseline | current | score_delta | trace |",
            "|---|---|---|---|---|---:|---|",
        ]
    )
    for row in rows[:20]:
        baseline = row.get("baseline", {})
        current = row.get("current", row)
        lines.append(
            f"| {row['suite_name']} | `{row['model_id']}` | `{row['case_id']}` | "
            f"{_fmt_case_status(baseline)} | {_fmt_case_status(current)} | "
            f"{float(row.get('score_delta', 0.0)):+.4f} | `{current.get('trace_path', '')}` |"
        )
    if len(rows) > 20:
        lines.append(f"| ... | ... | ... | ... | ... | ... | {len(rows) - 20} more |")
    lines.append("")


def _append_movement_table(lines: List[str], rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    lines.extend(
        [
            "## Worse Failure-Mode Movements",
            "",
            "| suite | model | case | baseline | current | severity_delta |",
            "|---|---|---|---|---|---:|",
        ]
    )
    for row in rows[:20]:
        lines.append(
            f"| {row['suite_name']} | `{row['model_id']}` | `{row['case_id']}` | "
            f"{row['baseline_failure_mode']} | {row['current_failure_mode']} | "
            f"{row['severity_delta']:+d} |"
        )
    if len(rows) > 20:
        lines.append(f"| ... | ... | ... | ... | ... | {len(rows) - 20} more |")
    lines.append("")


def _fmt_case_status(row: Dict[str, Any]) -> str:
    if not row:
        return ""
    passed = "pass" if bool(row.get("passed")) else "fail"
    return f"{passed}/{float(row.get('score', 0.0)):.3f}/{row.get('failure_mode', 'none')}"


def _as_float(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0
