from __future__ import annotations

import json
from pathlib import Path

from tensorfoundry.evaluation.diagnosis import diagnose_trace_events
from tensorfoundry.evaluation.regression import (
    RegressionPolicy,
    compare_pilot_readiness,
    load_pilot_readiness_artifact,
)


def _case(
    case_id: str,
    *,
    passed: bool = True,
    score: float = 1.0,
    failure_mode: str | None = None,
) -> dict:
    return {
        "case_id": case_id,
        "passed": passed,
        "score": score,
        "trace_id": f"trace-{case_id}",
        "trace_path": f"logs/{case_id}.jsonl",
        "terminal_status": "success" if passed else "failure",
        "terminal_reason": None,
        "notes": [],
        "failure_mode": failure_mode,
        "diagnosis": {},
        "artifact_refs": {"trace": f"logs/{case_id}.jsonl"},
    }


def _payload(cases: list[dict], *, model_id: str | None = "dummy_good") -> dict:
    passed = sum(1 for c in cases if c["passed"])
    suite: dict = {
        "suite_name": "decision_v0",
        "agent": "decision_agent",
        "run_id": "run-1",
        "run_logs_dir": "logs/run-1",
        "num_cases": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "pass_rate": passed / len(cases) if cases else 0.0,
        "results": cases,
    }
    if model_id is not None:
        suite["model_id"] = model_id
    return {
        "version": "pilot_readiness.v0",
        "ok": True,
        "work_dir": "results/pilot_check",
        "suites": [suite],
    }


def test_regression_identical_artifacts_pass() -> None:
    baseline = _payload([_case("a"), _case("b", passed=False, score=0.2)])
    result = compare_pilot_readiness(baseline, baseline)

    assert result["version"] == "eval_regression.v0"
    assert result["ok"] is True
    assert result["changed_cases"] == []


def test_regression_pass_rate_drop_fails() -> None:
    baseline = _payload([_case("a"), _case("b")])
    current = _payload([_case("a"), _case("b", passed=False, score=0.0)])

    result = compare_pilot_readiness(baseline, current)

    assert result["ok"] is False
    assert any("pass_rate_drop" in issue for issue in result["issues"])


def test_regression_mean_score_drop_fails() -> None:
    baseline = _payload([_case("a", score=0.9)])
    current = _payload([_case("a", score=0.8)])

    result = compare_pilot_readiness(
        baseline,
        current,
        policy=RegressionPolicy(max_mean_score_drop=0.05),
    )

    assert result["ok"] is False
    assert any("mean_score_drop" in issue for issue in result["issues"])


def test_regression_new_failing_case_fails_by_default() -> None:
    baseline = _payload([_case("a")])
    current = _payload([_case("a"), _case("b", passed=False, score=0.0)])

    result = compare_pilot_readiness(baseline, current)

    assert result["ok"] is False
    assert len(result["new_failing_cases"]) == 1


def test_regression_missing_current_case_is_issue() -> None:
    baseline = _payload([_case("a"), _case("b")])
    current = _payload([_case("a")])

    result = compare_pilot_readiness(baseline, current)

    assert result["ok"] is False
    assert len(result["missing_cases"]) == 1
    assert any("missing" in issue for issue in result["issues"])


def test_regression_worse_failure_mode_fails_by_default() -> None:
    baseline = _payload(
        [_case("a", passed=False, score=0.0, failure_mode="expectation_failed")]
    )
    current = _payload([_case("a", passed=False, score=0.0, failure_mode="scoring_error")])

    result = compare_pilot_readiness(baseline, current)

    assert result["ok"] is False
    assert len(result["worse_failure_mode_movements"]) == 1


def test_regression_legacy_missing_model_id_compares_as_unknown() -> None:
    baseline = _payload([_case("a", score=0.9)], model_id=None)
    current = _payload([_case("a", score=0.8)], model_id=None)

    result = compare_pilot_readiness(baseline, current)

    assert result["changed_cases"][0]["model_id"] == "unknown"


def test_load_pilot_readiness_artifact_accepts_directory(tmp_path: Path) -> None:
    payload = _payload([_case("a")])
    path = tmp_path / "pilot_readiness.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_pilot_readiness_artifact(tmp_path)["version"] == "pilot_readiness.v0"


def test_trace_diagnosis_records_failure_origin_and_recovery() -> None:
    events = [
        {"event_type": "task_received", "stage": "system", "outcome": {}},
        {
            "event_type": "tool_error",
            "stage": "tool",
            "payload": {"tool_name": "search", "success": False, "error": "timeout"},
            "outcome": {"status": "failure", "reason": "timeout"},
        },
        {"event_type": "retry_requested", "stage": "critic", "outcome": {}},
        {"event_type": "task_failed", "stage": "system", "outcome": {"status": "failure"}},
    ]

    diagnosis = diagnose_trace_events(events)

    assert diagnosis["failure_origin"]["index"] == 1
    assert diagnosis["failure_origin"]["event_type"] == "tool_error"
    assert diagnosis["recovery"]["attempted"] is True
    assert diagnosis["recovery"]["events"][0]["event_type"] == "retry_requested"
