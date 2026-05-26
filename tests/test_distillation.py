from __future__ import annotations

import json
from pathlib import Path

import pytest

from signalforgeai.distillation.check import (
    compare_distillation_results,
    load_training_preflight,
    main as distill_main,
    run_distillation_check,
)
from signalforgeai.distillation.recipe import (
    DISTILLATION_RECIPE_VERSION,
    DistillationThresholds,
    apply_recipe_overrides,
    load_distillation_recipe,
)
from signalforgeai.pilot_check import DEFAULT_SUITE
from signalforgeai.training.readiness import TRAINING_PREFLIGHT_VERSION


def test_distillation_recipe_loader_accepts_json_and_yaml(tmp_path: Path) -> None:
    preflight = _write_preflight(tmp_path)
    json_recipe = _write_recipe(tmp_path, preflight)
    yaml_recipe = tmp_path / "recipe.yaml"
    yaml_recipe.write_text(
        "\n".join(
            [
                f"version: {DISTILLATION_RECIPE_VERSION}",
                "id: yaml-recipe",
                "domain: pilot",
                f"suite: {DEFAULT_SUITE}",
                "baseline_model_id: dummy_good",
                "candidate_model_id: dummy_good",
                f"training_preflight_path: {preflight}",
                "thresholds:",
                "  max_pass_rate_drop: 0.0",
                "  max_mean_score_drop: 0.0",
                "  allow_worse_failure_modes: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_distillation_recipe(json_recipe).id == "pilot-distill"
    assert load_distillation_recipe(yaml_recipe).id == "yaml-recipe"


def test_distillation_recipe_loader_rejects_missing_fields(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"version": DISTILLATION_RECIPE_VERSION}), encoding="utf-8")

    try:
        load_distillation_recipe(recipe)
    except ValueError as exc:
        assert "missing required recipe field" in str(exc)
    else:
        raise AssertionError("expected recipe validation failure")


def test_distillation_recipe_overrides_take_precedence(tmp_path: Path) -> None:
    preflight = _write_preflight(tmp_path)
    recipe = load_distillation_recipe(_write_recipe(tmp_path, preflight))

    updated = apply_recipe_overrides(
        recipe,
        suite="custom.json",
        baseline_model_id="dummy_base",
        candidate_model_id="dummy_candidate",
        max_pass_rate_drop=0.25,
        max_mean_score_drop=0.5,
        allow_worse_failure_modes=True,
    )

    assert updated.suite == "custom.json"
    assert updated.baseline_model_id == "dummy_base"
    assert updated.candidate_model_id == "dummy_candidate"
    assert updated.thresholds.max_pass_rate_drop == 0.25
    assert updated.thresholds.max_mean_score_drop == 0.5
    assert updated.thresholds.allow_worse_failure_modes is True


def test_training_preflight_load_success_and_failures(tmp_path: Path) -> None:
    preflight = _write_preflight(tmp_path)
    assert load_training_preflight(preflight)["version"] == TRAINING_PREFLIGHT_VERSION

    missing = tmp_path / "missing.json"
    try:
        load_training_preflight(missing)
    except ValueError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("expected missing preflight failure")

    bad = tmp_path / "bad_preflight.json"
    bad.write_text(
        json.dumps({"version": TRAINING_PREFLIGHT_VERSION, "ok": False}),
        encoding="utf-8",
    )
    try:
        load_training_preflight(bad)
    except ValueError as exc:
        assert "not ok" in str(exc)
    else:
        raise AssertionError("expected non-ok preflight failure")


def test_distillation_comparison_identical_results_pass() -> None:
    baseline = _suite([_case("a"), _case("b", score=0.8)])

    result = compare_distillation_results(
        baseline,
        baseline,
        thresholds=DistillationThresholds(),
    )

    assert result["ok"] is True
    assert result["changed_cases"] == []


def test_distillation_comparison_pass_rate_drop_fails() -> None:
    baseline = _suite([_case("a"), _case("b")])
    candidate = _suite([_case("a"), _case("b", passed=False, score=0.0)])

    result = compare_distillation_results(
        baseline,
        candidate,
        thresholds=DistillationThresholds(),
    )

    assert result["ok"] is False
    assert any("pass_rate_drop" in issue for issue in result["issues"])


def test_distillation_comparison_mean_score_drop_fails() -> None:
    baseline = _suite([_case("a", score=0.9)])
    candidate = _suite([_case("a", score=0.8)])

    result = compare_distillation_results(
        baseline,
        candidate,
        thresholds=DistillationThresholds(max_mean_score_drop=0.05),
    )

    assert result["ok"] is False
    assert any("mean_score_drop" in issue for issue in result["issues"])


def test_distillation_comparison_worse_failure_mode_fails() -> None:
    baseline = _suite([_case("a", passed=False, score=0.0, failure_mode="expectation_failed")])
    candidate = _suite([_case("a", passed=False, score=0.0, failure_mode="runtime_error")])

    result = compare_distillation_results(
        baseline,
        candidate,
        thresholds=DistillationThresholds(),
    )

    assert result["ok"] is False
    assert len(result["worse_failure_mode_movements"]) == 1


def test_distillation_cost_latency_gates_skip_or_enforce(tmp_path: Path) -> None:
    baseline = _suite([_case("a")])
    candidate = _suite([_case("a")])
    skipped = compare_distillation_results(
        baseline,
        candidate,
        thresholds=DistillationThresholds(min_cost_per_success_improvement=0.1),
    )
    assert skipped["ok"] is True
    assert any(g["name"] == "cost_per_success_improvement" and g["skipped"] for g in skipped["gate_results"])

    baseline_logs = _write_reward_run(tmp_path / "baseline_logs", cost=1.0, latency=100)
    candidate_logs = _write_reward_run(tmp_path / "candidate_logs", cost=0.95, latency=95)
    baseline = _suite([_case("a")], run_logs_dir=str(baseline_logs))
    candidate = _suite([_case("a")], run_logs_dir=str(candidate_logs))
    enforced = compare_distillation_results(
        baseline,
        candidate,
        thresholds=DistillationThresholds(
            min_cost_per_success_improvement=0.1,
            min_latency_ms_p50_improvement=10.0,
        ),
    )

    assert enforced["ok"] is False
    assert any("cost_per_success_improvement" in issue for issue in enforced["issues"])
    assert any("latency_ms_p50_improvement" in issue for issue in enforced["issues"])


def test_distillation_check_writes_reports_for_dummy_recipe(tmp_path: Path) -> None:
    preflight = _write_preflight(tmp_path)
    recipe = _write_recipe(tmp_path, preflight)

    payload = run_distillation_check(
        recipe_path=recipe,
        work_dir=tmp_path / "distill",
    )

    assert payload["ok"] is True
    assert payload["version"] == "distillation_eval.v0"
    assert Path(payload["report_json"]).exists()
    assert Path(payload["report_md"]).exists()


def test_distillation_check_default_suite_runs_from_arbitrary_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _write_preflight(tmp_path)
    recipe = _write_recipe(tmp_path, preflight)
    cwd = tmp_path / "outside-repo"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    payload = run_distillation_check(
        recipe_path=recipe,
        work_dir=tmp_path / "distill",
    )

    assert payload["ok"] is True
    assert Path(payload["report_json"]).exists()


def test_distillation_check_degraded_candidate_exits_nonzero(tmp_path: Path) -> None:
    preflight = _write_preflight(tmp_path)
    recipe = _write_recipe(tmp_path, preflight, candidate_model_id="dummy_bad")

    code = distill_main(
        [
            "--recipe",
            str(recipe),
            "--work-dir",
            str(tmp_path / "distill"),
        ]
    )

    report = json.loads((tmp_path / "distill" / "distillation_eval.json").read_text())
    assert code == 1
    assert report["ok"] is False
    assert any("mean_score_drop" in issue for issue in report["issues"])


def test_distillation_check_cli_overrides_recipe(tmp_path: Path) -> None:
    preflight = _write_preflight(tmp_path)
    recipe = _write_recipe(tmp_path, preflight, candidate_model_id="dummy_good")

    code = distill_main(
        [
            "--recipe",
            str(recipe),
            "--work-dir",
            str(tmp_path / "distill"),
            "--candidate-model-id",
            "dummy_bad",
            "--max-mean-score-drop",
            "1.0",
        ]
    )

    report = json.loads((tmp_path / "distill" / "distillation_eval.json").read_text())
    assert code == 0
    assert report["recipe"]["candidate_model_id"] == "dummy_bad"


def _write_preflight(tmp_path: Path) -> Path:
    path = tmp_path / "training_preflight.json"
    path.write_text(
        json.dumps(
            {
                "version": TRAINING_PREFLIGHT_VERSION,
                "ok": True,
                "base_model": "dummy/base",
                "quality_gate": True,
                "datasets": [],
                "artifact_manifest": {
                    "version": "training_artifact.v0",
                    "base_model": "dummy/base",
                    "dataset_sources": {"sft": "datasets/pilot.sft.jsonl"},
                    "dataset_hashes": {"sft": "abc"},
                    "split_counts": {"sft": {"train": 1}},
                    "outputs": {"sft_out": "artifacts/sft"},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_recipe(
    tmp_path: Path,
    preflight: Path,
    *,
    candidate_model_id: str = "dummy_good",
) -> Path:
    path = tmp_path / "distillation_recipe.json"
    path.write_text(
        json.dumps(
            {
                "version": DISTILLATION_RECIPE_VERSION,
                "id": "pilot-distill",
                "domain": "pilot",
                "suite": DEFAULT_SUITE,
                "baseline_model_id": "dummy_good",
                "candidate_model_id": candidate_model_id,
                "training_preflight_path": str(preflight),
                "thresholds": {
                    "max_pass_rate_drop": 0.0,
                    "max_mean_score_drop": 0.0,
                    "allow_worse_failure_modes": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


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
        "trace_path": f"logs/{case_id}.jsonl",
        "terminal_status": "success" if passed else "failure",
        "terminal_reason": None,
        "failure_mode": failure_mode,
        "artifact_refs": {"trace": f"logs/{case_id}.jsonl"},
    }


def _suite(cases: list[dict], *, run_logs_dir: str = "") -> dict:
    passed = sum(1 for case in cases if case["passed"])
    return {
        "suite_name": "decision_v0",
        "agent": "decision_agent",
        "model_id": "dummy_good",
        "run_id": "run-1",
        "run_logs_dir": run_logs_dir,
        "num_cases": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "pass_rate": passed / len(cases) if cases else 0.0,
        "results": cases,
    }


def _write_reward_run(path: Path, *, cost: float, latency: int) -> Path:
    path.mkdir(parents=True)
    row = {
        "version": "reward.v0",
        "suite_id": "decision_v0",
        "case_id": "a",
        "model_id": "dummy_good",
        "success": True,
        "overall_score": 1.0,
        "cost_usd": cost,
        "latency_ms": latency,
    }
    (path / "reward.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path
