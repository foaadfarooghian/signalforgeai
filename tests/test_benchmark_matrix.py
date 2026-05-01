from __future__ import annotations

import json
from pathlib import Path

from tensorfoundry.evaluation import matrix
from tensorfoundry.evaluation.matrix import (
    BENCHMARK_MATRIX_VERSION,
    aggregate_suite_metrics,
    apply_matrix_overrides,
    compute_frontiers,
    load_benchmark_matrix_config,
    main as matrix_main,
    resolve_suite_path,
    run_benchmark_matrix,
)
from tensorfoundry.models.registry import ProviderCheck


def test_benchmark_matrix_config_loader_accepts_json_and_yaml(tmp_path: Path) -> None:
    json_config = _write_config(tmp_path / "matrix.json")
    yaml_config = tmp_path / "matrix.yaml"
    yaml_config.write_text(
        "\n".join(
            [
                f"version: {BENCHMARK_MATRIX_VERSION}",
                "id: yaml-matrix",
                "suites:",
                "  - decision_v0",
                "model_ids:",
                "  - dummy_good",
                "metric_weights:",
                "  lambda_cost: 0.1",
                "  mu_latency: 0.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_benchmark_matrix_config(json_config).id == "offline-matrix"
    loaded_yaml = load_benchmark_matrix_config(yaml_config)
    assert loaded_yaml.id == "yaml-matrix"
    assert loaded_yaml.metric_weights.lambda_cost == 0.1
    assert loaded_yaml.metric_weights.mu_latency == 0.2


def test_benchmark_matrix_config_loader_rejects_missing_fields(tmp_path: Path) -> None:
    config = tmp_path / "bad.json"
    config.write_text(json.dumps({"version": BENCHMARK_MATRIX_VERSION}), encoding="utf-8")

    try:
        load_benchmark_matrix_config(config)
    except ValueError as exc:
        assert "missing required matrix config field" in str(exc)
    else:
        raise AssertionError("expected config validation failure")


def test_suite_aliases_resolve_to_packaged_suites(tmp_path: Path) -> None:
    config = load_benchmark_matrix_config(_write_config(tmp_path / "matrix.json"))

    suite_path = resolve_suite_path(config, "decision_v0")

    assert suite_path.name == "decision_v0.json"
    assert suite_path.exists()


def test_matrix_overrides_filter_suites_and_models(tmp_path: Path) -> None:
    config = load_benchmark_matrix_config(_write_config(tmp_path / "matrix.json"))

    updated = apply_matrix_overrides(
        config,
        suites=["research_v0"],
        model_ids=["dummy_bad"],
        lambda_cost=0.3,
        mu_latency=0.4,
        require_providers=["all"],
    )

    assert updated.suites == ["research_v0"]
    assert updated.model_ids == ["dummy_bad"]
    assert updated.metric_weights.lambda_cost == 0.3
    assert updated.metric_weights.mu_latency == 0.4
    assert updated.require_providers == ["all"]


def test_provider_skip_and_required_failure(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(
        tmp_path / "matrix.json",
        model_ids=["openai:gpt-5-mini"],
    )

    def fake_check(model_id: str, *, required: bool = False, timeout_s: float = 2.0) -> ProviderCheck:
        return ProviderCheck(
            model_id=model_id,
            provider="openai",
            ok=False,
            required=required,
            skipped=not required,
            reason="missing test credential",
        )

    monkeypatch.setattr(matrix, "check_provider_for_model", fake_check)

    optional = run_benchmark_matrix(
        config_path=config_path,
        work_dir=tmp_path / "optional",
    )
    required = run_benchmark_matrix(
        config_path=config_path,
        work_dir=tmp_path / "required",
        require_providers=["openai"],
    )

    assert optional["ok"] is True
    assert optional["scorecard"][0]["skipped"] is True
    assert required["ok"] is False
    assert required["scorecard"][0]["skipped"] is False
    assert "missing test credential" in required["issues"][0]


def test_aggregate_suite_metrics_computes_tradeoff_fields() -> None:
    suite = _suite_payload()
    rows = [
        _reward("a", score=1.0, success=True, cost=0.2, latency=100, tokens=10),
        _reward("b", score=0.5, success=False, cost=0.1, latency=300, tokens=20),
    ]

    metrics = aggregate_suite_metrics(
        suite,
        reward_rows=rows,
        lambda_cost=0.5,
        mu_latency=0.1,
    )

    assert metrics["pass_rate"] == 0.5
    assert metrics["mean_score"] == 0.75
    assert metrics["cost_usd_total"] == 0.3
    assert metrics["cost_per_success_usd"] == 0.3
    assert metrics["latency_ms_p50"] == 100
    assert metrics["latency_ms_p95"] == 300
    assert metrics["total_tokens"] == 30
    assert round(metrics["mean_effective"], 3) == 0.655


def test_frontier_selection_identifies_expected_rows() -> None:
    rows = [
        _scorecard_row("suite", "fast", pass_rate=0.8, score=0.8, cost=0.2, latency=20, effective=0.7),
        _scorecard_row("suite", "quality", pass_rate=1.0, score=0.95, cost=0.5, latency=60, effective=0.8),
        _scorecard_row("suite", "cheap", pass_rate=0.7, score=0.7, cost=0.01, latency=80, effective=0.9),
    ]

    frontiers = compute_frontiers(rows)
    picks = frontiers["suite"]["picks"]

    assert picks["best_quality"]["model_id"] == "quality"
    assert picks["best_effective"]["model_id"] == "cheap"
    assert picks["lowest_cost_per_success"]["model_id"] == "cheap"
    assert picks["lowest_latency_p50"]["model_id"] == "fast"
    assert {row["model_id"] for row in frontiers["suite"]["pareto"]} == {
        "cheap",
        "fast",
        "quality",
    }


def test_benchmark_matrix_runs_offline_and_writes_reports(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "matrix.json")

    code = matrix_main(
        [
            "--config",
            str(config),
            "--work-dir",
            str(tmp_path / "matrix-out"),
        ]
    )

    report_path = tmp_path / "matrix-out" / "benchmark_matrix.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["version"] == BENCHMARK_MATRIX_VERSION
    assert payload["ok"] is True
    assert len(payload["scorecard"]) == 2
    assert (tmp_path / "matrix-out" / "benchmark_matrix.md").exists()
    assert not (tmp_path / "matrix-out" / "learning_ops").exists()


def test_required_provider_cli_exits_nonzero(tmp_path: Path, monkeypatch) -> None:
    config = _write_config(tmp_path / "matrix.json", model_ids=["ollama:test"])

    def fake_check(model_id: str, *, required: bool = False, timeout_s: float = 2.0) -> ProviderCheck:
        return ProviderCheck(
            model_id=model_id,
            provider="ollama",
            ok=False,
            required=required,
            skipped=not required,
            reason="ollama unavailable in test",
        )

    monkeypatch.setattr(matrix, "check_provider_for_model", fake_check)

    code = matrix_main(
        [
            "--config",
            str(config),
            "--work-dir",
            str(tmp_path / "matrix-out"),
            "--require-provider",
            "ollama",
        ]
    )

    payload = json.loads((tmp_path / "matrix-out" / "benchmark_matrix.json").read_text())
    assert code == 1
    assert payload["ok"] is False
    assert "ollama unavailable in test" in payload["issues"][0]


def _write_config(
    path: Path,
    *,
    model_ids: list[str] | None = None,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": BENCHMARK_MATRIX_VERSION,
                "id": "offline-matrix",
                "suites": ["decision_v0"],
                "model_ids": model_ids or ["dummy_good", "dummy_bad"],
                "metric_weights": {
                    "lambda_cost": 0.0,
                    "mu_latency": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _suite_payload() -> dict:
    return {
        "suite_name": "suite",
        "agent": "decision_agent",
        "model_id": "dummy_good",
        "run_id": "run-1",
        "run_logs_dir": "",
        "num_cases": 2,
        "passed": 1,
        "failed": 1,
        "pass_rate": 0.5,
        "results": [],
    }


def _reward(
    case_id: str,
    *,
    score: float,
    success: bool,
    cost: float,
    latency: int,
    tokens: int,
) -> dict:
    return {
        "version": "reward.v0",
        "suite_id": "suite",
        "case_id": case_id,
        "model_id": "dummy_good",
        "success": success,
        "overall_score": score,
        "cost_usd": cost,
        "latency_ms": latency,
        "total_tokens": tokens,
    }


def _scorecard_row(
    suite: str,
    model_id: str,
    *,
    pass_rate: float,
    score: float,
    cost: float,
    latency: int,
    effective: float,
) -> dict:
    return {
        "suite": suite,
        "model_id": model_id,
        "provider": "dummy",
        "ok": True,
        "skipped": False,
        "metrics": {
            "pass_rate": pass_rate,
            "mean_score": score,
            "cost_per_success_usd": cost,
            "latency_ms_p50": latency,
            "latency_ms_p95": latency,
            "mean_effective": effective,
        },
    }
