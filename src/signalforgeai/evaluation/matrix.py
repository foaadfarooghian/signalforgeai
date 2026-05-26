"""Benchmark matrix runner and frontier reports for SignalForge AI."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yaml

from signalforgeai.evaluation.harness import run_suite
from signalforgeai.models.registry import ProviderCheck, check_provider_for_model


BENCHMARK_MATRIX_VERSION = "benchmark_matrix.v0"
BUILTIN_SUITE_ROOT = Path(__file__).parent / "benchmarks"


@dataclass(frozen=True)
class MetricWeights:
    """Weights used to compute mean effective score."""

    lambda_cost: float = 0.0
    mu_latency: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "lambda_cost": self.lambda_cost,
            "mu_latency": self.mu_latency,
        }


@dataclass(frozen=True)
class BenchmarkMatrixConfig:
    """A v0 benchmark matrix configuration."""

    version: str
    id: str
    suites: List[str]
    model_ids: List[str]
    metric_weights: MetricWeights
    require_providers: List[str]
    source_path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "id": self.id,
            "suites": self.suites,
            "model_ids": self.model_ids,
            "metric_weights": self.metric_weights.to_dict(),
            "require_providers": self.require_providers,
            "source_path": self.source_path,
        }


@dataclass(frozen=True)
class MatrixRow:
    """One suite/model benchmark matrix row."""

    suite: str
    suite_path: str
    model_id: str
    provider: str
    provider_required: bool
    provider_ok: bool
    provider_skipped: bool
    skipped: bool
    ok: bool
    issue: Optional[str]
    result_path: Optional[str]
    summary_path: Optional[str]
    run_logs_dir: Optional[str]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_benchmark_matrix_config(path: str | Path) -> BenchmarkMatrixConfig:
    """Load and validate a `benchmark_matrix.v0` JSON or YAML config."""
    source = Path(path)
    if not source.exists():
        raise ValueError(f"benchmark matrix config not found: {source}")
    payload = _load_mapping(source)
    for field in ("version", "id", "suites", "model_ids"):
        if field not in payload:
            raise ValueError(f"missing required matrix config field: {field}")

    version = _as_str(payload["version"], "version")
    if version != BENCHMARK_MATRIX_VERSION:
        raise ValueError(
            f"unsupported benchmark matrix version: {version!r}; "
            f"expected {BENCHMARK_MATRIX_VERSION!r}"
        )

    metric_weights_raw = payload.get("metric_weights", {})
    if metric_weights_raw is None:
        metric_weights_raw = {}
    if not isinstance(metric_weights_raw, dict):
        raise ValueError("metric_weights must be an object")
    lambda_cost = payload.get("lambda_cost", metric_weights_raw.get("lambda_cost", 0.0))
    mu_latency = payload.get("mu_latency", metric_weights_raw.get("mu_latency", 0.0))

    require_providers = payload.get("require_providers", [])
    if require_providers is None:
        require_providers = []

    return BenchmarkMatrixConfig(
        version=version,
        id=_as_str(payload["id"], "id"),
        suites=_as_str_list(payload["suites"], "suites"),
        model_ids=_as_str_list(payload["model_ids"], "model_ids"),
        metric_weights=MetricWeights(
            lambda_cost=_as_float(lambda_cost, "lambda_cost"),
            mu_latency=_as_float(mu_latency, "mu_latency"),
        ),
        require_providers=_as_str_list(
            require_providers,
            "require_providers",
            allow_empty=True,
        ),
        source_path=str(source),
    )


def apply_matrix_overrides(
    config: BenchmarkMatrixConfig,
    *,
    suites: Optional[Sequence[str]] = None,
    model_ids: Optional[Sequence[str]] = None,
    lambda_cost: Optional[float] = None,
    mu_latency: Optional[float] = None,
    require_providers: Optional[Sequence[str]] = None,
) -> BenchmarkMatrixConfig:
    """Return a matrix config with CLI overrides applied."""
    weights = config.metric_weights
    if lambda_cost is not None:
        weights = replace(weights, lambda_cost=float(lambda_cost))
    if mu_latency is not None:
        weights = replace(weights, mu_latency=float(mu_latency))
    providers = list(config.require_providers)
    if require_providers:
        providers = list(require_providers)
    return replace(
        config,
        suites=list(suites) if suites else config.suites,
        model_ids=list(model_ids) if model_ids else config.model_ids,
        metric_weights=weights,
        require_providers=providers,
    )


def resolve_suite_path(config: BenchmarkMatrixConfig, value: str) -> Path:
    """Resolve a suite alias or path to a concrete suite JSON path."""
    raw = value.strip()
    path = Path(raw)
    if path.is_absolute() and path.exists():
        return path
    config_relative = Path(config.source_path).parent / path
    if config_relative.exists():
        return config_relative
    if path.exists():
        return path

    aliases = _builtin_suite_aliases()
    found = _resolve_builtin_suite_alias(aliases, raw, path)
    if found is not None:
        return found

    raise ValueError(f"unknown suite alias or path: {value}")


def resolve_suite_value(value: str, *, source_path: str | Path | None = None) -> Path:
    """Resolve a single suite alias or path without requiring a matrix config."""

    config = BenchmarkMatrixConfig(
        version=BENCHMARK_MATRIX_VERSION,
        id="suite-resolution",
        suites=[value],
        model_ids=["dummy_good"],
        metric_weights=MetricWeights(),
        require_providers=[],
        source_path=str(
            Path(source_path)
            if source_path is not None
            else Path.cwd() / "suite_resolution.json"
        ),
    )
    return resolve_suite_path(config, value)


def run_benchmark_matrix(
    *,
    config_path: Path,
    work_dir: Path,
    suites: Optional[Sequence[str]] = None,
    model_ids: Optional[Sequence[str]] = None,
    lambda_cost: Optional[float] = None,
    mu_latency: Optional[float] = None,
    require_providers: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run the configured benchmark matrix and write JSON/Markdown reports."""
    config = apply_matrix_overrides(
        load_benchmark_matrix_config(config_path),
        suites=suites,
        model_ids=model_ids,
        lambda_cost=lambda_cost,
        mu_latency=mu_latency,
        require_providers=require_providers,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    _reset_generated_outputs(work_dir)

    issues: List[str] = []
    rows: List[MatrixRow] = []
    required_providers = _expand_required_providers(config.require_providers)

    for suite_value in config.suites:
        try:
            suite_path = resolve_suite_path(config, suite_value)
        except ValueError as exc:
            issues.append(str(exc))
            continue

        suite_name = _suite_name(suite_path)
        for model_id in config.model_ids:
            row = _run_matrix_row(
                suite_name=suite_name,
                suite_path=suite_path,
                model_id=model_id,
                work_dir=work_dir,
                required_providers=required_providers,
                weights=config.metric_weights,
            )
            rows.append(row)
            if row.issue and not row.skipped:
                issues.append(row.issue)

    scorecard = [row.to_dict() for row in rows]
    frontiers = compute_frontiers(scorecard)
    ok = bool(not issues and all(row.ok or row.skipped for row in rows))
    payload: Dict[str, Any] = {
        "version": BENCHMARK_MATRIX_VERSION,
        "ok": ok,
        "config": config.to_dict(),
        "work_dir": str(work_dir),
        "scorecard": scorecard,
        "frontiers": frontiers,
        "issues": issues,
        "artifact_refs": {
            "json": str(work_dir / "benchmark_matrix.json"),
            "markdown": str(work_dir / "benchmark_matrix.md"),
            "results_dir": str(work_dir / "results"),
            "logs_dir": str(work_dir / "logs"),
        },
    }
    write_benchmark_matrix_reports(
        payload,
        json_path=work_dir / "benchmark_matrix.json",
        markdown_path=work_dir / "benchmark_matrix.md",
    )
    return payload


def aggregate_suite_metrics(
    suite: Dict[str, Any],
    *,
    reward_rows: Sequence[Dict[str, Any]],
    lambda_cost: float = 0.0,
    mu_latency: float = 0.0,
) -> Dict[str, Any]:
    """Aggregate matrix metrics from suite results and reward rows."""
    cases = int(suite.get("num_cases") or len(suite.get("results", [])))
    passed = int(suite.get("passed") or 0)
    failed = int(suite.get("failed") or max(0, cases - passed))
    pass_rate = float(suite.get("pass_rate") or (passed / cases if cases else 0.0))
    scores: List[float] = []
    costs: List[float] = []
    latencies: List[int] = []
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    saw_input_tokens = False
    saw_output_tokens = False
    saw_total_tokens = False
    effective_scores: List[float] = []
    failure_modes: Dict[str, int] = {}
    retry_events = 0

    for row in reward_rows:
        if row.get("version") != "reward.v0":
            continue
        score = _optional_float(row.get("overall_score"))
        if score is not None:
            scores.append(score)
        cost = _optional_float(row.get("cost_usd"))
        if cost is not None:
            costs.append(cost)
        latency = _optional_int(row.get("latency_ms"))
        if latency is not None:
            latencies.append(latency)
        in_tok = _optional_int(row.get("input_tokens"))
        out_tok = _optional_int(row.get("output_tokens"))
        tot_tok = _optional_int(row.get("total_tokens"))
        if in_tok is not None:
            input_tokens += in_tok
            saw_input_tokens = True
        if out_tok is not None:
            output_tokens += out_tok
            saw_output_tokens = True
        if tot_tok is not None:
            total_tokens += tot_tok
            saw_total_tokens = True
        mode = row.get("failure_mode")
        if isinstance(mode, str) and mode:
            failure_modes[mode] = failure_modes.get(mode, 0) + 1
        diagnosis = row.get("diagnosis")
        if isinstance(diagnosis, dict):
            retry_events += _count_recovery_like_events(diagnosis)
        if score is not None:
            effective_scores.append(
                effective_score(
                    score=score,
                    cost_usd=cost or 0.0,
                    latency_ms=latency or 0,
                    lambda_cost=lambda_cost,
                    mu_latency=mu_latency,
                )
            )

    if not scores:
        scores = [
            float(case.get("score", 0.0))
            for case in suite.get("results", [])
            if isinstance(case, dict) and isinstance(case.get("score"), (int, float))
        ]

    success_count = sum(1 for row in reward_rows if row.get("success") is True)
    total_cost = round(sum(costs), 10) if costs else None
    cost_per_success = (
        round(sum(costs) / success_count, 10) if costs and success_count > 0 else None
    )
    return {
        "cases": cases,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "failure_rate": 1.0 - pass_rate,
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "failure_modes": failure_modes,
        "retry_event_count": retry_events,
        "retry_rate": retry_events / cases if cases else 0.0,
        "cost_usd_total": total_cost,
        "cost_per_success_usd": cost_per_success,
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p95": percentile(latencies, 0.95),
        "input_tokens_total": input_tokens if saw_input_tokens else None,
        "output_tokens_total": output_tokens if saw_output_tokens else None,
        "total_tokens": total_tokens if saw_total_tokens else None,
        "mean_effective": (
            sum(effective_scores) / len(effective_scores) if effective_scores else None
        ),
    }


def effective_score(
    *,
    score: float,
    cost_usd: float,
    latency_ms: int,
    lambda_cost: float,
    mu_latency: float,
) -> float:
    """Compute clamped effective score from quality, cost, and latency."""
    eff = score - lambda_cost * cost_usd - mu_latency * (latency_ms / 1000.0)
    return max(0.0, min(1.0, eff))


def percentile(values: Sequence[int], q: float) -> Optional[int]:
    """Return a nearest-rank percentile for integer latency values."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def compute_frontiers(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute named picks and Pareto-efficient rows per suite."""
    by_suite: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("skipped") or not row.get("ok"):
            continue
        suite = str(row.get("suite") or "unknown")
        by_suite.setdefault(suite, []).append(row)

    out: Dict[str, Any] = {}
    for suite, suite_rows in sorted(by_suite.items()):
        picks = {
            "best_quality": _pick_best(suite_rows, [("pass_rate", True), ("mean_score", True)]),
            "best_effective": _pick_best(suite_rows, [("mean_effective", True)]),
            "lowest_cost_per_success": _pick_best(
                suite_rows,
                [("cost_per_success_usd", False), ("pass_rate", True)],
            ),
            "lowest_latency_p50": _pick_best(
                suite_rows,
                [("latency_ms_p50", False), ("pass_rate", True)],
            ),
        }
        out[suite] = {
            "picks": picks,
            "pareto": [_frontier_ref(row) for row in _pareto_rows(suite_rows)],
        }
    return out


def write_benchmark_matrix_reports(
    payload: Dict[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    """Write JSON and Markdown benchmark matrix reports."""
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(format_benchmark_matrix_markdown(payload), encoding="utf-8")


def format_benchmark_matrix_markdown(payload: Dict[str, Any]) -> str:
    """Render a `benchmark_matrix.v0` payload as Markdown."""
    config_raw = payload.get("config")
    config = config_raw if isinstance(config_raw, dict) else {}
    weights_raw = config.get("metric_weights")
    weights = weights_raw if isinstance(weights_raw, dict) else {}
    lines = [
        "# SignalForge AI Benchmark Matrix",
        "",
        f"- OK: `{str(payload.get('ok')).lower()}`",
        f"- Matrix: `{config.get('id')}`",
        f"- Work dir: `{payload.get('work_dir')}`",
        f"- Lambda cost: `{_fmt(weights.get('lambda_cost'))}`",
        f"- Mu latency: `{_fmt(weights.get('mu_latency'))}`",
        "",
        "## Scorecard",
        "",
        "| suite | model_id | provider | ok | skipped | cases | pass_rate | mean_score | cost/success | latency p50 | latency p95 | failure_rate | retry_rate | mean_effective |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("scorecard", []):
        if not isinstance(row, dict):
            continue
        metrics_raw = row.get("metrics")
        metrics = metrics_raw if isinstance(metrics_raw, dict) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("suite") or ""),
                    f"`{row.get('model_id') or ''}`",
                    str(row.get("provider") or ""),
                    str(row.get("ok")).lower(),
                    str(row.get("skipped")).lower(),
                    _fmt(metrics.get("cases"), digits=0),
                    _fmt_pct(metrics.get("pass_rate")),
                    _fmt(metrics.get("mean_score")),
                    _fmt(metrics.get("cost_per_success_usd"), digits=6),
                    _fmt(metrics.get("latency_ms_p50"), digits=0),
                    _fmt(metrics.get("latency_ms_p95"), digits=0),
                    _fmt_pct(metrics.get("failure_rate")),
                    _fmt_pct(metrics.get("retry_rate")),
                    _fmt(metrics.get("mean_effective")),
                ]
            )
            + " |"
        )

    frontiers = payload.get("frontiers")
    if isinstance(frontiers, dict) and frontiers:
        lines.extend(["", "## Frontier Picks", ""])
        for suite, data in sorted(frontiers.items()):
            data_dict = data if isinstance(data, dict) else {}
            picks_raw = data_dict.get("picks")
            picks = picks_raw if isinstance(picks_raw, dict) else {}
            lines.extend(
                [
                    f"### {suite}",
                    "",
                    "| pick | model_id | pass_rate | mean_score | cost/success | latency p50 | mean_effective |",
                    "|---|---|---:|---:|---:|---:|---:|",
                ]
            )
            for name, pick in picks.items():
                if not isinstance(pick, dict):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            name,
                            f"`{pick.get('model_id') or ''}`",
                            _fmt_pct(pick.get("pass_rate")),
                            _fmt(pick.get("mean_score")),
                            _fmt(pick.get("cost_per_success_usd"), digits=6),
                            _fmt(pick.get("latency_ms_p50"), digits=0),
                            _fmt(pick.get("mean_effective")),
                        ]
                    )
                    + " |"
                )
            pareto = data_dict.get("pareto")
            if isinstance(pareto, list):
                pareto_models = ", ".join(
                    f"`{row.get('model_id')}`" for row in pareto if isinstance(row, dict)
                )
                lines.extend(["", f"Pareto: {pareto_models or '_none_'}", ""])

    if payload.get("issues"):
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in payload["issues"])
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a SignalForge AI benchmark matrix.")
    parser.add_argument("--config", type=str, required=True, help="benchmark_matrix.v0 config path")
    parser.add_argument("--work-dir", type=str, required=True, help="Output work directory")
    parser.add_argument("--suite", action="append", default=[], help="Suite alias/path; may be repeated")
    parser.add_argument("--model-id", action="append", default=[], help="Model id; may be repeated")
    parser.add_argument("--lambda-cost", type=float, default=None, help="Cost penalty weight")
    parser.add_argument("--mu-latency", type=float, default=None, help="Latency penalty weight per second")
    parser.add_argument(
        "--require-provider",
        action="append",
        default=[],
        choices=["dummy", "hosted", "local", "openai", "ollama", "hf", "all"],
        help="Fail if rows for this provider group are unavailable",
    )
    args = parser.parse_args(argv)

    try:
        payload = run_benchmark_matrix(
            config_path=Path(args.config),
            work_dir=Path(args.work_dir),
            suites=args.suite or None,
            model_ids=args.model_id or None,
            lambda_cost=args.lambda_cost,
            mu_latency=args.mu_latency,
            require_providers=args.require_provider or None,
        )
    except ValueError as exc:
        print(f"Benchmark matrix: FAILED ({exc})")
        return 2

    print(f"Benchmark matrix: {'OK' if payload['ok'] else 'FAILED'}")
    print(f"Report JSON: {payload['artifact_refs']['json']}")
    print(f"Report Markdown: {payload['artifact_refs']['markdown']}")
    return 0 if payload["ok"] else 1


def _run_matrix_row(
    *,
    suite_name: str,
    suite_path: Path,
    model_id: str,
    work_dir: Path,
    required_providers: Set[str],
    weights: MetricWeights,
) -> MatrixRow:
    provider_check = check_provider_for_model(
        model_id,
        required=_is_provider_required(model_id, required_providers),
    )
    if provider_check.skipped:
        return _skipped_row(
            suite_name=suite_name,
            suite_path=suite_path,
            model_id=model_id,
            provider_check=provider_check,
            issue=provider_check.reason or "provider unavailable",
        )
    if not provider_check.ok:
        return _failed_row(
            suite_name=suite_name,
            suite_path=suite_path,
            model_id=model_id,
            provider_check=provider_check,
            issue=provider_check.reason or "provider unavailable",
        )

    model_slug = _slug(model_id)
    suite_slug = _slug(suite_name)
    output_dir = work_dir / "results" / suite_slug / model_slug
    logs_dir = work_dir / "logs" / suite_slug / model_slug
    old_model_id = os.environ.get("SIGNALFORGEAI_MODEL_ID")
    os.environ["SIGNALFORGEAI_MODEL_ID"] = model_id
    try:
        suite_result = run_suite(suite_path=suite_path, output_dir=output_dir, logs_dir=logs_dir)
    except Exception as exc:
        return _failed_row(
            suite_name=suite_name,
            suite_path=suite_path,
            model_id=model_id,
            provider_check=provider_check,
            issue=f"suite execution failed: {type(exc).__name__}: {exc}",
        )
    finally:
        if old_model_id is None:
            os.environ.pop("SIGNALFORGEAI_MODEL_ID", None)
        else:
            os.environ["SIGNALFORGEAI_MODEL_ID"] = old_model_id

    suite_payload = asdict(suite_result)
    reward_rows = _read_reward_rows(Path(suite_result.run_logs_dir))
    metrics = aggregate_suite_metrics(
        suite_payload,
        reward_rows=reward_rows,
        lambda_cost=weights.lambda_cost,
        mu_latency=weights.mu_latency,
    )
    ok = True
    issue = None
    return MatrixRow(
        suite=suite_result.suite_name,
        suite_path=str(suite_path),
        model_id=model_id,
        provider=provider_check.provider,
        provider_required=provider_check.required,
        provider_ok=provider_check.ok,
        provider_skipped=provider_check.skipped,
        skipped=False,
        ok=ok,
        issue=issue,
        result_path=str(output_dir / f"{suite_result.suite_name}.results.json"),
        summary_path=str(output_dir / f"{suite_result.suite_name}.summary.md"),
        run_logs_dir=suite_result.run_logs_dir,
        metrics=metrics,
    )


def _skipped_row(
    *,
    suite_name: str,
    suite_path: Path,
    model_id: str,
    provider_check: ProviderCheck,
    issue: str,
) -> MatrixRow:
    return MatrixRow(
        suite=suite_name,
        suite_path=str(suite_path),
        model_id=model_id,
        provider=provider_check.provider,
        provider_required=provider_check.required,
        provider_ok=provider_check.ok,
        provider_skipped=provider_check.skipped,
        skipped=True,
        ok=True,
        issue=issue,
        result_path=None,
        summary_path=None,
        run_logs_dir=None,
        metrics={},
    )


def _failed_row(
    *,
    suite_name: str,
    suite_path: Path,
    model_id: str,
    provider_check: ProviderCheck,
    issue: str,
) -> MatrixRow:
    return MatrixRow(
        suite=suite_name,
        suite_path=str(suite_path),
        model_id=model_id,
        provider=provider_check.provider,
        provider_required=provider_check.required,
        provider_ok=provider_check.ok,
        provider_skipped=provider_check.skipped,
        skipped=False,
        ok=False,
        issue=issue,
        result_path=None,
        summary_path=None,
        run_logs_dir=None,
        metrics={},
    )


def _reset_generated_outputs(work_dir: Path) -> None:
    for name in ("results", "logs"):
        path = work_dir / name
        if path.exists():
            shutil.rmtree(path)
    for name in ("benchmark_matrix.json", "benchmark_matrix.md"):
        path = work_dir / name
        if path.exists() or path.is_symlink():
            path.unlink()


def _load_mapping(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("benchmark matrix config must be an object")
    return data


def _as_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _as_str_list(value: Any, field: str, *, allow_empty: bool = False) -> List[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        requirement = "a list" if allow_empty else "a non-empty list"
        raise ValueError(f"{field} must be {requirement}")
    return [_as_str(item, field) for item in value]


def _as_float(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _optional_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _optional_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return int(value)
    return None


def _builtin_suite_aliases() -> Dict[str, Path]:
    aliases: Dict[str, Path] = {}
    for suite_path in BUILTIN_SUITE_ROOT.rglob("*.json"):
        aliases[suite_path.stem] = suite_path
        try:
            suite = json.loads(suite_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        suite_name = suite.get("suite_name")
        if isinstance(suite_name, str) and suite_name:
            aliases[suite_name] = suite_path
    return aliases


def _resolve_builtin_suite_alias(
    aliases: Dict[str, Path],
    raw: str,
    path: Path,
) -> Optional[Path]:
    candidates = [raw, raw.removesuffix(".json")]
    if _looks_like_builtin_suite_path(path):
        candidates.append(path.stem)
    for candidate in candidates:
        found = aliases.get(candidate)
        if found is not None:
            return found
    return None


def _looks_like_builtin_suite_path(path: Path) -> bool:
    parts = path.parts
    return (
        path.suffix == ".json"
        and "signalforgeai" in parts
        and "evaluation" in parts
        and "benchmarks" in parts
    )


def _suite_name(suite_path: Path) -> str:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    value = suite.get("suite_name")
    if not isinstance(value, str) or not value:
        raise ValueError(f"suite_name must be a non-empty string: {suite_path}")
    return value


def _expand_required_providers(values: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for value in values:
        normalized = value.strip().lower()
        if normalized == "all":
            out.add("all")
        elif normalized == "hosted":
            out.add("openai")
        elif normalized == "local":
            out.update({"ollama", "hf"})
        elif normalized:
            out.add(normalized)
    return out


def _is_provider_required(model_id: str, required_providers: Set[str]) -> bool:
    provider = check_provider_for_model(model_id, required=False).provider
    return provider == "dummy" or "all" in required_providers or provider in required_providers


def _read_reward_rows(run_logs_dir: Path) -> List[Dict[str, Any]]:
    path = run_logs_dir / "reward.jsonl"
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _count_recovery_like_events(diagnosis: Dict[str, Any]) -> int:
    events = diagnosis.get("recovery_like_events")
    if isinstance(events, list):
        return len(events)
    count = diagnosis.get("recovery_event_count")
    if isinstance(count, int):
        return count
    return 0


def _pick_best(rows: Sequence[Dict[str, Any]], criteria: Sequence[Tuple[str, bool]]) -> Optional[Dict[str, Any]]:
    candidates = [row for row in rows if _has_metric(row, criteria[0][0])]
    if not candidates:
        return None

    def sort_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
        parts: List[Any] = []
        for metric, higher_better in criteria:
            value = _metric(row, metric)
            if value is None:
                parts.append(float("-inf") if higher_better else float("inf"))
                continue
            parts.append(float(value) if higher_better else -float(value))
        parts.append(str(row.get("model_id") or ""))
        return tuple(parts)

    return _frontier_ref(max(candidates, key=sort_key))


def _pareto_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not any(_dominates(other, row) for other in rows if other is not row):
            out.append(row)
    return sorted(out, key=lambda r: (str(r.get("suite")), str(r.get("model_id"))))


def _dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    metrics = (
        ("pass_rate", True),
        ("mean_score", True),
        ("cost_per_success_usd", False),
        ("latency_ms_p50", False),
    )
    better_or_equal = True
    strictly_better = False
    for metric, higher_better in metrics:
        av = _metric(a, metric)
        bv = _metric(b, metric)
        if av is None or bv is None:
            continue
        if higher_better:
            if av < bv:
                better_or_equal = False
            if av > bv:
                strictly_better = True
        else:
            if av > bv:
                better_or_equal = False
            if av < bv:
                strictly_better = True
    return better_or_equal and strictly_better


def _has_metric(row: Dict[str, Any], name: str) -> bool:
    return _metric(row, name) is not None


def _metric(row: Dict[str, Any], name: str) -> Optional[float]:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(name)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _frontier_ref(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    metrics = row.get("metrics")
    metrics_dict = metrics if isinstance(metrics, dict) else {}
    return {
        "suite": row.get("suite"),
        "model_id": row.get("model_id"),
        "provider": row.get("provider"),
        "pass_rate": metrics_dict.get("pass_rate"),
        "mean_score": metrics_dict.get("mean_score"),
        "cost_per_success_usd": metrics_dict.get("cost_per_success_usd"),
        "latency_ms_p50": metrics_dict.get("latency_ms_p50"),
        "latency_ms_p95": metrics_dict.get("latency_ms_p95"),
        "mean_effective": metrics_dict.get("mean_effective"),
    }


def _slug(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)
    return out.strip("-") or "unknown"


def _fmt(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return f"{value:d}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return ""
    return f"{float(value):.2%}"


if __name__ == "__main__":
    raise SystemExit(main())
