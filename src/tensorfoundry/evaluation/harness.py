"""Evaluation harness for TensorFoundry agents.

Runs a suite of task cases, captures traces, validates them, and scores outcomes.
"""

from __future__ import annotations
import os
import json
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from tensorfoundry.logging.emitter import JsonlEmitter
from tensorfoundry.logging.inspect import summarise_trace, read_jsonl
from tensorfoundry.logging.validate import validate_trace_file
from tensorfoundry.evaluation.scorers import ScoringContext, resolve_scorer
from tensorfoundry.evaluation.reward_schema import RewardV0
from tensorfoundry.evaluation.reward_writer import write_rewards_jsonl

@dataclass(frozen=True)
class CaseResult:
    """Outcome and metadata for a single evaluation case."""
    case_id: str
    passed: bool
    score: float
    trace_id: str
    trace_path: str
    terminal_status: Optional[str]
    terminal_reason: Optional[str]
    notes: List[str]


@dataclass(frozen=True)
class SuiteResult:
    """Aggregate result for a full evaluation suite run."""
    suite_name: str
    agent: str
    run_id : str
    run_logs_dir: str
    num_cases: int
    passed: int
    failed: int
    pass_rate: float
    results: List[CaseResult]

def _get_commit_sha() -> str:
    """Resolve commit SHA from common CI environment variables."""
    # Prefer CI env if present; fall back to "unknown"
    return (
        os.getenv("GITHUB_SHA")
        or os.getenv("CI_COMMIT_SHA")
        or os.getenv("COMMIT_SHA")
        or "unknown"
    )

def _get_model_id() -> str:
    """Resolve the current model identifier from environment variables."""
    # If you route models elsewhere, replace this later.
    return os.getenv("TENSORFOUNDRY_MODEL_ID") or "unknown"


def _load_suite(path: Path) -> Dict[str, Any]:
    """Load a suite JSON file into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(p: Path) -> None:
    """Create a directory and parents if missing."""
    p.mkdir(parents=True, exist_ok=True)


def _extract_tradeoff_metrics(
    events: List[Dict[str, Any]],
) -> tuple[Optional[float], Optional[int], Dict[str, Optional[int]]]:
    """
    Extract aggregate trade-off metrics from trace events.

    Returns:
      (cost_usd, latency_ms, tokens_dict)

    tokens_dict keys: input_tokens, output_tokens, total_tokens
    """
    total_cost: float = 0.0
    total_latency: int = 0
    saw_cost = False
    saw_latency = False

    tok_in = 0
    tok_out = 0
    tok_total = 0
    saw_any_tokens = False

    def _as_int(x) -> Optional[int]:
        return int(x) if isinstance(x, (int, float)) else None

    for ev in events:
        if ev.get("event_type") != "model_called":
            continue

        m = ev.get("metrics") or {}
        c = m.get("cost_usd")
        latency = m.get("latency_ms")

        if isinstance(c, (int, float)):
            total_cost += float(c)
            saw_cost = True

        if isinstance(latency, int):
            total_latency += latency
            saw_latency = True
        
        usage = {}
        extra = m.get("extra") or {}
        if isinstance(extra, dict) and isinstance(extra.get("usage"), dict):
            usage = extra["usage"]
        else:
            # fallback: tokens were hoisted to top-level metrics
            usage = {
                "input_tokens": m.get("input_tokens"),
                "output_tokens": m.get("output_tokens"),
                "total_tokens": m.get("total_tokens"),
            }

        if isinstance(usage, dict):
            i = _as_int(usage.get("input_tokens"))
            o = _as_int(usage.get("output_tokens"))
            t = _as_int(usage.get("total_tokens"))

            if i is not None:
                tok_in += i
                saw_any_tokens = True
            if o is not None:
                tok_out += o
                saw_any_tokens = True
            if t is not None:
                tok_total += t
                saw_any_tokens = True

    cost_usd: Optional[float] = round(total_cost, 10) if saw_cost else None
    latency_ms: Optional[int] = total_latency if saw_latency else None

    tokens = {
        "input_tokens": tok_in if saw_any_tokens else None,
        "output_tokens": tok_out if saw_any_tokens else None,
        "total_tokens": tok_total if saw_any_tokens else None,
    }
    return cost_usd, latency_ms, tokens

def run_suite(
    *,
    suite_path: Path | str,
    output_dir: Path | str = Path("results"),
    logs_dir: Path | str = Path("logs"),
) -> SuiteResult:
    """Run a suite end-to-end and write results, summaries, and reward logs."""
    suite_path = Path(suite_path)
    output_dir = Path(output_dir)
    logs_dir = Path(logs_dir)

    suite = _load_suite(suite_path)
    suite_name: str = suite["suite_name"]
    agent_name: str = suite["agent"]
    cases: List[Dict[str, Any]] = suite["cases"]
    scoring_raw = suite.get("scoring", "default_v0")
    scoring_name = scoring_raw if isinstance(scoring_raw, str) else "default_v0"
    scoring_params_raw = suite.get("scoring_params", {})
    scoring_params = scoring_params_raw if isinstance(scoring_params_raw, dict) else {}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_logs_dir = logs_dir / run_id

    commit_sha = _get_commit_sha()
    model_id = _get_model_id()

    _ensure_dir(run_logs_dir)
    _ensure_dir(output_dir)

    runner = get_agent_runner(agent_name)
    scorer = resolve_scorer(scoring_name)

    case_results: List[CaseResult] = []
    rewards: List[RewardV0] = []

    for case in cases:
        case_id = case["id"]
        task = case["task"]
        inputs = case.get("inputs", {})
        expect = case.get("expect", {})
        if not isinstance(expect, dict):
            expect = {}
 
        # Create per-case trace file
        emitter = JsonlEmitter(
            run_logs_dir / "temp.jsonl",
            agent_name=agent_name,
            agent_version="0.1.0",
            default_stage="system",
        )
        trace_id = emitter.start_trace()
        trace_path = run_logs_dir / f"{trace_id}.jsonl"
        emitter.file_path = trace_path

        # Run
        with emitter:
            result_obj = runner(task, inputs, emitter)
        # Validate trace
        issues = validate_trace_file(trace_path)
        if issues:

            notes = [f"trace invalid: {i.code} {i.message}" for i in issues]
            events = read_jsonl(trace_path)
            cost_usd, latency_ms, tokens = _extract_tradeoff_metrics(events)
            input_tokens = None
            output_tokens = None
            total_tokens = None
            case_results.append(
                CaseResult(
                    case_id=case_id,
                    passed=False,
                    score=0.0,
                    trace_id=trace_id,
                    trace_path=str(trace_path),
                    terminal_status=None,
                    terminal_reason=None,
                    notes=notes,
                )
            )

            violations = [f"trace_invalid:{i.code}" for i in issues]
            rationale = "; ".join([f"{i.code} {i.message}" for i in issues])[:300]  # short

            rewards.append(
                RewardV0(
                    version="reward.v0",
                    trace_id=trace_id,
                    run_id=run_id,
                    suite_id=suite_name,
                    case_id=case_id,
                    agent_id=agent_name,
                    model_id=model_id,
                    commit_sha=commit_sha,
                    success=False,
                    overall_score=0.0,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    subscores={},
                    violations=violations,
                    terminal_status=None,
                    terminal_reason=None,
                    rationale=rationale,
                )
            )

            continue

        # Inspect terminal outcome
        events = read_jsonl(trace_path)
        summary = summarise_trace(events, path=trace_path)

        cost_usd, latency_ms, tokens = _extract_tradeoff_metrics(events)
        input_tokens = tokens.get("input_tokens")
        output_tokens = tokens.get("output_tokens")
        total_tokens = tokens.get("total_tokens")

        terminal = summary.terminal_outcome or {}
        terminal_status = terminal.get("status")
        terminal_reason = terminal.get("reason")

        # Extract a result text for basic expectation checks
        result_text = _extract_result_text(agent_name, result_obj)

        ctx = ScoringContext(
            suite_name=suite_name,
            case=case,
            result_obj=result_obj,
            terminal_outcome=terminal,
            result_text=result_text,
            events=events,
            scoring_params=scoring_params,
        )
        try:
            passed, score, notes = scorer(ctx)
        except Exception as exc:
            passed = False
            score = 0.0
            notes = [f"scoring_failed: {type(exc).__name__}: {exc}"]

        violations = []
        # make evaluation reasons machine-readable
        for n in notes:
            if n.startswith("status mismatch"):
                violations.append("expectation:status_mismatch")
            if n.startswith("result missing any of"):
                violations.append("expectation:contains_any_missing")
            if n.startswith("scoring_failed"):
                violations.append("scoring_failed")

        rewards.append(
            RewardV0(
                version="reward.v0",
                trace_id=trace_id,
                run_id=run_id,
                suite_id=suite_name,
                case_id=case_id,
                agent_id=agent_name,
                model_id=model_id,
                commit_sha=commit_sha,
                success=passed,
                overall_score=score,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                subscores={
                    # optional decomposition now, can refine later
                    "status": 1.0 if expect.get("status") is None or terminal_status == expect.get("status") else 0.0,
                    "contains_any": 1.0 if not expect.get("contains_any") or any(
                        t.lower() in (result_text or "").lower() for t in (expect.get("contains_any") or [])
                    ) else 0.0,
                },
                violations=violations,
                terminal_status=str(terminal_status) if terminal_status is not None else None,
                terminal_reason=str(terminal_reason) if terminal_reason is not None else None,
                rationale=("; ".join(notes)[:300] if notes else None),
            )
        )

        case_results.append(
            CaseResult(
                case_id=case_id,
                passed=passed,
                score=score,
                trace_id=trace_id,
                trace_path=str(trace_path),
                terminal_status=str(terminal_status) if terminal_status is not None else None,
                terminal_reason=str(terminal_reason) if terminal_reason is not None else None,
                notes=notes,
            )
        )

    passed_n = sum(1 for r in case_results if r.passed)
    failed_n = len(case_results) - passed_n
    pass_rate = passed_n / len(case_results) if case_results else 0.0

    suite_result = SuiteResult(
        suite_name=suite_name,
        agent=agent_name,
        run_id=run_id,
        run_logs_dir=str(run_logs_dir),
        num_cases=len(case_results),
        passed=passed_n,
        failed=failed_n,
        pass_rate=pass_rate,
        results=case_results,
    )

    # Write outputs
    (output_dir / f"{suite_name}.results.json").write_text(
        json.dumps(asdict(suite_result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"{suite_name}.summary.md").write_text(
        _render_summary_md(suite_result),
        encoding="utf-8",
    )

    (run_logs_dir / "reward.summary.json").write_text(
    json.dumps(
        {
            "run_id": run_id,
            "suite_id": suite_name,
            "agent_id": agent_name,
            "model_id": model_id,
            "commit_sha": commit_sha,
            "num_cases": len(rewards),
            "mean_score": sum(r.overall_score for r in rewards) / len(rewards) if rewards else 0.0,
            "success_rate": sum(1 for r in rewards if r.success) / len(rewards) if rewards else 0.0,
        },
        indent=2,
    ),
    encoding="utf-8",
)

    write_rewards_jsonl(run_logs_dir / "reward.jsonl", rewards)

    return suite_result


def _render_summary_md(s: SuiteResult) -> str:
    """Render a Markdown summary for a suite result."""
    lines = [
        f"# Suite: {s.suite_name}",
        "",
        f"- Agent: `{s.agent}`",
        f"- Cases: {s.num_cases}",
        f"- Passed: {s.passed}",
        f"- Failed: {s.failed}",
        f"- Pass rate: {s.pass_rate:.2%}",
        "",
        "## Results",
        "",
        "| case_id | passed | score | trace_id | terminal_status | notes |",
        "|---|---:|---:|---|---|---|",
    ]
    for r in s.results:
        notes = "; ".join(r.notes) if r.notes else ""
        lines.append(
            f"| {r.case_id} | {str(r.passed).lower()} | {r.score:.2f} | `{r.trace_id}` | {r.terminal_status or ''} | {notes} |"
        )
    lines.append("")
    return "\n".join(lines)


def _extract_result_text(agent_name: str, result_obj: Dict[str, Any]) -> str:
    """Heuristic extraction used for `contains_any` checks."""
    if agent_name == "decision_agent":
        memo = result_obj.get("memo", {})
        if isinstance(memo, dict):
            return " ".join(memo.get("next_steps", []) or [])
    if agent_name == "research_agent":
        return str(result_obj.get("summary", "")) or str(result_obj.get("result", ""))
    
    if agent_name == "refactor_agent":
        pr = result_obj.get("patch_result", {})
        if isinstance(pr, dict):
            return str(pr.get("diff_summary", "")) + " " + str(pr.get("reason", ""))

    return str(result_obj)


# --- Agent registry ---

def get_agent_runner(agent_name: str):
    """Return the runner function for a known agent name."""
    if agent_name == "decision_agent":
        return _run_decision_agent
    if agent_name == "research_agent":
        return _run_research_agent
    if agent_name == "refactor_agent":
        return _run_refactor_agent
    if agent_name == "synth_agent":
        return _run_synth_agent

    raise ValueError(f"Unknown agent: {agent_name!r}")

def _run_synth_agent(task: str, inputs: Dict[str, Any], emitter: JsonlEmitter) -> Dict[str, Any]:
    from tensorfoundry.agents.synth_agent import SynthAgent
    agent = SynthAgent(emitter=emitter)
    return agent.run(task, sources=inputs.get("sources", []))


def _run_decision_agent(task: str, inputs: Dict[str, Any], emitter: JsonlEmitter) -> Dict[str, Any]:
    """Run the decision agent with inputs from a suite case."""
    from tensorfoundry.agents.decision_agent import DecisionAgent

    agent = DecisionAgent(emitter=emitter)
    return agent.run(
        task,
        constraints=inputs.get("constraints"),
        options=inputs.get("options"),
    )


def _run_research_agent(task: str, inputs: Dict[str, Any], emitter: JsonlEmitter) -> Dict[str, Any]:
    """Run the research agent with inputs from a suite case."""
    from tensorfoundry.agents.research_agent import ResearchAgent

    agent = ResearchAgent(emitter=emitter)
    return agent.run(task)

def _run_refactor_agent(task: str, inputs: Dict[str, Any], emitter: JsonlEmitter) -> Dict[str, Any]:
    """Run the refactor agent with inputs from a suite case."""
    from tensorfoundry.agents.refactor_agent import RefactorAgent

    agent = RefactorAgent(emitter=emitter)
    return agent.run(
        task,
        repo_root=inputs.get("repo_root", "."),
        target_file=inputs.get("target_file"),
        find=inputs.get("find", ""),
        replace=inputs.get("replace", ""),
        dry_run=bool(inputs.get("dry_run", True)),
    )
