"""Evaluation harness for TensorFoundry agents.

Runs a suite of task cases, captures traces, validates them, and scores outcomes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tensorfoundry.logging.emitter import JsonlEmitter
from tensorfoundry.logging.inspect import summarise_trace, read_jsonl
from tensorfoundry.logging.validate import validate_trace_file


@dataclass(frozen=True)
class CaseResult:
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
    suite_name: str
    agent: str
    num_cases: int
    passed: int
    failed: int
    pass_rate: float
    results: List[CaseResult]


def _load_suite(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _score_case(
    *,
    expect: Dict[str, Any],
    terminal_outcome: Dict[str, Any],
    result_text: str,
) -> Tuple[bool, float, List[str]]:
    """Simple scoring: status match + contains_any check."""
    notes: List[str] = []
    score = 0.0

    expected_status = expect.get("status")
    actual_status = terminal_outcome.get("status")

    if expected_status is not None:
        if actual_status == expected_status:
            score += 0.7
        else:
            notes.append(f"status mismatch: expected {expected_status!r}, got {actual_status!r}")

    contains_any = expect.get("contains_any", [])
    if contains_any:
        if any(token.lower() in (result_text or "").lower() for token in contains_any):
            score += 0.3
        else:
            notes.append(f"result missing any of: {contains_any}")

    passed = score >= 0.7  # status match is enough to pass, contains adds confidence
    return passed, score, notes


def run_suite(
    *,
    suite_path: Path | str,
    output_dir: Path | str = Path("results"),
    logs_dir: Path | str = Path("logs"),
) -> SuiteResult:
    suite_path = Path(suite_path)
    output_dir = Path(output_dir)
    logs_dir = Path(logs_dir)

    suite = _load_suite(suite_path)
    suite_name: str = suite["suite_name"]
    agent_name: str = suite["agent"]
    cases: List[Dict[str, Any]] = suite["cases"]

    _ensure_dir(output_dir)
    _ensure_dir(logs_dir)

    runner = get_agent_runner(agent_name)

    case_results: List[CaseResult] = []

    for case in cases:
        case_id = case["id"]
        task = case["task"]
        inputs = case.get("inputs", {})
        expect = case.get("expect", {})

        # Create per-case trace file
        emitter = JsonlEmitter(
            logs_dir / "temp.jsonl",
            agent_name=agent_name,
            agent_version="0.1.0",
            default_stage="system",
        )
        trace_id = emitter.start_trace()
        trace_path = logs_dir / f"{trace_id}.jsonl"
        emitter.file_path = trace_path

        # Run
        with emitter:
            result_obj = runner(task, inputs, emitter)

        # Validate trace
        issues = validate_trace_file(trace_path)
        if issues:
            notes = [f"trace invalid: {i.code} {i.message}" for i in issues]
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
            continue

        # Inspect terminal outcome
        events = read_jsonl(trace_path)
        summary = summarise_trace(events, path=trace_path)
        terminal = summary.terminal_outcome or {}
        terminal_status = terminal.get("status")
        terminal_reason = terminal.get("reason")

        # Extract a result text for basic expectation checks
        result_text = _extract_result_text(agent_name, result_obj)

        passed, score, notes = _score_case(
            expect=expect,
            terminal_outcome=terminal,
            result_text=result_text,
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

    return suite_result


def _render_summary_md(s: SuiteResult) -> str:
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
            return str(memo.get("recommendation", "")) + " " + " ".join(memo.get("next_steps", []) or [])
    if agent_name == "research_agent":
        return str(result_obj.get("summary", "")) or str(result_obj.get("result", ""))
    return str(result_obj)


# --- Agent registry ---

def get_agent_runner(agent_name: str):
    if agent_name == "decision_agent":
        return _run_decision_agent
    if agent_name == "research_agent":
        return _run_research_agent
    raise ValueError(f"Unknown agent: {agent_name!r}")


def _run_decision_agent(task: str, inputs: Dict[str, Any], emitter: JsonlEmitter) -> Dict[str, Any]:
    from tensorfoundry.agents.decision_agent import DecisionAgent

    agent = DecisionAgent(emitter=emitter)
    return agent.run(
        task,
        constraints=inputs.get("constraints"),
        options=inputs.get("options"),
    )


def _run_research_agent(task: str, inputs: Dict[str, Any], emitter: JsonlEmitter) -> Dict[str, Any]:
    from tensorfoundry.agents.research_agent import ResearchAgent

    agent = ResearchAgent(emitter=emitter)
    return agent.run(task)