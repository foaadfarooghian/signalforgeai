"""Validation utilities for TensorFoundry JSONL traces.

Usage:
  python -m tensorfoundry.logging.validate path/to/trace.jsonl
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


VALID_STAGES: Set[str] = {"planner", "executor", "critic", "tool", "system"}

# Minimum top-level keys required by schema.md
REQUIRED_KEYS: Set[str] = {
    "trace_id",
    "span_id",
    "parent_span_id",
    "timestamp",
    "agent",
    "stage",
    "event_type",
    "payload",
    "metrics",
    "outcome",
}

TERMINAL_EVENT_TYPES: Set[str] = {"task_completed", "task_failed"}

REWARD_REQUIRED_KEYS: Set[str] = {
    "version",
    "trace_id",
    "run_id",
    "suite_id",
    "case_id",
    "agent_id",
    "model_id",
    "commit_sha",
    "success",
    "overall_score",
    "subscores",
    "violations",
    "created_at",
}

def validate_reward_events(events: Iterable[Dict[str, Any]]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    for i, ev in enumerate(events, start=1):
        missing = REWARD_REQUIRED_KEYS.difference(ev.keys())
        if missing:
            issues.append(
                ValidationIssue(i, "reward_missing_keys", f"Missing required keys: {sorted(missing)}")
            )
            continue

        # version
        if ev.get("version") != "reward.v0":
            issues.append(
                ValidationIssue(i, "reward_bad_version", f"Unsupported reward version: {ev.get('version')!r}")
            )

        # types
        if not _is_str(ev.get("trace_id")):
            issues.append(ValidationIssue(i, "reward_bad_type", "`trace_id` must be a string."))
        if not _is_str(ev.get("run_id")):
            issues.append(ValidationIssue(i, "reward_bad_type", "`run_id` must be a string."))
        if not _is_str(ev.get("suite_id")):
            issues.append(ValidationIssue(i, "reward_bad_type", "`suite_id` must be a string."))
        if not _is_str(ev.get("case_id")):
            issues.append(ValidationIssue(i, "reward_bad_type", "`case_id` must be a string."))
        if not _is_str(ev.get("agent_id")):
            issues.append(ValidationIssue(i, "reward_bad_type", "`agent_id` must be a string."))
        if not _is_str(ev.get("model_id")):
            issues.append(ValidationIssue(i, "reward_bad_type", "`model_id` must be a string."))
        if not _is_str(ev.get("commit_sha")):
            issues.append(ValidationIssue(i, "reward_bad_type", "`commit_sha` must be a string."))

        if not isinstance(ev.get("success"), bool):
            issues.append(ValidationIssue(i, "reward_bad_type", "`success` must be a boolean."))

        # score 0..1
        score = ev.get("overall_score")
        if not isinstance(score, (int, float)):
            issues.append(ValidationIssue(i, "reward_bad_type", "`overall_score` must be a number."))
        else:
            s = float(score)
            if not (0.0 <= s <= 1.0):
                issues.append(ValidationIssue(i, "reward_score_out_of_range", f"`overall_score` out of range: {s}"))

        # subscores dict of numbers 0..1
        subs = ev.get("subscores")
        if not isinstance(subs, dict):
            issues.append(ValidationIssue(i, "reward_bad_type", "`subscores` must be an object."))
        else:
            for k, v in subs.items():
                if not _is_str(k):
                    issues.append(ValidationIssue(i, "reward_bad_type", "subscore keys must be strings."))
                    continue
                if not isinstance(v, (int, float)):
                    issues.append(ValidationIssue(i, "reward_bad_type", f"subscore {k!r} must be a number."))
                    continue
                vv = float(v)
                if not (0.0 <= vv <= 1.0):
                    issues.append(
                        ValidationIssue(i, "reward_score_out_of_range", f"subscore {k!r} out of range: {vv}")
                    )

        # violations list[str]
        viol = ev.get("violations")
        if not isinstance(viol, list) or any(not _is_str(x) for x in viol):
            issues.append(ValidationIssue(i, "reward_bad_type", "`violations` must be a list of strings."))

        # created_at
        if not _is_str(ev.get("created_at")):
            issues.append(ValidationIssue(i, "reward_bad_type", "`created_at` must be a string."))

    return issues

def _detect_kind(events: List[Dict[str, Any]]) -> str:
    # very lightweight heuristics
    if not events:
        return "unknown"
    first = events[0]
    if "event_type" in first and "agent" in first and "stage" in first:
        return "trace"
    if first.get("version") == "reward.v0" and "overall_score" in first:
        return "reward"
    return "unknown"

@dataclass(frozen=True)
class ValidationIssue:
    line_no: int
    code: str
    message: str

    def __str__(self) -> str:
        return f"line {self.line_no}: [{self.code}] {self.message}"


def _is_dict(x: Any) -> bool:
    return isinstance(x, dict)


def _is_str(x: Any) -> bool:
    return isinstance(x, str)


def _read_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], List[ValidationIssue]]:
    issues: List[ValidationIssue] = []
    events: List[Dict[str, Any]] = []

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [], [ValidationIssue(0, "file_not_found", f"File not found: {path}")]
    except OSError as e:
        return [], [ValidationIssue(0, "file_read_error", f"Could not read file: {e}")]

    if not text.strip():
        return [], [ValidationIssue(0, "empty_file", "Trace file is empty.")]

    for idx, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            issues.append(ValidationIssue(idx, "invalid_json", f"Invalid JSON: {e}"))
            continue

        if not _is_dict(obj):
            issues.append(ValidationIssue(idx, "not_object", "Each line must be a JSON object."))
            continue

        events.append(obj)

    if not events and not issues:
        issues.append(ValidationIssue(0, "no_events", "No events found (file may contain only blank lines)."))

    return events, issues


def validate_events(
    events: Iterable[Dict[str, Any]],
    *,
    require_single_trace_id: bool = True,
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    trace_ids: Set[str] = set()
    saw_terminal = False

    for i, ev in enumerate(events, start=1):
        # required keys
        missing = REQUIRED_KEYS.difference(ev.keys())
        if missing:
            issues.append(
                ValidationIssue(
                    i,
                    "missing_keys",
                    f"Missing required keys: {sorted(missing)}",
                )
            )
            continue  # can't do much else reliably

        # type checks (lightweight)
        if not _is_str(ev["trace_id"]):
            issues.append(ValidationIssue(i, "bad_type", "`trace_id` must be a string."))
        if not _is_str(ev["span_id"]):
            issues.append(ValidationIssue(i, "bad_type", "`span_id` must be a string."))
        # parent_span_id can be str or None
        if ev["parent_span_id"] is not None and not _is_str(ev["parent_span_id"]):
            issues.append(
                ValidationIssue(i, "bad_type", "`parent_span_id` must be a string or null.")
            )

        # agent must be an object with name/version strings
        agent = ev["agent"]
        if not _is_dict(agent):
            issues.append(ValidationIssue(i, "bad_type", "`agent` must be an object."))
        else:
            if "name" not in agent or "version" not in agent:
                issues.append(
                    ValidationIssue(i, "missing_keys", "`agent` must include `name` and `version`.")
                )
            else:
                if not _is_str(agent.get("name")):
                    issues.append(ValidationIssue(i, "bad_type", "`agent.name` must be a string."))
                if not _is_str(agent.get("version")):
                    issues.append(
                        ValidationIssue(i, "bad_type", "`agent.version` must be a string.")
                    )

        # stage validation
        stage = ev["stage"]
        if not _is_str(stage):
            issues.append(ValidationIssue(i, "bad_type", "`stage` must be a string."))
        elif stage not in VALID_STAGES:
            issues.append(
                ValidationIssue(
                    i,
                    "invalid_stage",
                    f"Invalid stage {stage!r}. Must be one of {sorted(VALID_STAGES)}",
                )
            )

        # event_type string
        if not _is_str(ev["event_type"]):
            issues.append(ValidationIssue(i, "bad_type", "`event_type` must be a string."))

        # payload/metrics/outcome must be objects
        for key in ("payload", "metrics", "outcome"):
            if not _is_dict(ev[key]):
                issues.append(ValidationIssue(i, "bad_type", f"`{key}` must be an object."))

        # collect trace IDs
        if _is_str(ev["trace_id"]):
            trace_ids.add(ev["trace_id"])

        # terminal event present?
        if ev.get("event_type") in TERMINAL_EVENT_TYPES:
            out = ev.get("outcome", {})
            # ensure terminal events include outcome.status (per schema expectation)
            status = out.get("status") if isinstance(out, dict) else None
            if not _is_str(status):
                issues.append(
                    ValidationIssue(
                        i,
                        "missing_terminal_status",
                        "Terminal event must include `outcome.status` as a string.",
                    )
                )
            saw_terminal = True

    if require_single_trace_id and len(trace_ids) > 1:
        issues.append(
            ValidationIssue(
                0,
                "multiple_trace_ids",
                f"File contains multiple trace_ids: {sorted(trace_ids)}",
            )
        )

    if not saw_terminal:
        issues.append(
            ValidationIssue(
                0,
                "missing_terminal_event",
                f"Trace must include a terminal event: {sorted(TERMINAL_EVENT_TYPES)}",
            )
        )

    return issues

def validate_jsonl_file(
    path: Path | str,
    *,
    require_single_trace_id: bool = True,
) -> List[ValidationIssue]:
    p = Path(path)
    events, parse_issues = _read_jsonl(p)
    if parse_issues:
        return parse_issues

    kind = _detect_kind(events)
    if kind == "trace":
        return validate_events(events, require_single_trace_id=require_single_trace_id)
    if kind == "reward":
        return validate_reward_events(events)

    return [ValidationIssue(0, "unknown_jsonl_kind", "Unrecognized JSONL format (not trace, not reward.v0).")]

def validate_trace_file(
    path: Path | str,
    *,
    require_single_trace_id: bool = True,
) -> List[ValidationIssue]:
    return validate_jsonl_file(path, require_single_trace_id=require_single_trace_id)


def _format_issues(issues: List[ValidationIssue]) -> str:
    if not issues:
        return "OK: trace is valid.\n"
    lines = ["INVALID: trace has issues:"]
    lines.extend([f"- {issue}" for issue in issues])
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a TensorFoundry JSONL trace.")
    parser.add_argument("path", type=str, help="Path to a .jsonl file OR a directory containing .jsonl files")
    parser.add_argument(
        "--allow-multiple-traces",
        action="store_true",
        help="Allow more than one trace_id per file (not recommended).",
    )
    args = parser.parse_args(argv)

    target = Path(args.path)

    all_issues: List[ValidationIssue] = []

    if target.is_dir():
        jsonl_files = sorted(target.glob("*.jsonl"))
        if not jsonl_files:
            all_issues.append(ValidationIssue(0, "no_jsonl_files", f"No .jsonl files found in directory: {target}"))
        for f in jsonl_files:
            issues = validate_jsonl_file(
                f,
                require_single_trace_id=not args.allow_multiple_traces,
            )
            if issues:
                # namespace issues with file context
                all_issues.append(ValidationIssue(0, "file", f"{f.name}"))
                all_issues.extend(issues)
    else:
        all_issues = validate_jsonl_file(
            target,
            require_single_trace_id=not args.allow_multiple_traces,
        )

    print(_format_issues(all_issues), end="")
    return 0 if not all_issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
