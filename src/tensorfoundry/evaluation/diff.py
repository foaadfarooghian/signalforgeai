"""Diff two suite results JSON files produced by the evaluation harness.

Usage:
  python -m tensorfoundry.evaluation.diff results/a.results.json results/b.results.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ResultsDiff:
    left: str
    right: str
    suite_left: str
    suite_right: str
    pass_rate_left: float
    pass_rate_right: float
    pass_rate_delta: float
    passed_left: int
    passed_right: int
    failed_left: int
    failed_right: int
    changed_cases: List[str]
    regressed_cases: List[str]
    improved_cases: List[str]


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def diff_results(left_path: Path, right_path: Path) -> ResultsDiff:
    left = _load(left_path)
    right = _load(right_path)

    left_cases = {r["case_id"]: r for r in left["results"]}
    right_cases = {r["case_id"]: r for r in right["results"]}
    all_ids = sorted(set(left_cases) | set(right_cases))

    changed: List[str] = []
    regressed: List[str] = []
    improved: List[str] = []

    for cid in all_ids:
        left_case = left_cases.get(cid)
        right_case = right_cases.get(cid)
        if left_case is None or right_case is None:
            changed.append(cid)
            continue

        lp = bool(left_case["passed"])
        rp = bool(right_case["passed"])
        if lp != rp or float(left_case.get("score", 0.0)) != float(right_case.get("score", 0.0)):
            changed.append(cid)
        if lp and not rp:
            regressed.append(cid)
        if not lp and rp:
            improved.append(cid)

    pr_left = float(left["pass_rate"])
    pr_right = float(right["pass_rate"])

    return ResultsDiff(
        left=str(left_path),
        right=str(right_path),
        suite_left=str(left["suite_name"]),
        suite_right=str(right["suite_name"]),
        pass_rate_left=pr_left,
        pass_rate_right=pr_right,
        pass_rate_delta=pr_right - pr_left,
        passed_left=int(left["passed"]),
        passed_right=int(right["passed"]),
        failed_left=int(left["failed"]),
        failed_right=int(right["failed"]),
        changed_cases=changed,
        regressed_cases=regressed,
        improved_cases=improved,
    )


def format_results_diff(d: ResultsDiff) -> str:
    lines: List[str] = []
    lines.append(f"Left:  {d.left} (suite={d.suite_left})")
    lines.append(f"Right: {d.right} (suite={d.suite_right})")
    lines.append("")
    lines.append(
        f"Pass rate: {d.pass_rate_left:.2%} → {d.pass_rate_right:.2%} (delta {d.pass_rate_delta:+.2%})"
    )
    lines.append(f"Passed: {d.passed_left} → {d.passed_right}")
    lines.append(f"Failed: {d.failed_left} → {d.failed_right}")
    lines.append("")
    if d.regressed_cases:
        lines.append("Regressions:")
        for c in d.regressed_cases:
            lines.append(f"  - {c}")
        lines.append("")
    if d.improved_cases:
        lines.append("Improvements:")
        for c in d.improved_cases:
            lines.append(f"  - {c}")
        lines.append("")
    if d.changed_cases:
        lines.append("Changed cases:")
        for c in d.changed_cases:
            lines.append(f"  - {c}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Diff two TensorFoundry evaluation results files.")
    p.add_argument("left", type=str)
    p.add_argument("right", type=str)
    args = p.parse_args(argv)

    d = diff_results(Path(args.left), Path(args.right))
    print(format_results_diff(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
