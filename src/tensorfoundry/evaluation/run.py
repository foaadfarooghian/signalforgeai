"""CLI entrypoint for running evaluation suites."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from tensorfoundry.evaluation.harness import run_suite


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a TensorFoundry evaluation suite.")
    parser.add_argument("suite", type=str, help="Path to suite JSON (e.g. src/.../suites/quickstart.json)")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for results outputs")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Directory for trace logs")
    args = parser.parse_args(argv)

    result = run_suite(
        suite_path=Path(args.suite),
        output_dir=Path(args.output_dir),
        logs_dir=Path(args.logs_dir),
    )

    print(f"Suite {result.suite_name!r} finished: pass_rate={result.pass_rate:.2%} ({result.passed}/{result.num_cases})")
    print(f"Run logs: {result.run_logs_dir}")
    print(f"Results: {Path(args.output_dir) / (result.suite_name + '.results.json')}")
    print(f"Summary:  {Path(args.output_dir) / (result.suite_name + '.summary.md')}")
    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())