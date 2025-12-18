from __future__ import annotations

import json
from pathlib import Path

from tensorfoundry.evaluation.harness import run_suite


def test_eval_quickstart_runs_and_writes_outputs(tmp_path: Path) -> None:
    # Create a minimal suite in temp
    suite = {
        "suite_name": "tmp_quickstart",
        "agent": "decision_agent",
        "cases": [
            {
                "id": "dec_001",
                "task": "What should we build next?",
                "inputs": {
                    "constraints": ["Keep it simple"],
                    "options": ["Build eval", "Build diff"],
                },
                "expect": {"status": "success", "contains_any": ["build", "eval", "diff"]},
            }
        ],
    }

    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")

    out_dir = tmp_path / "results"
    logs_dir = tmp_path / "logs"

    res = run_suite(suite_path=suite_path, output_dir=out_dir, logs_dir=logs_dir)

    assert res.num_cases == 1
    assert (out_dir / "tmp_quickstart.results.json").exists()
    assert (out_dir / "tmp_quickstart.summary.md").exists()

    # Should create at least one trace file
    assert any(p.suffix == ".jsonl" for p in logs_dir.iterdir())