from __future__ import annotations

import json
from pathlib import Path

from tensorfoundry.evaluation.report_tradeoffs import main


def test_report_tradeoffs_renders_mean_effective(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "logs" / "run-1"
    run_dir.mkdir(parents=True)
    row = {
        "version": "reward.v0",
        "suite_id": "suite",
        "model_id": "model-a",
        "overall_score": 0.8,
        "success": True,
        "cost_usd": 0.1,
        "latency_ms": 100,
        "total_tokens": 12,
    }
    (run_dir / "reward.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert main([str(tmp_path / "logs")]) == 0
    out = capsys.readouterr().out

    assert "mean_effective" in out
    assert "| model-a | 1 | 0.800 | 0.100000 | 100 | 12 | 100.00% | 0.800 |" in out
