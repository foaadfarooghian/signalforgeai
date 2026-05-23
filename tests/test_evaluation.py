from __future__ import annotations

import json
from pathlib import Path

from signalforgeai.evaluation import harness
from signalforgeai.evaluation.harness import _extract_tradeoff_metrics, run_suite
from signalforgeai.evaluation.scorers import ScoringContext, synth_v1
from signalforgeai.logging.validate import ValidationIssue


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

    # Should create at least one trace file (now under logs/<run_id>/)
    run_logs_dir = Path(res.run_logs_dir)
    assert run_logs_dir.exists()
    assert any(p.suffix == ".jsonl" for p in run_logs_dir.iterdir())
    assert (run_logs_dir / "reward.jsonl").exists()


def test_score_synth_case_conflict_success() -> None:
    sources = [
        {"source_id": "S1", "title": "Source 1", "text": "Alpha is 10."},
        {"source_id": "S2", "title": "Source 2", "text": "Alpha is 12."},
    ]
    result_obj = {
        "result": {
            "answer": (
                "Sources conflict: one says Alpha is 10 while another says Alpha is 12. "
                "This is uncertain and should be treated as conflicting evidence."
            ),
            "citations": [
                {"source_id": "S1", "quote": "Alpha is 10."},
                {"source_id": "S2", "quote": "Alpha is 12."},
            ],
            "assumptions": [],
            "risks": [],
            "confidence": 0.4,
        }
    }
    ctx = ScoringContext(
        suite_name="benchmark_v1_synth",
        case={"id": "case_conflict_001", "inputs": {"sources": sources}},
        result_obj=result_obj,
        terminal_outcome={},
        result_text="",
        events=[],
    )
    passed, score, notes = synth_v1(ctx)
    assert passed is True
    assert score >= 0.7
    assert notes == []


def test_score_synth_case_flags_bad_quotes() -> None:
    sources = [{"source_id": "S1", "title": "Doc", "text": "Alpha is 10."}]
    result_obj = {
        "result": {
            "answer": "Alpha is 10, but this citation is malformed and not grounded.",
            "citations": [{"source_id": "S1", "quote": "[S1]"}],
            "assumptions": [],
            "risks": [],
            "confidence": 0.2,
        }
    }
    ctx = ScoringContext(
        suite_name="benchmark_v1_synth",
        case={"id": "case_basic", "inputs": {"sources": sources}},
        result_obj=result_obj,
        terminal_outcome={},
        result_text="",
        events=[],
    )
    passed, _score, notes = synth_v1(ctx)
    assert passed is False
    assert "placeholder_quotes" in notes
    assert "ungrounded_or_empty_quotes" in notes


def test_extract_tradeoff_metrics_sums_cost_latency_tokens() -> None:
    events = [
        {
            "event_type": "model_called",
            "metrics": {
                "cost_usd": 0.1,
                "latency_ms": 100,
                "extra": {"usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}},
            },
        },
        {
            "event_type": "model_called",
            "metrics": {"cost_usd": 0.2, "latency_ms": 50, "input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        },
        {"event_type": "tool_called", "metrics": {"cost_usd": 99}},
    ]
    cost_usd, latency_ms, tokens = _extract_tradeoff_metrics(events)
    assert cost_usd == 0.3
    assert latency_ms == 150
    assert tokens == {"input_tokens": 8, "output_tokens": 11, "total_tokens": 19}


def test_run_suite_records_invalid_trace(monkeypatch, tmp_path: Path) -> None:
    suite = {
        "suite_name": "tmp_invalid_trace",
        "agent": "decision_agent",
        "cases": [
            {
                "id": "dec_001",
                "task": "What should we build next?",
                "inputs": {"constraints": [], "options": ["Build eval"]},
                "expect": {"status": "success", "contains_any": ["build", "eval"]},
            }
        ],
    }
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")

    monkeypatch.setenv("SIGNALFORGEAI_MODEL_ID", "dummy_good")
    monkeypatch.setattr(
        harness,
        "validate_trace_file",
        lambda _p: [ValidationIssue(1, "bad_trace", "broken trace")],
    )

    out_dir = tmp_path / "results"
    logs_dir = tmp_path / "logs"
    res = run_suite(suite_path=suite_path, output_dir=out_dir, logs_dir=logs_dir)

    assert res.failed == 1
    assert res.results[0].notes and res.results[0].notes[0].startswith("trace invalid:")

    reward_path = Path(res.run_logs_dir) / "reward.jsonl"
    reward_row = json.loads(reward_path.read_text(encoding="utf-8").splitlines()[0])
    assert "trace_invalid:bad_trace" in reward_row.get("violations", [])


def test_run_suite_records_trace_failure_origin(monkeypatch, tmp_path: Path) -> None:
    suite = {
        "suite_name": "tmp_runner_failure",
        "agent": "decision_agent",
        "cases": [
            {
                "id": "dec_001",
                "task": "What should we build next?",
                "inputs": {"constraints": [], "options": ["Build eval"]},
                "expect": {"status": "success", "contains_any": ["build", "eval"]},
            }
        ],
    }
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")

    def _boom(_task, _inputs, _emitter):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(harness, "get_agent_runner", lambda _agent_name: _boom)

    res = run_suite(
        suite_path=suite_path,
        output_dir=tmp_path / "results",
        logs_dir=tmp_path / "logs",
    )

    assert res.failed == 1
    reward_path = Path(res.run_logs_dir) / "reward.jsonl"
    reward_row = json.loads(reward_path.read_text(encoding="utf-8").splitlines()[0])
    diagnosis = reward_row["diagnosis"]
    assert diagnosis["failure_origin"]["event_type"] == "task_failed"
    assert diagnosis["failure_origin"]["stage"] == "system"
    assert diagnosis["recovery"]["terminal_status"] == "failure"
