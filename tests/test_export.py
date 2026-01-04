from __future__ import annotations

import json
import random
from pathlib import Path

from tensorfoundry.export.dataset import export_sft
from tensorfoundry.export.preferences import export_preferences
from tensorfoundry.export.repairs import export_repairs
from tensorfoundry.learning.curriculum import export_curriculum


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _trace_events(task: str, prompt: str, response: str, *, step: str = "critic_check") -> list[dict]:
    return [
        {"event_type": "task_received", "payload": {"task": task}},
        {
            "event_type": "model_called",
            "payload": {"input_summary": step},
            "outcome": {"result": {"text_full": response, "prompt_full": prompt}},
        },
    ]


def _reward_row(
    *,
    trace_id: str,
    suite_id: str,
    case_id: str,
    model_id: str,
    overall_score: float,
    success: bool,
    violations: list[str] | None = None,
) -> dict:
    return {
        "version": "reward.v0",
        "trace_id": trace_id,
        "suite_id": suite_id,
        "case_id": case_id,
        "model_id": model_id,
        "overall_score": overall_score,
        "success": success,
        "violations": violations or [],
    }


def test_export_sft_writes_row(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs"
    run_dir = logs_root / "run1"
    trace_id = "t1"
    prompt = "Prompt for SFT"
    response = "This is a response long enough for SFT export."

    _write_jsonl(run_dir / f"{trace_id}.jsonl", _trace_events("do the thing", prompt, response))
    _write_jsonl(
        run_dir / "reward.jsonl",
        [
            _reward_row(
                trace_id=trace_id,
                suite_id="s1",
                case_id="c1",
                model_id="ollama:model_a",
                overall_score=0.9,
                success=True,
            )
        ],
    )

    out_path = tmp_path / "out" / "sft.jsonl"
    written, skipped = export_sft(
        logs_root=logs_root,
        out_path=out_path,
        suite=None,
        min_score=0.7,
        success_only=True,
        limit=None,
        include_prefixes=[],
        exclude_prefixes=[],
        exclude_model_ids=set(),
        step="critic_check",
        use_step_prompt=False,
        min_response_chars=10,
    )

    assert written == 1
    assert skipped == 0
    row = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["instruction"].startswith("Task:")
    assert row["response"] == response
    assert row["prompt"] == row["instruction"]


def test_export_sft_skips_short_response(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs"
    run_dir = logs_root / "run1"
    trace_id = "t1"
    prompt = "Prompt for SFT"
    response = "short"

    _write_jsonl(run_dir / f"{trace_id}.jsonl", _trace_events("do the thing", prompt, response))
    _write_jsonl(
        run_dir / "reward.jsonl",
        [
            _reward_row(
                trace_id=trace_id,
                suite_id="s1",
                case_id="c1",
                model_id="ollama:model_a",
                overall_score=0.9,
                success=True,
            )
        ],
    )

    out_path = tmp_path / "out" / "sft.jsonl"
    written, skipped = export_sft(
        logs_root=logs_root,
        out_path=out_path,
        suite=None,
        min_score=0.7,
        success_only=True,
        limit=None,
        include_prefixes=[],
        exclude_prefixes=[],
        exclude_model_ids=set(),
        step="critic_check",
        use_step_prompt=False,
        min_response_chars=10,
    )

    assert written == 0
    assert skipped == 1
    assert out_path.read_text(encoding="utf-8") == ""


def test_export_preferences_pairs_best_vs_other(tmp_path: Path) -> None:
    random.seed(0)
    logs_root = tmp_path / "logs"
    run_dir = logs_root / "run1"
    prompt = "Prompt for prefs"
    response_a = "Response A"
    response_b = "Response B"

    _write_jsonl(run_dir / "ta.jsonl", _trace_events("task", prompt, response_a))
    _write_jsonl(run_dir / "tb.jsonl", _trace_events("task", prompt, response_b))
    _write_jsonl(
        run_dir / "reward.jsonl",
        [
            _reward_row(
                trace_id="ta",
                suite_id="s1",
                case_id="c1",
                model_id="ollama:model_a",
                overall_score=0.9,
                success=True,
            ),
            _reward_row(
                trace_id="tb",
                suite_id="s1",
                case_id="c1",
                model_id="ollama:model_b",
                overall_score=0.8,
                success=True,
            ),
        ],
    )

    out_path = tmp_path / "out" / "prefs.jsonl"
    written, skipped = export_preferences(
        logs_root=logs_root,
        out_path=out_path,
        suite=None,
        min_score=0.0,
        success_only=False,
        include_prefixes=[],
        exclude_prefixes=[],
        exclude_model_ids=set(),
        lambda_cost=0.0,
        mu_latency=0.0,
        max_abs_score_gap=1.0,
        limit=None,
        prompt_normalize="none",
        prompt_source="auto",
        output_format="prefs",
    )

    assert written == 1
    assert skipped == 0
    row = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["preferred"] in {"a", "b"}
    assert row["meta"]["winner_model_id"] == "ollama:model_a"
    chosen = row["response_a"] if row["preferred"] == "a" else row["response_b"]
    assert chosen == response_a


def test_export_repairs_pairs_failure_to_success(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs"
    run_dir = logs_root / "run1"
    prompt = "Prompt for repairs"
    success_resp = "Successful response"
    failure_resp = "Failed response"

    _write_jsonl(run_dir / "ts.jsonl", _trace_events("task", prompt, success_resp))
    _write_jsonl(run_dir / "tf.jsonl", _trace_events("task", prompt, failure_resp))
    _write_jsonl(
        run_dir / "reward.jsonl",
        [
            _reward_row(
                trace_id="ts",
                suite_id="s1",
                case_id="c1",
                model_id="ollama:model_a",
                overall_score=0.9,
                success=True,
            ),
            _reward_row(
                trace_id="tf",
                suite_id="s1",
                case_id="c1",
                model_id="ollama:model_b",
                overall_score=0.2,
                success=False,
            ),
        ],
    )

    out_path = tmp_path / "out" / "repairs.jsonl"
    written, skipped = export_repairs(
        logs_root=logs_root,
        out_path=out_path,
        suite=None,
        min_success_score=0.7,
        max_failure_score=0.4,
        include_prefixes=[],
        exclude_prefixes=[],
        exclude_model_ids=set(),
        limit=None,
        prompt_normalize="none",
        prompt_source="auto",
        output_format="repairs",
    )

    assert written == 1
    assert skipped == 0
    row = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["failed"] == failure_resp
    assert row["repaired"] == success_resp
    assert row["meta"]["pair_type"] == "cross_model"


def test_export_curriculum_assigns_easy_bucket(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs"
    run_dir = logs_root / "run1"
    trace_id = "t1"

    _write_jsonl(run_dir / f"{trace_id}.jsonl", _trace_events("task", "prompt", "response long enough"))
    _write_jsonl(
        run_dir / "reward.jsonl",
        [
            _reward_row(
                trace_id=trace_id,
                suite_id="s1",
                case_id="c1",
                model_id="ollama:model_a",
                overall_score=0.9,
                success=True,
                violations=[],
            )
        ],
    )

    out_path = tmp_path / "out" / "curriculum.jsonl"
    written, skipped = export_curriculum(
        logs_root=logs_root,
        out_path=out_path,
        suite=None,
        min_score=0.0,
        success_only=False,
        include_prefixes=[],
        exclude_prefixes=[],
        exclude_model_ids=set(),
        step="critic_check",
        use_step_prompt=False,
        min_response_chars=1,
        easy_min_score=0.8,
        escalation_max_score=0.5,
        buckets=None,
        limit=None,
    )

    assert written == 1
    assert skipped == 0
    row = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["bucket"] == "easy"
    assert row["difficulty"] == 1
