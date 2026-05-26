from __future__ import annotations

import json
from pathlib import Path

import pytest

from signalforgeai.pilot_check import DEFAULT_SUITE, _write_report, run_pilot_check


def test_pilot_check_dummy_mode_produces_readiness_artifacts(tmp_path: Path) -> None:
    work_dir = tmp_path / "pilot"
    payload = run_pilot_check(
        work_dir=work_dir,
        suites=[DEFAULT_SUITE],
        require_provider=set(),
        hosted_model_id="dummy_hosted",
        local_model_id="dummy_local",
    )

    assert payload["ok"] is True
    assert Path(payload["report_json"]).exists()
    assert Path(payload["report_md"]).exists()
    assert Path(payload["dataset_manifest"]).exists()
    assert {d["kind"] for d in payload["datasets"]} == {"sft", "prefs", "repairs", "curriculum"}
    assert all(d["rows"] > 0 for d in payload["datasets"])
    assert all(d["content_sha256"] for d in payload["datasets"])
    assert all(d["provenance_checked"] is True for d in payload["datasets"])
    assert all(d["duplicate_count"] == 0 for d in payload["datasets"])
    assert all(not d["quality_issues"] for d in payload["datasets"])
    assert "Readiness Summary" in Path(payload["report_md"]).read_text(encoding="utf-8")
    assert "Dataset Quality" in Path(payload["report_md"]).read_text(encoding="utf-8")
    assert "regression" not in payload
    assert "training_preflight" not in payload

    repeated = run_pilot_check(
        work_dir=work_dir,
        suites=[DEFAULT_SUITE],
        require_provider=set(),
        hosted_model_id="dummy_hosted",
        local_model_id="dummy_local",
    )
    assert repeated["ok"] is True
    assert all(d["duplicate_count"] == 0 for d in repeated["datasets"])


def test_pilot_check_default_suite_runs_from_arbitrary_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cwd = tmp_path / "outside-repo"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    payload = run_pilot_check(
        work_dir=tmp_path / "pilot",
        suites=[DEFAULT_SUITE],
        require_provider=set(),
        hosted_model_id="dummy_hosted",
        local_model_id="dummy_local",
    )

    assert payload["ok"] is True
    assert Path(payload["report_json"]).exists()


def test_pilot_check_with_equivalent_baseline_succeeds(tmp_path: Path) -> None:
    baseline = run_pilot_check(
        work_dir=tmp_path / "baseline",
        suites=[DEFAULT_SUITE],
        require_provider=set(),
        hosted_model_id="dummy_hosted",
        local_model_id="dummy_local",
    )

    current = run_pilot_check(
        work_dir=tmp_path / "current",
        suites=[DEFAULT_SUITE],
        require_provider=set(),
        hosted_model_id="dummy_hosted",
        local_model_id="dummy_local",
        baseline=Path(baseline["report_json"]),
    )

    assert current["ok"] is True
    assert current["regression"]["ok"] is True
    assert Path(current["regression"]["report_json"]).exists()
    assert Path(current["regression"]["report_md"]).exists()


def test_pilot_check_baseline_regression_fails_and_reports(tmp_path: Path) -> None:
    baseline = run_pilot_check(
        work_dir=tmp_path / "baseline",
        suites=[DEFAULT_SUITE],
        require_provider=set(),
        hosted_model_id="dummy_hosted",
        local_model_id="dummy_local",
    )
    baseline_payload = json.loads(Path(baseline["report_json"]).read_text(encoding="utf-8"))
    baseline_payload["suites"][0]["results"].append(
        {
            "case_id": "missing_in_current",
            "passed": True,
            "score": 1.0,
            "trace_id": "trace-extra",
            "trace_path": "logs/trace-extra.jsonl",
            "terminal_status": "success",
            "terminal_reason": None,
            "notes": [],
            "failure_mode": None,
            "diagnosis": {},
            "artifact_refs": {"trace": "logs/trace-extra.jsonl"},
        }
    )
    mutated_baseline = tmp_path / "baseline_with_extra_case.json"
    mutated_baseline.write_text(json.dumps(baseline_payload), encoding="utf-8")

    current = run_pilot_check(
        work_dir=tmp_path / "current",
        suites=[DEFAULT_SUITE],
        require_provider=set(),
        hosted_model_id="dummy_hosted",
        local_model_id="dummy_local",
        baseline=mutated_baseline,
    )

    assert current["ok"] is False
    assert current["regression"]["ok"] is False
    assert any("missing" in issue for issue in current["regression"]["issues"])
    assert "Regression Gate" in Path(current["report_md"]).read_text(encoding="utf-8")


def test_pilot_check_training_preflight_writes_evidence(tmp_path: Path) -> None:
    payload = run_pilot_check(
        work_dir=tmp_path / "pilot",
        suites=[DEFAULT_SUITE],
        require_provider=set(),
        hosted_model_id="dummy_hosted",
        local_model_id="dummy_local",
        training_preflight=True,
        training_base_model="dummy/base",
    )

    assert payload["ok"] is True
    training = payload["training_preflight"]
    assert training["version"] == "training_preflight.v0"
    assert training["ok"] is True
    assert training["artifact_manifest_version"] == "training_artifact.v0"
    assert {d["role"] for d in training["datasets"]} == {"sft", "dpo"}
    assert Path(training["report_json"]).exists()
    assert Path(training["distillation_recipe"]).exists()
    assert Path(payload["distillation_recipe"]).exists()
    full_report = json.loads(Path(training["report_json"]).read_text(encoding="utf-8"))
    assert full_report["artifact_manifest"]["dataset_hashes"]["dpo"]
    assert "Training Preflight" in Path(payload["report_md"]).read_text(encoding="utf-8")


def test_pilot_report_highlights_provider_and_dataset_failures(tmp_path: Path) -> None:
    report = tmp_path / "pilot_readiness.md"
    payload = {
        "ok": False,
        "work_dir": str(tmp_path),
        "suites": [],
        "dataset_manifest": str(tmp_path / "datasets" / "manifest.json"),
        "providers": [
            {
                "provider": "dummy",
                "model_id": "dummy_good",
                "ok": True,
                "required": True,
                "skipped": False,
                "reason": "",
            },
            {
                "provider": "openai",
                "model_id": "openai:gpt-5-mini",
                "ok": False,
                "required": False,
                "skipped": True,
                "reason": "OPENAI_API_KEY is not set",
            },
            {
                "provider": "ollama",
                "model_id": "ollama:missing",
                "ok": False,
                "required": True,
                "skipped": False,
                "reason": "Ollama is not reachable | connection refused",
            },
        ],
        "datasets": [
            {
                "kind": "sft",
                "rows": 0,
                "ok": False,
                "path": str(tmp_path / "datasets" / "pilot.sft.jsonl"),
                "issues": ["dataset has no rows"],
                "content_sha256": "",
                "split_counts": {},
                "duplicate_count": 0,
                "provenance_checked": True,
                "missing_artifact_refs": [],
                "quality_issues": ["line 1: missing meta.split"],
            }
        ],
        "issues": ["dataset sft: dataset has no rows"],
    }

    _write_report(payload, report)

    text = report.read_text(encoding="utf-8")
    assert "First failing gate: `Provider readiness`" in text
    assert "Fix required ollama configuration" in text
    assert "optional skip" in text
    assert "hard failure" in text
    assert "Dataset Failure Details" in text
    assert "SFT training rows" in text
    assert "Regenerate pilot outputs" in text
    assert "Ollama is not reachable \\| connection refused" in text
