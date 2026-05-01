from __future__ import annotations

from pathlib import Path

from tensorfoundry.pilot_check import DEFAULT_SUITE, run_pilot_check


def test_pilot_check_dummy_mode_produces_readiness_artifacts(tmp_path: Path) -> None:
    payload = run_pilot_check(
        work_dir=tmp_path / "pilot",
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
