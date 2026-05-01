from __future__ import annotations

import json
from pathlib import Path

from tensorfoundry.logging.events import make_event, new_span_id, new_trace_id
from tensorfoundry.logging.otel import export_otel_json


def test_export_otel_json_writes_spans(tmp_path: Path) -> None:
    trace_id = new_trace_id()
    event = make_event(
        trace_id=trace_id,
        span_id=new_span_id(),
        agent_name="agent",
        agent_version="0.1.0",
        stage="system",
        event_type="task_completed",
        outcome={"status": "success"},
    ).to_dict()
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps(event) + "\n", encoding="utf-8")

    out = export_otel_json(trace)
    data = json.loads(out.read_text(encoding="utf-8"))
    spans = data["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert spans[0]["traceId"] == trace_id
    assert spans[0]["name"] == "task_completed"
    attrs = {a["key"]: a["value"]["stringValue"] for a in spans[0]["attributes"]}
    assert attrs["tf.schema_version"] == "trace.v0"
    assert attrs["tf.outcome.status"] == "success"
