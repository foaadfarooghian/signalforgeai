"""OpenTelemetry-compatible JSON export for TensorFoundry traces."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def _nanos(timestamp: str) -> str:
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return str(int(dt.timestamp() * 1_000_000_000))
    except Exception:
        return "0"


def _span_id(value: Any) -> str:
    text = str(value or "")
    return (text[:16]).ljust(16, "0")


def _attrs(ev: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_payload = ev.get("payload")
    raw_outcome = ev.get("outcome")
    raw_agent = ev.get("agent")
    payload: Dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
    outcome: Dict[str, Any] = raw_outcome if isinstance(raw_outcome, dict) else {}
    agent: Dict[str, Any] = raw_agent if isinstance(raw_agent, dict) else {}
    pairs = {
        "tf.schema_version": ev.get("schema_version", ""),
        "tf.stage": ev.get("stage", ""),
        "tf.event_type": ev.get("event_type", ""),
        "tf.agent.name": agent.get("name", ""),
        "tf.agent.version": agent.get("version", ""),
        "tf.outcome.status": outcome.get("status", ""),
        "tf.tool.name": payload.get("tool_name", ""),
    }
    attrs: List[Dict[str, Any]] = []
    for key, value in pairs.items():
        if value is None or value == "":
            continue
        attrs.append({"key": key, "value": {"stringValue": str(value)}})
    return attrs


def trace_events_to_otel(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Map TensorFoundry trace events to an OTel JSON-shaped span list."""
    spans: List[Dict[str, Any]] = []
    for ev in events:
        trace_id = str(ev.get("trace_id") or "").ljust(32, "0")[:32]
        span = {
            "traceId": trace_id,
            "spanId": _span_id(ev.get("span_id")),
            "parentSpanId": _span_id(ev.get("parent_span_id")) if ev.get("parent_span_id") else "",
            "name": str(ev.get("event_type") or "event"),
            "kind": "SPAN_KIND_INTERNAL",
            "startTimeUnixNano": _nanos(str(ev.get("timestamp") or "")),
            "endTimeUnixNano": _nanos(str(ev.get("timestamp") or "")),
            "attributes": _attrs(ev),
        }
        spans.append(span)
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "service.name",
                            "value": {"stringValue": "tensorfoundry"},
                        }
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "tensorfoundry.trace_export", "version": "trace.v0"},
                        "spans": spans,
                    }
                ],
            }
        ]
    }


def export_otel_json(trace_path: str | Path, out_path: Optional[str | Path] = None) -> Path:
    """Export a trace JSONL file as OTel-compatible JSON."""
    trace_path = Path(trace_path)
    out = Path(out_path) if out_path else trace_path.with_suffix(".otel.json")
    data = trace_events_to_otel(list(_iter_jsonl(trace_path)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export a TensorFoundry trace as OTel JSON.")
    parser.add_argument("trace", type=str, help="Trace .jsonl file")
    parser.add_argument("--out", type=str, default="", help="Output .json path")
    args = parser.parse_args(argv)
    out = export_otel_json(args.trace, args.out or None)
    print(f"OTel JSON -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
