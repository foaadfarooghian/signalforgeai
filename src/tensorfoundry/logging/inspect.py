"""Inspect and summarise TensorFoundry JSONL traces.

Usage:
  python -m tensorfoundry.logging.inspect path/to/trace.jsonl
  python -m tensorfoundry.logging.inspect path/to/trace.jsonl --json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TERMINAL_EVENT_TYPES = {"task_completed", "task_failed"}


@dataclass(frozen=True)
class TraceSummary:
    path: str
    trace_ids: List[str]
    num_events: int
    event_type_counts: Dict[str, int]
    stage_counts: Dict[str, int]
    terminal_event_type: Optional[str]
    terminal_outcome: Dict[str, Any]
    first_timestamp: Optional[str]
    last_timestamp: Optional[str]
    timeline: List[Dict[str, str]]  # timestamp, stage, event_type


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _extract_timeline(events: Iterable[Dict[str, Any]], *, max_lines: int = 25) -> List[Dict[str, str]]:
    timeline: List[Dict[str, str]] = []
    for ev in events:
        timeline.append(
            {
                "timestamp": str(ev.get("timestamp", "")),
                "stage": str(ev.get("stage", "")),
                "event_type": str(ev.get("event_type", "")),
            }
        )
    # Keep timeline compact
    if len(timeline) <= max_lines:
        return timeline

    head = timeline[: max_lines // 2]
    tail = timeline[-(max_lines - len(head)) :]
    return head + [{"timestamp": "...", "stage": "...", "event_type": "..."}] + tail


def _find_terminal(events: List[Dict[str, Any]]) -> Tuple[Optional[str], Dict[str, Any]]:
    # Search from end, terminal event is most likely near the end
    for ev in reversed(events):
        et = ev.get("event_type")
        if et in TERMINAL_EVENT_TYPES:
            outcome = ev.get("outcome") if isinstance(ev.get("outcome"), dict) else {}
            return str(et), outcome
    return None, {}


def summarise_trace(events: List[Dict[str, Any]], *, path: Path) -> TraceSummary:
    trace_ids = sorted({str(e.get("trace_id")) for e in events if e.get("trace_id") is not None})
    event_type_counts = Counter(str(e.get("event_type")) for e in events)
    stage_counts = Counter(str(e.get("stage")) for e in events)

    terminal_event_type, terminal_outcome = _find_terminal(events)

    timestamps = [str(e.get("timestamp")) for e in events if e.get("timestamp") is not None]
    first_ts = timestamps[0] if timestamps else None
    last_ts = timestamps[-1] if timestamps else None

    timeline = _extract_timeline(events, max_lines=25)

    return TraceSummary(
        path=str(path),
        trace_ids=trace_ids,
        num_events=len(events),
        event_type_counts=dict(event_type_counts),
        stage_counts=dict(stage_counts),
        terminal_event_type=terminal_event_type,
        terminal_outcome=terminal_outcome,
        first_timestamp=first_ts,
        last_timestamp=last_ts,
        timeline=timeline,
    )


def format_summary(summary: TraceSummary) -> str:
    lines: List[str] = []
    lines.append(f"Trace file: {summary.path}")
    lines.append(f"Events: {summary.num_events}")
    lines.append(f"Trace IDs: {', '.join(summary.trace_ids) if summary.trace_ids else '(none)'}")
    lines.append(f"Time range: {summary.first_timestamp or '(unknown)'} → {summary.last_timestamp or '(unknown)'}")
    lines.append("")

    lines.append("Event types:")
    for k, v in sorted(summary.event_type_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  - {k}: {v}")
    lines.append("")

    lines.append("Stages:")
    for k, v in sorted(summary.stage_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  - {k}: {v}")
    lines.append("")

    lines.append("Terminal:")
    if summary.terminal_event_type:
        lines.append(f"  - event_type: {summary.terminal_event_type}")
        if summary.terminal_outcome:
            status = summary.terminal_outcome.get("status")
            reason = summary.terminal_outcome.get("reason")
            confidence = summary.terminal_outcome.get("confidence")
            lines.append(f"  - outcome.status: {status!r}")
            if reason is not None:
                lines.append(f"  - outcome.reason: {reason!r}")
            if confidence is not None:
                lines.append(f"  - outcome.confidence: {confidence!r}")
        else:
            lines.append("  - outcome: {}")
    else:
        lines.append("  - (no terminal event found)")
    lines.append("")

    lines.append("Timeline (compact):")
    for row in summary.timeline:
        lines.append(f"  {row['timestamp']}  [{row['stage']}]  {row['event_type']}")

    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a TensorFoundry JSONL trace.")
    parser.add_argument("path", type=str, help="Path to a .jsonl trace file")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    path = Path(args.path)
    try:
        events = read_jsonl(path)
    except FileNotFoundError:
        print(f"ERROR: file not found: {path}")
        return 1
    except json.JSONDecodeError as e:
        print(f"ERROR: invalid JSON in file: {e}")
        return 1

    summary = summarise_trace(events, path=path)

    if args.json:
        print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    else:
        print(format_summary(summary), end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())