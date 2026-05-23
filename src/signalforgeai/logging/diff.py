"""Diff two SignalForge AI JSONL traces.

Usage:
  python -m signalforgeai.logging.diff logs/a.jsonl logs/b.jsonl
  python -m signalforgeai.logging.diff logs/a.jsonl logs/b.jsonl --json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from signalforgeai.logging.inspect import read_jsonl, summarise_trace


@dataclass(frozen=True)
class TraceDiff:
    """Summary of differences between two trace files."""
    left: str
    right: str
    left_events: int
    right_events: int
    event_type_delta: Dict[str, int]
    stage_delta: Dict[str, int]
    terminal_left: Dict[str, Any]
    terminal_right: Dict[str, Any]
    terminal_changed: bool


def _counter_delta(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    """Return per-key deltas for two counters (b - a), excluding zeros."""
    keys = set(a) | set(b)
    return {k: int(b.get(k, 0)) - int(a.get(k, 0)) for k in sorted(keys) if (b.get(k, 0) - a.get(k, 0)) != 0}


def diff_traces(left_path: Path, right_path: Path) -> TraceDiff:
    """Compute a TraceDiff summary between two JSONL trace files."""
    left_events = read_jsonl(left_path)
    right_events = read_jsonl(right_path)

    left_sum = summarise_trace(left_events, path=left_path)
    right_sum = summarise_trace(right_events, path=right_path)

    event_type_delta = _counter_delta(left_sum.event_type_counts, right_sum.event_type_counts)
    stage_delta = _counter_delta(left_sum.stage_counts, right_sum.stage_counts)

    terminal_left = {
        "event_type": left_sum.terminal_event_type,
        **(left_sum.terminal_outcome or {}),
    }
    terminal_right = {
        "event_type": right_sum.terminal_event_type,
        **(right_sum.terminal_outcome or {}),
    }

    terminal_changed = terminal_left != terminal_right

    return TraceDiff(
        left=str(left_path),
        right=str(right_path),
        left_events=left_sum.num_events,
        right_events=right_sum.num_events,
        event_type_delta=event_type_delta,
        stage_delta=stage_delta,
        terminal_left=terminal_left,
        terminal_right=terminal_right,
        terminal_changed=terminal_changed,
    )


def format_diff(d: TraceDiff) -> str:
    """Render a TraceDiff as a human-readable report."""
    lines: List[str] = []
    lines.append(f"Left:  {d.left}")
    lines.append(f"Right: {d.right}")
    lines.append("")
    lines.append(f"Events: {d.left_events} → {d.right_events} (delta {d.right_events - d.left_events:+d})")
    lines.append("")

    lines.append("Event type delta (right - left):")
    if d.event_type_delta:
        for k, v in d.event_type_delta.items():
            lines.append(f"  - {k}: {v:+d}")
    else:
        lines.append("  (no change)")
    lines.append("")

    lines.append("Stage delta (right - left):")
    if d.stage_delta:
        for k, v in d.stage_delta.items():
            lines.append(f"  - {k}: {v:+d}")
    else:
        lines.append("  (no change)")
    lines.append("")

    lines.append("Terminal outcome:")
    lines.append(f"  - left:  {d.terminal_left}")
    lines.append(f"  - right: {d.terminal_right}")
    lines.append(f"  - changed: {str(d.terminal_changed).lower()}")

    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for diffing two trace files."""
    p = argparse.ArgumentParser(description="Diff two SignalForge AI JSONL traces.")
    p.add_argument("left", type=str)
    p.add_argument("right", type=str)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    d = diff_traces(Path(args.left), Path(args.right))
    if args.json:
        print(json.dumps(asdict(d), ensure_ascii=False, indent=2))
    else:
        print(format_diff(d), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
