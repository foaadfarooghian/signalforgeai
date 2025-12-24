"""Utilities for writing reward logs to JSONL."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
from .reward_schema import RewardV0


def write_rewards_jsonl(path: str | Path, rewards: Iterable[RewardV0]) -> Path:
    """Append reward records to a JSONL file and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        for r in rewards:
            f.write(r.to_jsonl() + "\n")

    return path
