from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeAlias


ClassInfo: TypeAlias = type[Any] | tuple[type[Any], ...]

REQUIRED: dict[str, ClassInfo] = {
    "version": str,
    "trace_id": str,
    "run_id": str,
    "suite_id": str,
    "task_id": str,
    "agent_id": str,
    "model_id": str,
    "commit_sha": str,
    "success": bool,
    "overall_score": (int, float),
    "subscores": dict,
    "violations": list,
    "created_at": str,
}


def validate_reward_obj(obj: dict[str, Any]) -> None:
    for key, typ in REQUIRED.items():
        if key not in obj:
            raise ValueError(f"Missing key: {key}")
        if not isinstance(obj[key], typ):
            raise TypeError(f"Key {key} expected {typ}, got {type(obj[key])}")

    if obj["version"] != "reward.v0":
        raise ValueError(f"Unsupported reward version: {obj['version']}")

    score = float(obj["overall_score"])
    if not (0.0 <= score <= 1.0):
        raise ValueError(f"overall_score out of range: {score}")

    for k, v in obj.get("subscores", {}).items():
        vv = float(v)
        if not (0.0 <= vv <= 1.0):
            raise ValueError(f"subscore {k} out of range: {vv}")


def validate_reward_jsonl(path: str | Path) -> None:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i}: {e}") from e
            validate_reward_obj(obj)
