from __future__ import annotations
import json
from typing import Any, Dict

def dumps_row(row: Dict[str, Any], *, deterministic: bool) -> str:
    if deterministic:
        return json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return json.dumps(row, ensure_ascii=False)