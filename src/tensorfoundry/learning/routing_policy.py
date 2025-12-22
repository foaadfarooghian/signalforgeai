from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RoutingPolicyV0:
    version: str
    default_model: str
    by_suite: dict[str, str]

    @staticmethod
    def load(path: str | Path) -> "RoutingPolicyV0":
        p = Path(path)
        obj = json.loads(p.read_text(encoding="utf-8"))
        if obj.get("version") != "routing.v0":
            raise ValueError(f"Unsupported policy version: {obj.get('version')!r}")
        return RoutingPolicyV0(
            version=obj["version"],
            default_model=obj["default_model"],
            by_suite=dict(obj.get("by_suite", {})),
        )

    def pick_model(self, suite_id: str) -> str:
        return self.by_suite.get(suite_id, self.default_model)


def load_policy_if_present(path: Optional[str | Path]) -> Optional[RoutingPolicyV0]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return RoutingPolicyV0.load(p)