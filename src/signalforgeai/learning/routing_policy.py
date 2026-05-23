"""Routing policy definitions for model selection by suite."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RoutingPolicyV0:
    """Static routing policy mapping suites to preferred models."""
    version: str
    default_model: str
    by_suite: dict[str, str]

    @staticmethod
    def load(path: str | Path) -> "RoutingPolicyV0":
        """Load a routing policy file from disk."""
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
        """Return the model for a suite or the default model."""
        return self.by_suite.get(suite_id, self.default_model)


def load_policy_if_present(path: Optional[str | Path]) -> Optional[RoutingPolicyV0]:
    """Return a policy if the file exists; otherwise None."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return RoutingPolicyV0.load(p)
