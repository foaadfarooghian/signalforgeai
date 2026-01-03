from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

@dataclass
class EWMA:
    value: Optional[float] = None
    decay: float = 0.9  # higher = smoother

    def update(self, x: float) -> None:
        x = float(x)
        if self.value is None:
            self.value = x
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * x

@dataclass
class ModelStats:
    # Expected cost/latency/tokens for routing decisions (NOT used to train bandit)
    cost_usd: EWMA = field(default_factory=EWMA)
    latency_s: EWMA = field(default_factory=EWMA)
    total_tokens: EWMA = field(default_factory=EWMA)

@dataclass
class RoutingStatsV0:
    version: str = "stats.v0"
    # by_suite[suite_id][model_id] = ModelStats
    by_suite: Dict[str, Dict[str, ModelStats]] = field(default_factory=dict)

    @staticmethod
    def load(path: str | Path) -> "RoutingStatsV0":
        p = Path(path)
        if not p.exists():
            return RoutingStatsV0()
        obj = json.loads(p.read_text(encoding="utf-8"))
        if obj.get("version") != "stats.v0":
            raise ValueError(f"Unsupported stats version: {obj.get('version')!r}")

        inst = RoutingStatsV0()
        for suite_id, models in (obj.get("by_suite") or {}).items():
            inst.by_suite.setdefault(suite_id, {})
            for model_id, s in (models or {}).items():
                ms = ModelStats()
                ms.cost_usd.value = s.get("cost_usd")
                ms.latency_s.value = s.get("latency_s")
                ms.total_tokens.value = s.get("total_tokens")
                inst.by_suite[suite_id][model_id] = ms
        return inst

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        obj: Dict[str, Any] = {"version": self.version, "by_suite": {}}
        for suite_id, models in self.by_suite.items():
            obj["by_suite"][suite_id] = {}
            for model_id, ms in models.items():
                obj["by_suite"][suite_id][model_id] = {
                    "cost_usd": ms.cost_usd.value,
                    "latency_s": ms.latency_s.value,
                    "total_tokens": ms.total_tokens.value,
                }
        p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        return p

    def update_from_reward_row(self, suite_id: str, model_id: str, r: dict) -> None:
        self.by_suite.setdefault(suite_id, {})
        self.by_suite[suite_id].setdefault(model_id, ModelStats())
        ms = self.by_suite[suite_id][model_id]

        c = r.get("cost_usd")
        if isinstance(c, (int, float)):
            ms.cost_usd.update(float(c))

        lat_ms = r.get("latency_ms")
        if isinstance(lat_ms, int):
            ms.latency_s.update(float(lat_ms) / 1000.0)

        t = r.get("total_tokens")
        if isinstance(t, (int, float)):
            ms.total_tokens.update(float(t))
