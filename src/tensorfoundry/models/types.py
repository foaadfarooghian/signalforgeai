from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass(frozen=True)
class ModelMetrics:
    latency_ms: Optional[int] = None
    cost_usd: Optional[float] = None
    extra: Dict[str, Any] = None  # optional extensibility

@dataclass(frozen=True)
class ModelOutput:
    text: str
    metrics: ModelMetrics