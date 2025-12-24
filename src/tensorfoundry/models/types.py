from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass(frozen=True)
class ModelMetrics:
    latency_ms: Optional[int] = None
    cost_usd: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)  # optional extensibility

@dataclass(frozen=True)
class ModelOutput:
    text: str
    metrics: ModelMetrics
