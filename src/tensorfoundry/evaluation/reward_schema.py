from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Literal, Optional
from datetime import datetime, timezone
import json


RewardVersion = Literal["reward.v0"]


@dataclass(frozen=True)
class RewardV0:
    # --- identity / join keys
    version: RewardVersion
    trace_id: str
    run_id: str
    suite_id: str
    task_id: str

    # --- what was evaluated
    agent_id: str
    model_id: str
    commit_sha: str

    # --- scores
    success: bool
    overall_score: float  # 0..1
    subscores: dict[str, float] = field(default_factory=dict)  # each 0..1

    # --- constraints / penalties
    violations: list[str] = field(default_factory=list)
    cost_usd: Optional[float] = None
    latency_ms: Optional[int] = None

    # --- optional short explanation (non-sensitive)
    rationale: Optional[str] = None

    # --- timestamp
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # hard clamp (defensive)
        d["overall_score"] = float(max(0.0, min(1.0, d["overall_score"])))
        for k, v in list(d["subscores"].items()):
            d["subscores"][k] = float(max(0.0, min(1.0, float(v))))
        return d

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)