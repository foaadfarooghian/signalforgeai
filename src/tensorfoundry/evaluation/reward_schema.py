from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Literal, Optional
from datetime import datetime, timezone
import json


RewardVersion = Literal["reward.v0"]


@dataclass(frozen=True)
class RewardV0:
    # --- identity / join keys (required)
    version: RewardVersion
    trace_id: str
    run_id: str
    suite_id: str
    case_id: str

    # --- what was evaluated (required)
    agent_id: str
    model_id: str
    commit_sha: str

    # --- scores (required)
    success: bool
    overall_score: float  # 0..1

    # --- optional fields (defaults)
    subscores: dict[str, float] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    cost_usd: Optional[float] = None
    latency_ms: Optional[int] = None
    terminal_status: Optional[str] = None
    terminal_reason: Optional[str] = None
    rationale: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if self.case_id is None:
            object.__setattr__(self, "case_id", self.task_id)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # hard clamp (defensive)
        d["overall_score"] = float(max(0.0, min(1.0, d["overall_score"])))
        for k, v in list(d["subscores"].items()):
            d["subscores"][k] = float(max(0.0, min(1.0, float(v))))
        return d

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
