from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class Arm:
    # Beta distribution parameters
    alpha: float = 1.0
    beta: float = 1.0

    def sample(self) -> float:
        return random.betavariate(self.alpha, self.beta)

    def update_from_reward(self, r: float) -> None:
        # fractional success update (r in [0,1])
        r = max(0.0, min(1.0, float(r)))
        self.alpha += r
        self.beta += (1.0 - r)


@dataclass
class RoutingBanditsV0:
    version: str = "bandit.v0"
    default_model: str = "unknown"
    # by_suite[suite_id][model_id] = Arm(...)
    by_suite: Dict[str, Dict[str, Arm]] = field(default_factory=dict)

    @staticmethod
    def load(path: str | Path) -> "RoutingBanditsV0":
        p = Path(path)
        if not p.exists():
            return RoutingBanditsV0()
        obj = json.loads(p.read_text(encoding="utf-8"))
        if obj.get("version") != "bandit.v0":
            raise ValueError(f"Unsupported bandit version: {obj.get('version')!r}")

        inst = RoutingBanditsV0(
            version="bandit.v0",
            default_model=obj.get("default_model", "unknown"),
            by_suite={},
        )
        for suite_id, arms in (obj.get("by_suite") or {}).items():
            inst.by_suite[suite_id] = {}
            for model_id, params in (arms or {}).items():
                inst.by_suite[suite_id][model_id] = Arm(
                    alpha=float(params.get("alpha", 1.0)),
                    beta=float(params.get("beta", 1.0)),
                )
        return inst

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        obj: Dict[str, Any] = {
            "version": self.version,
            "default_model": self.default_model,
            "by_suite": {
                suite_id: {
                    model_id: {"alpha": arm.alpha, "beta": arm.beta}
                    for model_id, arm in models.items()
                }
                for suite_id, models in self.by_suite.items()
            },
        }
        p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        return p

    def ensure_arms(self, suite_id: str, candidate_models: List[str]) -> None:
        if suite_id not in self.by_suite:
            self.by_suite[suite_id] = {}
        for m in candidate_models:
            self.by_suite[suite_id].setdefault(m, Arm())
    

    def choose_model(self, suite_id: str, candidate_models: List[str]) -> str:
        candidate_models = [m.strip() for m in candidate_models if m and m.strip()]
        if not candidate_models:
            return self.default_model

        # Ensure suite + arms exist before reading them
        self.ensure_arms(suite_id, candidate_models)

        min_pulls = int(os.getenv("TENSORFOUNDRY_BANDIT_MIN_PULLS", "5"))

        def pulls(arm: Arm) -> float:
            return (arm.alpha + arm.beta) - 2.0

        # Explore under-sampled arms first
        under = sorted(candidate_models, key=lambda m: pulls(self.by_suite[suite_id][m]))
        if pulls(self.by_suite[suite_id][under[0]]) < min_pulls:
            return under[0]

        # Thompson Sampling once all arms have enough pulls
        best_m = None
        best_draw = -1.0
        for m in candidate_models:
            draw = self.by_suite[suite_id][m].sample()
            if draw > best_draw:
                best_draw = draw
                best_m = m

        return best_m or self.default_model

    def update_from_rewards(
        self,
        suite_id: str,
        model_id: str,
        rewards: List[float],
    ) -> None:
        if not rewards:
            return
        self.by_suite.setdefault(suite_id, {})
        self.by_suite[suite_id].setdefault(model_id, Arm())
        arm = self.by_suite[suite_id][model_id]
        for r in rewards:
            arm.update_from_reward(r)


def candidate_models_from_env() -> List[str]:
    # Comma-separated list: "gpt-5-mini,gpt-5,claude-sonnet"
    s = os.getenv("TENSORFOUNDRY_CANDIDATE_MODELS", "").strip()
    if not s:
        # fallback to current env model id if set
        mid = os.getenv("TENSORFOUNDRY_MODEL_ID", "").strip()
        return [mid] if mid else []
    return [x.strip() for x in s.split(",") if x.strip()]


def load_rewards_for_run(run_logs_dir: str | Path) -> List[dict[str, Any]]:
    p = Path(run_logs_dir) / "reward.jsonl"
    if not p.exists():
        return []
    rows: List[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows
