"""Routing bandits to choose models based on reward feedback."""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Arm:
    """Beta distribution parameters for Thompson sampling."""
    # Beta distribution parameters
    alpha: float = 1.0
    beta: float = 1.0

    def sample(self) -> float:
        """Draw a probability sample from the arm's beta distribution."""
        return random.betavariate(self.alpha, self.beta)

    def update_from_reward(self, r: float) -> None:
        """Update the arm with a fractional reward in [0, 1]."""
        # fractional success update (r in [0,1])
        r = max(0.0, min(1.0, float(r)))
        self.alpha += r
        self.beta += (1.0 - r)


@dataclass
class RoutingBanditsV0:
    """Bandit state for model routing per suite."""
    version: str = "bandit.v0"
    default_model: str = "unknown"
    # by_suite[suite_id][model_id] = Arm(...)
    by_suite: Dict[str, Dict[str, Arm]] = field(default_factory=dict)

    @staticmethod
    def load(path: str | Path) -> "RoutingBanditsV0":
        """Load bandit state from disk (or return defaults if missing)."""
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
        """Persist bandit state to disk and return the path."""
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
        """Ensure a suite has Arm entries for each candidate model."""
        if suite_id not in self.by_suite:
            self.by_suite[suite_id] = {}
        for m in candidate_models:
            self.by_suite[suite_id].setdefault(m, Arm())

    def update_from_rewards(
        self,
        suite_id: str,
        model_id: str,
        rewards: List[float],
    ) -> None:
        """Update arm parameters from a list of reward values."""
        if not rewards:
            return
        self.by_suite.setdefault(suite_id, {})
        self.by_suite[suite_id].setdefault(model_id, Arm())
        arm = self.by_suite[suite_id][model_id]
        for r in rewards:
            arm.update_from_reward(r)

    def choose_model(
        self,
        suite_id: str,
        candidate_models: List[str],
        *,
        stats=None,  # RoutingStatsV0
        max_cost_usd: Optional[float] = None,
        max_latency_s: Optional[float] = None,
        utility_lambda_cost: float = 0.0,
        utility_mu_latency: float = 0.0,
    ) -> str:
        """Choose a model via exploration then Thompson sampling (quality) with optional cost/latency constraints."""
        candidate_models = [m.strip() for m in candidate_models if m and m.strip()]
        if not candidate_models:
            return self.default_model

        # Ensure suite + arms exist
        self.ensure_arms(suite_id, candidate_models)

        min_pulls = int(os.getenv("TENSORFOUNDRY_BANDIT_MIN_PULLS", "10"))

        def pulls(arm: Arm) -> float:
            return (arm.alpha + arm.beta) - 2.0

        # Explore under-sampled arms first (stats-aware exploration is optional; keep simple)
        under = [m for m in candidate_models if pulls(self.by_suite[suite_id][m]) < min_pulls]
        if under:
            return random.choice(under)

        # Thompson draw quality for each candidate
        draws = {m: self.by_suite[suite_id][m].sample() for m in candidate_models}

        # Hard constraint filter (optional)
        feasible = list(candidate_models)
        if stats is not None and (max_cost_usd is not None or max_latency_s is not None):
            feasible = []
            suite_stats = (stats.by_suite.get(suite_id) or {})
            for m in candidate_models:
                ms = suite_stats.get(m)
                est_cost = ms.cost_usd.value if ms else None
                est_lat = ms.latency_s.value if ms else None

                ok = True
                if max_cost_usd is not None and est_cost is not None and est_cost > max_cost_usd:
                    ok = False
                if max_latency_s is not None and est_lat is not None and est_lat > max_latency_s:
                    ok = False

                if ok:
                    feasible.append(m)

        # If feasible exists, pick best sampled quality
        if feasible:
            return max(feasible, key=lambda m: draws[m])

        # Otherwise, soft utility fallback
        if stats is not None and (utility_lambda_cost > 0 or utility_mu_latency > 0):
            suite_stats = (stats.by_suite.get(suite_id) or {})

            def utility(m: str) -> float:
                ms = suite_stats.get(m)
                est_cost = ms.cost_usd.value if (ms and ms.cost_usd.value is not None) else 0.0
                est_lat = ms.latency_s.value if (ms and ms.latency_s.value is not None) else 0.0
                return draws[m] - utility_lambda_cost * est_cost - utility_mu_latency * est_lat

            return max(candidate_models, key=utility)

        # Default: best quality draw
        return max(candidate_models, key=lambda m: draws[m])


def candidate_models_from_env() -> List[str]:
    """Read candidate model IDs from environment variables."""
    # Comma-separated list: "gpt-5-mini,gpt-5,claude-sonnet"
    s = os.getenv("TENSORFOUNDRY_CANDIDATE_MODELS", "").strip()
    if not s:
        # fallback to current env model id if set
        mid = os.getenv("TENSORFOUNDRY_MODEL_ID", "").strip()
        return [mid] if mid else []
    return [x.strip() for x in s.split(",") if x.strip()]


def load_rewards_for_run(run_logs_dir: str | Path) -> List[dict[str, Any]]:
    """Load reward.jsonl rows for a specific run directory."""
    p = Path(run_logs_dir) / "reward.jsonl"
    if not p.exists():
        return []
    rows: List[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows
