"""Recipe loading for TensorFoundry distillation evaluation gates."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DISTILLATION_RECIPE_VERSION = "distillation_recipe.v0"


@dataclass(frozen=True)
class DistillationThresholds:
    """Gate thresholds for distillation candidate comparison."""

    max_pass_rate_drop: float = 0.0
    max_mean_score_drop: float = 0.0
    allow_worse_failure_modes: bool = False
    min_cost_per_success_improvement: Optional[float] = None
    min_latency_ms_p50_improvement: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_pass_rate_drop": self.max_pass_rate_drop,
            "max_mean_score_drop": self.max_mean_score_drop,
            "allow_worse_failure_modes": self.allow_worse_failure_modes,
            "min_cost_per_success_improvement": self.min_cost_per_success_improvement,
            "min_latency_ms_p50_improvement": self.min_latency_ms_p50_improvement,
        }


@dataclass(frozen=True)
class DistillationRecipe:
    """A lightweight v0 recipe for specialist candidate evaluation."""

    version: str
    id: str
    domain: str
    suite: str
    baseline_model_id: str
    candidate_model_id: str
    training_preflight_path: str
    thresholds: DistillationThresholds
    source_path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "id": self.id,
            "domain": self.domain,
            "suite": self.suite,
            "baseline_model_id": self.baseline_model_id,
            "candidate_model_id": self.candidate_model_id,
            "training_preflight_path": self.training_preflight_path,
            "thresholds": self.thresholds.to_dict(),
            "source_path": self.source_path,
        }


def load_distillation_recipe(path: str | Path) -> DistillationRecipe:
    """Load and validate a `distillation_recipe.v0` JSON or YAML file."""
    source = Path(path)
    if not source.exists():
        raise ValueError(f"distillation recipe not found: {source}")

    payload = _load_mapping(source)
    required = (
        "version",
        "id",
        "domain",
        "suite",
        "baseline_model_id",
        "candidate_model_id",
        "training_preflight_path",
        "thresholds",
    )
    for field in required:
        if field not in payload:
            raise ValueError(f"missing required recipe field: {field}")

    version = _as_str(payload["version"], "version")
    if version != DISTILLATION_RECIPE_VERSION:
        raise ValueError(
            f"unsupported distillation recipe version: {version!r}; "
            f"expected {DISTILLATION_RECIPE_VERSION!r}"
        )

    thresholds_raw = payload["thresholds"]
    if not isinstance(thresholds_raw, dict):
        raise ValueError("thresholds must be an object")

    return DistillationRecipe(
        version=version,
        id=_as_str(payload["id"], "id"),
        domain=_as_str(payload["domain"], "domain"),
        suite=_as_str(payload["suite"], "suite"),
        baseline_model_id=_as_str(payload["baseline_model_id"], "baseline_model_id"),
        candidate_model_id=_as_str(payload["candidate_model_id"], "candidate_model_id"),
        training_preflight_path=_as_str(
            payload["training_preflight_path"],
            "training_preflight_path",
        ),
        thresholds=_load_thresholds(thresholds_raw),
        source_path=str(source),
    )


def apply_recipe_overrides(
    recipe: DistillationRecipe,
    *,
    suite: Optional[str] = None,
    baseline_model_id: Optional[str] = None,
    candidate_model_id: Optional[str] = None,
    max_pass_rate_drop: Optional[float] = None,
    max_mean_score_drop: Optional[float] = None,
    allow_worse_failure_modes: Optional[bool] = None,
) -> DistillationRecipe:
    """Return a recipe with CLI overrides applied."""
    thresholds = recipe.thresholds
    if max_pass_rate_drop is not None:
        thresholds = replace(thresholds, max_pass_rate_drop=float(max_pass_rate_drop))
    if max_mean_score_drop is not None:
        thresholds = replace(thresholds, max_mean_score_drop=float(max_mean_score_drop))
    if allow_worse_failure_modes is not None:
        thresholds = replace(
            thresholds,
            allow_worse_failure_modes=bool(allow_worse_failure_modes),
        )
    return replace(
        recipe,
        suite=suite or recipe.suite,
        baseline_model_id=baseline_model_id or recipe.baseline_model_id,
        candidate_model_id=candidate_model_id or recipe.candidate_model_id,
        thresholds=thresholds,
    )


def resolve_recipe_relative_path(recipe: DistillationRecipe, value: str | Path) -> Path:
    """Resolve a path relative to the recipe file if it is not absolute."""
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(recipe.source_path).parent / path


def _load_mapping(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("distillation recipe must be an object")
    return data


def _load_thresholds(payload: Dict[str, Any]) -> DistillationThresholds:
    return DistillationThresholds(
        max_pass_rate_drop=_as_float(payload.get("max_pass_rate_drop", 0.0)),
        max_mean_score_drop=_as_float(payload.get("max_mean_score_drop", 0.0)),
        allow_worse_failure_modes=bool(payload.get("allow_worse_failure_modes", False)),
        min_cost_per_success_improvement=_optional_float(
            payload.get("min_cost_per_success_improvement")
        ),
        min_latency_ms_p50_improvement=_optional_float(
            payload.get("min_latency_ms_p50_improvement")
        ),
    )


def _as_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _as_float(value: Any) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError("threshold values must be numeric")
    return float(value)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return _as_float(value)
