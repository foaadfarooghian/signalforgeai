from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml  # add PyYAML dependency
from importlib.resources import files


@dataclass(frozen=True)
class ModelPrice:
    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: Optional[float] = None

def _read_default_yaml() -> str:
    return (files("tensorfoundry.models.pricing.config") / "openai_pricing.yaml").read_text(encoding="utf-8")

def load_openai_pricing() -> Dict[str, ModelPrice]:
    # Allow override; otherwise load packaged default
    override = os.getenv("TENSORFOUNDRY_OPENAI_PRICING_PATH")
    if override:
        data = yaml.safe_load(Path(override).read_text(encoding="utf-8"))
    else:
        data = yaml.safe_load(_read_default_yaml())

    if not isinstance(data, dict) or "pricing" not in data:
        raise ValueError("OpenAI pricing config must be a mapping with key: pricing")

    pricing = data["pricing"]
    if not isinstance(pricing, dict):
        raise ValueError("OpenAI pricing config: 'pricing' must be a mapping")

    out: Dict[str, ModelPrice] = {}
    for model_name, cfg in pricing.items():
        if not isinstance(cfg, dict):
            continue
        try:
            out[model_name] = ModelPrice(
                input_per_1m=float(cfg["input_per_1m"]),
                output_per_1m=float(cfg["output_per_1m"]),
                cached_input_per_1m=float(cfg["cached_input_per_1m"]) if "cached_input_per_1m" in cfg else None,
            )
        except KeyError as e:
            raise ValueError(f"Missing required pricing field for {model_name}: {e}") from e
    return out