from __future__ import annotations

import time
from typing import Optional

from openai import OpenAI

from tensorfoundry.models.pricing.load_pricing import load_openai_pricing, ModelPrice
from tensorfoundry.models.types import ModelOutput, ModelMetrics


class OpenAIProvider:
    def __init__(self) -> None:
        self.client = OpenAI()
        self._pricing = load_openai_pricing()  # loads once

    def _compute_cost_usd(self, *, model_name: str, usage_norm: dict) -> Optional[float]:
        price: Optional[ModelPrice] = self._pricing.get(model_name)
        if not price:
            return None

        in_tokens = usage_norm.get("input_tokens") or 0
        out_tokens = usage_norm.get("output_tokens") or 0

        cached_tokens = 0
        itd = usage_norm.get("input_tokens_details") or {}
        if isinstance(itd, dict):
            cached_tokens = int(itd.get("cached_tokens") or 0)

        billable_in_tokens = max(int(in_tokens) - int(cached_tokens), 0)

        cost_in = (billable_in_tokens * price.input_per_1m) / 1_000_000
        cost_out = (int(out_tokens) * price.output_per_1m) / 1_000_000

        cost_cached = 0.0
        if cached_tokens and price.cached_input_per_1m is not None:
            cost_cached = (int(cached_tokens) * price.cached_input_per_1m) / 1_000_000

        return round(cost_in + cost_cached + cost_out, 10)

    def generate(self, *, prompt: str, model_id: str, task_type: Optional[str] = None) -> ModelOutput:
        if not model_id.startswith("openai:"):
            raise ValueError(f"OpenAIProvider got non-openai model_id: {model_id}")
        model_name = model_id.split(":", 1)[1]

        t0 = time.time()
        resp = self.client.responses.create(
            model=model_name,
            input=prompt,
            max_output_tokens=256,
        )
        text = getattr(resp, "output_text", "") or ""
        latency_ms = int((time.time() - t0) * 1000)

        usage = getattr(resp, "usage", None)
        if usage is None:
            usage_d = {}
        else:
            try:
                usage_d = {
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                    "input_tokens_details": getattr(usage, "input_tokens_details", None),
                    "output_tokens_details": getattr(usage, "output_tokens_details", None),
                }
            except Exception:
                usage_d = {}

        if not usage_d.get("total_tokens"):
            try:
                dumped = resp.model_dump()
                usage_d = dumped.get("usage", usage_d) or usage_d
            except Exception:
                pass

        def as_int(x):
            return int(x) if isinstance(x, (int, float)) else None

        usage_norm = {
            "input_tokens": as_int(usage_d.get("input_tokens")),
            "output_tokens": as_int(usage_d.get("output_tokens")),
            "total_tokens": as_int(usage_d.get("total_tokens")),
            "input_tokens_details": usage_d.get("input_tokens_details"),
            "output_tokens_details": usage_d.get("output_tokens_details"),
        }

        cost_usd = self._compute_cost_usd(model_name=model_name, usage_norm=usage_norm)

        return ModelOutput(
            text=text,
            metrics=ModelMetrics(
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                extra={
                    "provider": "openai",
                    "model_name": model_name,
                    "usage": usage_norm,
                },
            ),
        )