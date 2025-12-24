from __future__ import annotations

import time
from typing import Optional

from openai import OpenAI

from tensorfoundry.models.types import ModelOutput, ModelMetrics


class OpenAIProvider:
    def __init__(self) -> None:
        self.client = OpenAI()

   
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

        # --- usage extraction (robust across SDK shapes) ---
        usage = getattr(resp, "usage", None)

        if usage is None:
            usage_d = {}
        else:
            # In the SDK, usage may be a typed object; normalize it
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

        # If anything is missing, fall back to model_dump()
        if not usage_d.get("total_tokens"):
            try:
                dumped = resp.model_dump()
                usage_d = dumped.get("usage", usage_d) or usage_d
            except Exception:
                pass

        # Make sure numbers are plain ints (nice for JSONL)
        def as_int(x):
            return int(x) if isinstance(x, (int, float)) else None

        usage_norm = {
            "input_tokens": as_int(usage_d.get("input_tokens")),
            "output_tokens": as_int(usage_d.get("output_tokens")),
            "total_tokens": as_int(usage_d.get("total_tokens")),
            "input_tokens_details": usage_d.get("input_tokens_details"),
            "output_tokens_details": usage_d.get("output_tokens_details"),
        }

        return ModelOutput(
            text=text,
            metrics=ModelMetrics(
                latency_ms=latency_ms,
                cost_usd=None,  # keep None for now; compute from tokens later if you want
                extra={
                    "provider": "openai",
                    "usage": usage_norm,
                },
            ),
        )
    
