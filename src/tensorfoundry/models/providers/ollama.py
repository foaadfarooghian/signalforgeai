from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import asdict
from typing import Optional

from tensorfoundry.models.types import ModelOutput, ModelMetrics


class OllamaProvider:
    def __init__(self, *, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def generate(self, *, prompt: str, model_id: str, task_type: Optional[str] = None) -> ModelOutput:
        # model_id format: "ollama:<model_name>"
        if not model_id.startswith("ollama:"):
            raise ValueError(f"OllamaProvider got non-ollama model_id: {model_id}")
        model_name = model_id.split(":", 1)[1]

        t0 = time.time()

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,  # return a single JSON object
        }

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Ollama returns fields like "response" (text) when stream=False. :contentReference[oaicite:2]{index=2}
        text = data.get("response", "") or ""

        latency_ms = int((time.time() - t0) * 1000)
        return ModelOutput(
            text=text,
            metrics=ModelMetrics(latency_ms=latency_ms, cost_usd=0.0, extra={"provider": "ollama"}),
        )
