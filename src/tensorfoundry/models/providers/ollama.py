from __future__ import annotations

import os
import json
import time
import urllib.request
from typing import Optional

from tensorfoundry.models.types import ModelOutput, ModelMetrics

def _windows_host_from_wsl() -> Optional[str]:
    # Most reliable: default gateway in WSL2 points to Windows host
    try:
        with os.popen("ip route | awk '/default/ {print $3}'") as p:
            ip = p.read().strip()
            return ip or None
    except Exception:
        return None

class OllamaProvider:
    def __init__(self, *, base_url: str | None = None) -> None:
        # Allow override via env var
        base_url = base_url or os.getenv("OLLAMA_BASE_URL")

        # If not provided, choose a sensible default
        if not base_url:
            # If we’re in WSL, localhost is often wrong — prefer Windows host IP
            if "WSL_DISTRO_NAME" in os.environ:
                host = _windows_host_from_wsl()
                if host:
                    base_url = f"http://{host}:11434"
                else:
                    base_url = "http://localhost:11434"
            else:
                base_url = "http://localhost:11434"

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
