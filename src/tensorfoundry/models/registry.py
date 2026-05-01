from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from tensorfoundry.models.base import ModelProvider
from tensorfoundry.models.providers.dummy import DummyProvider
from tensorfoundry.models.providers.ollama import OllamaProvider
from tensorfoundry.models.providers.openai_provider import OpenAIProvider
from tensorfoundry.models.providers.hf import HFProvider, _parse_hf_model_id


@dataclass(frozen=True)
class ProviderCheck:
    """Provider readiness check result."""
    model_id: str
    provider: str
    ok: bool
    required: bool = False
    skipped: bool = False
    reason: str = ""
    details: dict[str, Any] | None = None

_dummy: Optional[ModelProvider] = None
_ollama: Optional[ModelProvider] = None
_openai: Optional[ModelProvider] = None
_hf: Optional[ModelProvider] = None

def _is_ci() -> bool:
    return os.getenv("CI", "").lower() in {"1", "true", "yes"}


def _get_dummy() -> ModelProvider:
    global _dummy
    if _dummy is None:
        _dummy = DummyProvider()
    return _dummy


def _get_ollama() -> ModelProvider:
    global _ollama
    if _ollama is None:
        _ollama = OllamaProvider()
    return _ollama


def _get_openai() -> ModelProvider:
    global _openai
    if _openai is None:
        _openai = OpenAIProvider()
    return _openai

def _get_hf() -> ModelProvider:
    global _hf
    if _hf is None:
        _hf = HFProvider()
    assert _hf is not None
    return _hf

def get_provider_for_model(model_id: str) -> ModelProvider:
    forced_provider = os.getenv("TENSORFOUNDRY_PROVIDER", "").strip().lower()
    if forced_provider == "dummy":
        return _get_dummy()

    # CI safety: never touch networked providers unless explicitly allowed
    if _is_ci() and model_id.startswith(("ollama:", "openai:")):
        return _get_dummy()

    if model_id.startswith("ollama:"):
        return _get_ollama()
    if model_id.startswith("openai:"):
        return _get_openai()
    if model_id.startswith("hf:"):
        return _get_hf()
    return _get_dummy()


def check_provider_for_model(
    model_id: str,
    *,
    required: bool = False,
    timeout_s: float = 2.0,
) -> ProviderCheck:
    """Check whether a model provider is configured enough for a smoke run."""
    model_id = (model_id or "").strip()
    if not model_id or model_id.startswith("dummy"):
        return ProviderCheck(model_id=model_id or "dummy_good", provider="dummy", ok=True, required=required)

    if model_id.startswith("openai:"):
        if os.getenv("OPENAI_API_KEY"):
            return ProviderCheck(model_id=model_id, provider="openai", ok=True, required=required)
        return ProviderCheck(
            model_id=model_id,
            provider="openai",
            ok=False,
            required=required,
            skipped=not required,
            reason="OPENAI_API_KEY is not set",
        )

    if model_id.startswith("ollama:"):
        provider = OllamaProvider()
        try:
            req = urllib.request.Request(f"{provider.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return ProviderCheck(
                model_id=model_id,
                provider="ollama",
                ok=True,
                required=required,
                details={"base_url": provider.base_url, "models": len(payload.get("models", []))},
            )
        except Exception as exc:
            return ProviderCheck(
                model_id=model_id,
                provider="ollama",
                ok=False,
                required=required,
                skipped=not required,
                reason=f"Ollama is not reachable: {exc}",
                details={"base_url": provider.base_url},
            )

    if model_id.startswith("hf:"):
        base, adapter = _parse_hf_model_id(model_id)
        if adapter and not Path(adapter).exists():
            return ProviderCheck(
                model_id=model_id,
                provider="hf",
                ok=False,
                required=required,
                skipped=not required,
                reason=f"HF adapter path does not exist: {adapter}",
            )
        if Path(base).exists() or "/" in base:
            return ProviderCheck(
                model_id=model_id,
                provider="hf",
                ok=True,
                required=required,
                details={"base_model": base, "adapter": adapter},
            )
        return ProviderCheck(
            model_id=model_id,
            provider="hf",
            ok=False,
            required=required,
            skipped=not required,
            reason=f"HF model id is not a local path or repo id: {base}",
        )

    return ProviderCheck(model_id=model_id, provider="dummy", ok=True, required=required)
