from __future__ import annotations

import os
from typing import Optional

from tensorfoundry.models.base import ModelProvider
from tensorfoundry.models.providers.dummy import DummyProvider
from tensorfoundry.models.providers.ollama import OllamaProvider
from tensorfoundry.models.providers.openai_provider import OpenAIProvider

_dummy: Optional[ModelProvider] = None
_ollama: Optional[ModelProvider] = None
_openai: Optional[ModelProvider] = None


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


def get_provider_for_model(model_id: str) -> ModelProvider:
    # CI safety: never touch networked providers unless explicitly allowed
    if _is_ci() and model_id.startswith(("ollama:", "openai:")):
        return _get_dummy()

    if model_id.startswith("ollama:"):
        return _get_ollama()
    if model_id.startswith("openai:"):
        return _get_openai()
    return _get_dummy()