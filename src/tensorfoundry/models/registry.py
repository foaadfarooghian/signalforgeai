from __future__ import annotations

from typing import Optional
from tensorfoundry.models.base import ModelProvider
from tensorfoundry.models.providers.dummy import DummyProvider
from tensorfoundry.models.providers.ollama import OllamaProvider
from tensorfoundry.models.providers.openai_provider import OpenAIProvider

_dummy = DummyProvider()
_ollama = OllamaProvider()
_openai = OpenAIProvider()

def get_provider_for_model(model_id: str) -> ModelProvider:
    if model_id.startswith("ollama:"):
        return _ollama
    if model_id.startswith("openai:"):
        return _openai
    return _dummy
