from __future__ import annotations

import os

from tensorfoundry.models.base import ModelProvider
from tensorfoundry.models.providers.dummy import DummyProvider
from tensorfoundry.models.providers.ollama import OllamaProvider
from tensorfoundry.models.providers.openai_provider import OpenAIProvider

_dummy = DummyProvider()
_ollama = OllamaProvider()
_openai = OpenAIProvider()

def _is_ci() -> bool:
    # GitHub Actions sets CI=true
    return os.getenv("CI", "").lower() in {"1", "true", "yes"}

def _allow_network_providers() -> bool:
    # Set in workflow/job when you *want* integration tests
    return os.getenv("TF_ALLOW_NETWORK", "").lower() in {"1", "true", "yes"}


def get_provider_for_model(model_id: str) -> ModelProvider:
    # In CI, don’t allow providers that require external services by default.
    if _is_ci() and not _allow_network_providers() and model_id.startswith(("ollama:", "openai:")):
        return _dummy

    if model_id.startswith("ollama:"):
        return _ollama
    if model_id.startswith("openai:"):
        return _openai
    return _dummy