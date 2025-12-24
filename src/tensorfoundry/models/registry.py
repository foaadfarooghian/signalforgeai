"""Provider registry for resolving the active model backend."""
from __future__ import annotations
import os
from typing import Optional
from .base import ModelProvider
from .providers.dummy import DummyProvider

_cached: Optional[ModelProvider] = None

def get_provider() -> ModelProvider:
    """Return a cached model provider based on environment configuration."""
    global _cached
    if _cached is not None:
        return _cached

    name = (os.getenv("TENSORFOUNDRY_PROVIDER") or "dummy").strip().lower()

    if name == "dummy":
        _cached = DummyProvider()
        return _cached

    raise ValueError(f"Unknown provider: {name!r}")
