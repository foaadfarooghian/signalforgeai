from __future__ import annotations

from tensorfoundry.models.providers.dummy import DummyProvider
from tensorfoundry.models.providers.hf import HFProvider, _parse_hf_model_id
from tensorfoundry.models.providers.ollama import OllamaProvider
from tensorfoundry.models.registry import get_provider_for_model


def test_parse_hf_model_id_with_adapter() -> None:
    base, adapter = _parse_hf_model_id("hf:org/model?adapter=/tmp/adapter")
    assert base == "org/model"
    assert adapter == "/tmp/adapter"


def test_registry_returns_dummy_in_ci(monkeypatch) -> None:
    monkeypatch.setenv("CI", "1")
    provider = get_provider_for_model("openai:gpt-5-mini")
    assert isinstance(provider, DummyProvider)


def test_registry_returns_hf_provider(monkeypatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    provider = get_provider_for_model("hf:org/model")
    assert isinstance(provider, HFProvider)


def test_registry_returns_ollama_provider(monkeypatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    provider = get_provider_for_model("ollama:llama3")
    assert isinstance(provider, OllamaProvider)
