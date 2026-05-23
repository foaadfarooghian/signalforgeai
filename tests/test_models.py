from __future__ import annotations

from signalforgeai.models.providers.dummy import DummyProvider
from signalforgeai.models.providers.hf import HFProvider, _parse_hf_model_id
from signalforgeai.models.providers.ollama import OllamaProvider
from signalforgeai.models.registry import check_provider_for_model, get_provider_for_model


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
    monkeypatch.delenv("SIGNALFORGEAI_PROVIDER", raising=False)
    provider = get_provider_for_model("hf:org/model")
    assert isinstance(provider, HFProvider)


def test_registry_returns_ollama_provider(monkeypatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("SIGNALFORGEAI_PROVIDER", raising=False)
    provider = get_provider_for_model("ollama:llama3")
    assert isinstance(provider, OllamaProvider)


def test_registry_forced_dummy_provider(monkeypatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("SIGNALFORGEAI_PROVIDER", "dummy")
    provider = get_provider_for_model("ollama:llama3")
    assert isinstance(provider, DummyProvider)


def test_provider_check_dummy_ok() -> None:
    check = check_provider_for_model("dummy_good", required=True)
    assert check.ok is True
    assert check.provider == "dummy"


def test_provider_check_openai_missing_key_skips_when_optional(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    check = check_provider_for_model("openai:gpt-5-mini", required=False)
    assert check.ok is False
    assert check.skipped is True
    assert "OPENAI_API_KEY" in check.reason


def test_provider_check_hf_missing_adapter_skips() -> None:
    check = check_provider_for_model("hf:org/model?adapter=/tmp/not-a-real-adapter", required=False)
    assert check.ok is False
    assert check.skipped is True
    assert "adapter path" in check.reason
