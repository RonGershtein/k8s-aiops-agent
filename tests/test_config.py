import pytest
from pydantic import ValidationError

from k8s_aiops_agent.config import AgentSettings, KubeBackend, LlmProvider


def load_settings() -> AgentSettings:
    """The real defaults: no .env file, and conftest strips AGENT_* from the environment."""
    return AgentSettings(_env_file=None)


def test_settings_default_to_offline_mode() -> None:
    settings = load_settings()

    assert settings.llm_provider is LlmProvider.SCRIPTED
    assert settings.backend is KubeBackend.FIXTURE


def test_settings_without_allowed_namespaces_denies_all() -> None:
    settings = load_settings()

    assert settings.allowed_namespaces == []


def test_settings_reads_allowed_namespaces_from_json_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_ALLOWED_NAMESPACES", '["team-a","team-b"]')

    settings = load_settings()

    assert settings.allowed_namespaces == ["team-a", "team-b"]


def test_settings_anthropic_without_key_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "anthropic")

    with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
        load_settings()


def test_settings_api_key_is_hidden_when_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-very-secret")

    settings = load_settings()

    assert "sk-ant-very-secret" not in repr(settings)
