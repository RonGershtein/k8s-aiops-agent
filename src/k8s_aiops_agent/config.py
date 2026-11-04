from enum import StrEnum
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LlmProvider(StrEnum):
    ANTHROPIC = "anthropic"
    SCRIPTED = "scripted"


class KubeBackend(StrEnum):
    LIVE = "live"
    FIXTURE = "fixture"


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    llm_provider: LlmProvider = LlmProvider.SCRIPTED
    model: str = "claude-sonnet-5"
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    backend: KubeBackend = KubeBackend.FIXTURE
    kubeconfig: Path | None = None
    # Deny by default: the agent reads nothing until namespaces are explicitly allowed.
    allowed_namespaces: list[str] = Field(default_factory=list)
    fixtures_dir: Path = Path("scenarios")

    max_steps: int = Field(default=12, ge=1, le=50)
    max_total_tokens: int = Field(default=60_000, ge=1_000)
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_tool_output_chars: int = Field(default=6_000, ge=500)

    trace_dir: Path = Path("traces")

    @model_validator(mode="after")
    def _anthropic_requires_api_key(self) -> "AgentSettings":
        if self.llm_provider is LlmProvider.ANTHROPIC and self.anthropic_api_key is None:
            raise ValueError("ANTHROPIC_API_KEY must be set when AGENT_LLM_PROVIDER=anthropic")
        return self
