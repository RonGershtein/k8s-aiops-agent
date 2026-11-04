from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LlmProvider(StrEnum):
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
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
    model: str = "claude-opus-5-5"
    judge_model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_output_tokens: int = Field(default=16_000, ge=1_000)
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_api_key: SecretStr | None = Field(default=None, validation_alias="GEMINI_API_KEY")

    backend: KubeBackend = KubeBackend.FIXTURE
    kubeconfig: Path | None = None
    remediator_kubeconfig: Path | None = None
    # Deny by default: the agent reads nothing until namespaces are explicitly allowed.
    allowed_namespaces: list[str] = Field(default_factory=list)
    fixtures_dir: Path = Path("scenarios")

    max_steps: int = Field(default=12, ge=1, le=50)
    max_total_tokens: int = Field(default=60_000, ge=1_000)
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_tool_output_chars: int = Field(default=6_000, ge=500)

    trace_dir: Path = Path("traces")
    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: SecretStr | None = Field(
        default=None, validation_alias="LANGFUSE_SECRET_KEY"
    )
    langfuse_host: str = Field(default="http://localhost:3000", validation_alias="LANGFUSE_HOST")

    @model_validator(mode="after")
    def _anthropic_requires_api_key(self) -> "AgentSettings":
        if self.llm_provider is LlmProvider.ANTHROPIC and self.anthropic_api_key is None:
            raise ValueError("ANTHROPIC_API_KEY must be set when AGENT_LLM_PROVIDER=anthropic")
        if self.llm_provider is LlmProvider.GEMINI and self.gemini_api_key is None:
            raise ValueError("GEMINI_API_KEY must be set when AGENT_LLM_PROVIDER=gemini")
        return self
