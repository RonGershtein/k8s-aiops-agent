"""Chat models behind one interface: Claude via the Anthropic SDK, or a scripted offline fake."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import anthropic
import yaml

from k8s_aiops_agent.config import AgentSettings, LlmProvider

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ModelError(Exception):
    """The model call failed after the SDK's own retries."""


@dataclass(frozen=True)
class ModelTurn:
    content: list[dict[str, Any]]
    stop_reason: str
    input_tokens: int
    output_tokens: int
    model: str


class ChatModel(Protocol):
    name: str

    def create(
        self, system: str, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
    ) -> ModelTurn: ...


class AnthropicModel:
    def __init__(self, settings: AgentSettings) -> None:
        key = settings.anthropic_api_key
        self.name = settings.model
        self._settings = settings
        self._client = anthropic.Anthropic(
            api_key=key.get_secret_value() if key else None,
            # With one retry, a stalled call still ends inside the run's own timeout.
            timeout=settings.timeout_seconds / 2,
            max_retries=1,
            open_telemetry=False,  # the agent emits its own generation spans for every model call
        )

    def create(
        self, system: str, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
    ) -> ModelTurn:
        try:
            # Messages are plain dicts built by the graph, not the SDK's TypedDicts.
            response = self._client.beta.messages.create(  # type: ignore[call-overload]
                model=self.name,
                max_tokens=self._settings.max_output_tokens,
                system=system,
                messages=list(messages),
                tools=list(tools),
                output_config={"effort": self._settings.effort},
                cache_control={"type": "ephemeral"},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.APIStatusError as error:
            raise ModelError(f"Claude API error {error.status_code}: {error.message}") from error
        except anthropic.APIConnectionError as error:
            raise ModelError(f"Claude API unreachable: {error}") from error
        # Append-only history: blocks (thinking included) go back to the API exactly as received.
        content: list[dict[str, Any]] = response.to_dict()["content"]
        return ModelTurn(
            content=content,
            stop_reason=str(response.stop_reason),
            input_tokens=response.usage.input_tokens
            + (response.usage.cache_read_input_tokens or 0)
            + (response.usage.cache_creation_input_tokens or 0),
            output_tokens=response.usage.output_tokens,
            model=response.model,
        )


@dataclass
class ScriptedModel:
    """Test double: replays fixed turns, so the graph runs offline and deterministically."""

    turns: list[list[dict[str, Any]]]
    name: str = "scripted"
    _position: int = field(default=0, init=False)

    @classmethod
    def from_file(cls, path: Path) -> "ScriptedModel":
        return cls(turns=yaml.safe_load(path.read_text())["turns"])

    def create(
        self, system: str, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
    ) -> ModelTurn:
        if self._position >= len(self.turns):
            blocks: list[dict[str, Any]] = [{"type": "text", "text": "(script exhausted)"}]
        else:
            blocks = self.turns[self._position]
        self._position += 1
        content = [
            {**b, "id": b.get("id", f"toolu_{self._position}_{i}")}
            if b["type"] == "tool_use"
            else b
            for i, b in enumerate(blocks)
        ]
        has_tools = any(b["type"] == "tool_use" for b in content)
        return ModelTurn(
            content=content,
            stop_reason="tool_use" if has_tools else "end_turn",
            input_tokens=1_000,
            output_tokens=100,
            model=self.name,
        )


def build_model(settings: AgentSettings, script: Path | None = None) -> ChatModel:
    if settings.llm_provider is LlmProvider.ANTHROPIC:
        return AnthropicModel(settings)
    if script is None:
        raise ValueError("the scripted provider needs a script file (see evals/scripted/)")
    return ScriptedModel.from_file(script)
