"""Chat models behind one interface: Claude, Gemini, or a scripted offline fake.

The graph speaks Anthropic-shaped messages; GeminiModel translates them at the edge.
"""

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import anthropic
import httpx
import yaml

from k8s_aiops_agent.config import AgentSettings, LlmProvider

FALLBACK_BETA = "server-side-fallback-2026-07-01"
GEMINI_RATE_LIMIT_RETRIES = 3
GEMINI_RETRY_AFTER_DEFAULT_SECONDS = 10.0
GEMINI_RETRY_AFTER_MAX_SECONDS = 30.0
ERROR_BODY_CHARS = 300


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


class GeminiModel:
    """Gemini through its OpenAI-compatible endpoint, plain httpx so no extra SDK is needed."""

    def __init__(self, settings: AgentSettings, client: httpx.Client | None = None) -> None:
        key = settings.gemini_api_key
        self.name = settings.gemini_model
        self._settings = settings
        self._client = client or httpx.Client(
            headers={"Authorization": f"Bearer {key.get_secret_value() if key else ''}"},
            timeout=settings.timeout_seconds / 2,
        )

    def create(
        self, system: str, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
    ) -> ModelTurn:
        body = gemini_request(self.name, system, messages, tools)
        body["max_tokens"] = self._settings.max_output_tokens
        data = post_gemini(self._client, self._settings.gemini_base_url, body)
        message = data["choices"][0]["message"]
        content: list[dict[str, Any]] = []
        if message.get("content"):
            content.append({"type": "text", "text": message["content"]})
        for call in message.get("tool_calls") or []:
            block = {
                "type": "tool_use",
                "id": call["id"],
                "name": call["function"]["name"],
                "input": json.loads(call["function"]["arguments"] or "{}"),
            }
            if "extra_content" in call:  # Gemini 3 thought signature: must be sent back as-is
                block["extra_content"] = call["extra_content"]
            content.append(block)
        usage = data.get("usage") or {}
        return ModelTurn(
            content=content,
            stop_reason="tool_use"
            if message.get("tool_calls")
            else str(data["choices"][0].get("finish_reason")),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            model=data.get("model", self.name),
        )


def post_gemini(client: httpx.Client, base_url: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST with a short back-off on 429: the free tier allows only a few requests a minute."""
    try:
        for attempt in range(GEMINI_RATE_LIMIT_RETRIES + 1):
            response = client.post(f"{base_url.rstrip('/')}/chat/completions", json=body)
            if response.status_code != 429 or attempt == GEMINI_RATE_LIMIT_RETRIES:
                break
            retry_after = float(
                response.headers.get("retry-after", GEMINI_RETRY_AFTER_DEFAULT_SECONDS)
            )
            time.sleep(min(retry_after, GEMINI_RETRY_AFTER_MAX_SECONDS))
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        detail = error.response.text[:ERROR_BODY_CHARS]
        raise ModelError(f"Gemini API error {error.response.status_code}: {detail}") from error
    except httpx.HTTPError as error:
        raise ModelError(f"Gemini API unreachable: {error}") from error
    data: dict[str, Any] = response.json()
    return data


def gemini_request(
    model: str,
    system: str,
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in messages:
        blocks = message["content"]
        if isinstance(blocks, str):
            out.append({"role": message["role"], "content": blocks})
        elif message["role"] == "assistant":
            calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
                    **({"extra_content": b["extra_content"]} if "extra_content" in b else {}),
                }
                for b in blocks
                if b["type"] == "tool_use"
            ]
            text = "".join(b["text"] for b in blocks if b["type"] == "text")
            out.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    **({"tool_calls": calls} if calls else {}),
                }
            )
        else:
            for b in blocks:  # tool results first, as the graph builds them, then any text
                if b["type"] == "tool_result":
                    out.append(
                        {"role": "tool", "tool_call_id": b["tool_use_id"], "content": b["content"]}
                    )
                elif b["type"] == "text":
                    out.append({"role": "user", "content": b["text"]})
    return {
        "model": model,
        "messages": out,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": inline_refs(t["input_schema"]),
                },
            }
            for t in tools
        ],
    }


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace Pydantic's $defs/$ref with inline copies; not every Gemini endpoint resolves refs."""
    defs = schema.get("$defs", {})

    def resolve_refs(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve_refs(defs[node["$ref"].rsplit("/", 1)[-1]])
            return {k: resolve_refs(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve_refs(v) for v in node]
        return node

    resolved: dict[str, Any] = resolve_refs(schema)
    return resolved


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
    if settings.llm_provider is LlmProvider.GEMINI:
        return GeminiModel(settings)
    if script is None:
        raise ValueError("the scripted provider needs a script file (see evals/scripted/)")
    return ScriptedModel.from_file(script)
