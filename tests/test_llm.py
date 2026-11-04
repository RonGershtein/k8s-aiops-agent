import json
from typing import Any
from unittest.mock import MagicMock

import anthropic
import httpx
import pytest

from k8s_aiops_agent import llm
from k8s_aiops_agent.llm import (
    FALLBACK_BETA,
    GEMINI_RATE_LIMIT_RETRIES,
    AnthropicModel,
    GeminiModel,
    ModelError,
)
from tests.builders import gemini_settings, make_settings, mock_http, tool_call


@pytest.fixture
def claude(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """A stand-in for the Anthropic client that AnthropicModel builds."""
    client = MagicMock()
    monkeypatch.setattr(llm.anthropic, "Anthropic", lambda **kwargs: client)
    return client


@pytest.fixture
def model(claude: MagicMock) -> AnthropicModel:
    return AnthropicModel(make_settings(llm_provider="anthropic", anthropic_api_key="sk-test"))


def api_response(content: list[dict[str, Any]], **usage: int) -> MagicMock:
    response = MagicMock(stop_reason="tool_use", model="claude-opus-5-5")
    response.to_dict.return_value = {"content": content}
    response.usage = MagicMock(
        input_tokens=usage.get("input", 0),
        output_tokens=usage.get("output", 0),
        cache_read_input_tokens=usage.get("cache_read"),
        cache_creation_input_tokens=usage.get("cache_write"),
    )
    return response


def test_anthropic_turn_keeps_blocks_and_counts_cache(
    claude: MagicMock, model: AnthropicModel
) -> None:
    blocks = [{"type": "thinking", "thinking": "", "signature": "s"}, tool_call("list_pods")]
    claude.beta.messages.create.return_value = api_response(
        blocks, input=10, output=5, cache_read=100, cache_write=20
    )

    turn = model.create("system", [{"role": "user", "content": "q"}], [])

    assert turn.content == blocks
    assert turn.input_tokens == 130
    assert turn.output_tokens == 5
    assert turn.stop_reason == "tool_use"


def test_anthropic_request_sets_effort_cache_and_fallback(
    claude: MagicMock, model: AnthropicModel
) -> None:
    claude.beta.messages.create.return_value = api_response([])

    model.create("system", [], [])
    request = claude.beta.messages.create.call_args.kwargs

    assert request["model"] == "claude-opus-5-5"
    assert request["output_config"] == {"effort": "medium"}
    assert request["cache_control"] == {"type": "ephemeral"}
    assert request["betas"] == [FALLBACK_BETA]
    assert "tool_choice" not in request


def test_anthropic_model_unreachable_api_raises_model_error(
    claude: MagicMock, model: AnthropicModel
) -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    claude.beta.messages.create.side_effect = anthropic.APIConnectionError(request=request)

    with pytest.raises(ModelError, match="unreachable"):
        model.create("system", [], [])


GEMINI_TOOL_CALL = {
    "model": "gemini-3.5-flash-lite",
    "choices": [
        {
            "finish_reason": "tool_calls",
            "message": {
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "list_pods", "arguments": '{"namespace": "ns"}'},
                        "extra_content": {"google": {"thought_signature": "sig"}},
                    }
                ],
            },
        }
    ],
    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
}


def test_gemini_model_maps_tool_calls_to_anthropic_blocks() -> None:
    client, _ = mock_http(httpx.Response(200, json=GEMINI_TOOL_CALL))
    model = GeminiModel(gemini_settings(), client)

    turn = model.create("system", [{"role": "user", "content": "q"}], [])

    assert turn.content == [
        {
            "type": "tool_use",
            "id": "c1",
            "name": "list_pods",
            "input": {"namespace": "ns"},
            "extra_content": {"google": {"thought_signature": "sig"}},
        }
    ]
    assert (turn.stop_reason, turn.input_tokens, turn.output_tokens) == ("tool_use", 7, 3)


def test_gemini_request_uses_openai_format() -> None:
    client, sent = mock_http(httpx.Response(200, json=GEMINI_TOOL_CALL))
    model = GeminiModel(gemini_settings(), client)
    call = {**tool_call("list_pods", namespace="ns"), "id": "c1"}
    result = {"type": "tool_result", "tool_use_id": "c1", "content": "out"}
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [{"type": "text", "text": "looking"}, call]},
        {"role": "user", "content": [result]},
    ]
    schema = {"$defs": {"A": {"type": "string"}}, "properties": {"a": {"$ref": "#/$defs/A"}}}
    tool = {"name": "t", "description": "d", "input_schema": schema}

    model.create("system", history, [tool])

    assert json.loads(sent[0].content)["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "looking",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "list_pods", "arguments": '{"namespace": "ns"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "out"},
    ]
    assert json.loads(sent[0].content)["tools"][0]["function"]["parameters"] == {
        "properties": {"a": {"type": "string"}}
    }


def test_gemini_rate_limit_retries_then_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)
    attempts = GEMINI_RATE_LIMIT_RETRIES + 1
    client, sent = mock_http(*[httpx.Response(429, text="quota")] * attempts)
    model = GeminiModel(gemini_settings(), client)

    with pytest.raises(ModelError, match="429"):
        model.create("system", [], [])

    assert len(sent) == attempts


def test_gemini_malformed_reply_raises_model_error() -> None:
    client, _ = mock_http(httpx.Response(200, json={"unexpected": True}))
    model = GeminiModel(gemini_settings(), client)

    with pytest.raises(ModelError, match="unexpected response"):
        model.create("system", [], [])


def test_gemini_invalid_tool_arguments_raise_model_error() -> None:
    reply = json.loads(json.dumps(GEMINI_TOOL_CALL))
    reply["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{not json"
    client, _ = mock_http(httpx.Response(200, json=reply))
    model = GeminiModel(gemini_settings(), client)

    with pytest.raises(ModelError, match="invalid tool arguments"):
        model.create("system", [], [])
