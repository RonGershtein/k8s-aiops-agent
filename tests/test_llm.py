from typing import Any
from unittest.mock import MagicMock

import anthropic
import httpx
import pytest

from k8s_aiops_agent import llm
from k8s_aiops_agent.llm import FALLBACK_BETA, AnthropicModel, ModelError
from tests.builders import make_settings, tool_call


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


def test_anthropic_model_counts_cached_input_and_returns_blocks_unchanged(
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


def test_anthropic_model_requests_effort_caching_and_fallback_without_forcing_tools(
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
