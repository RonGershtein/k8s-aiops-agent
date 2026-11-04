"""Small builders shared by the tests. Fixtures live in conftest.py."""

import json
from pathlib import Path
from typing import Any

import httpx
from kubernetes.client.exceptions import ApiException

from k8s_aiops_agent.config import AgentSettings
from k8s_aiops_agent.evals import load_scenarios
from k8s_aiops_agent.llm import ScriptedModel

SCENARIOS_DIR = Path("scenarios")
SCRIPTS_DIR = Path("evals/scripted")
SCENARIO_NAMESPACES = [s.namespace for s in load_scenarios(SCENARIOS_DIR, True, None)]


def make_settings(**overrides: Any) -> AgentSettings:
    """Settings that ignore .env and allow the scenario namespaces unless overridden."""
    values: dict[str, Any] = {"allowed_namespaces": SCENARIO_NAMESPACES, **overrides}
    return AgentSettings(_env_file=None, **values)  # type: ignore[call-arg]


def scripted_model(scenario_id: str) -> ScriptedModel:
    return ScriptedModel.from_file(SCRIPTS_DIR / f"{scenario_id}.yaml")


def tool_call(tool: str, **args: Any) -> dict[str, Any]:
    return {"type": "tool_use", "name": tool, "input": args}


def api_error(status: int, message: str = "error") -> ApiException:
    error = ApiException(status=status, reason="Error")
    error.body = json.dumps({"message": message})
    return error


def gemini_settings() -> AgentSettings:
    return make_settings(llm_provider="gemini", gemini_api_key="g-test")


def mock_http(*responses: httpx.Response) -> tuple[httpx.Client, list[httpx.Request]]:
    """An httpx client that replays responses in order and records every request it sent."""
    sent: list[httpx.Request] = []
    queue = list(responses)

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return queue.pop(0)

    return httpx.Client(transport=httpx.MockTransport(handle)), sent
