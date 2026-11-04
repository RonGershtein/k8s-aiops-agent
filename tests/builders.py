"""Small builders shared by the tests. Fixtures live in conftest.py."""

import json
from pathlib import Path
from typing import Any

from kubernetes.client.exceptions import ApiException

from k8s_aiops_agent.config import AgentSettings
from k8s_aiops_agent.llm import ScriptedModel

SCENARIOS_DIR = Path("scenarios")
SCRIPTS_DIR = Path("evals/scripted")
SCENARIO_NAMESPACES = [f"scenario-0{n}" for n in range(6)]


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
