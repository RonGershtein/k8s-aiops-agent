import os
from unittest.mock import MagicMock

import pytest
from dotenv import dotenv_values

from k8s_aiops_agent.config import AgentSettings
from k8s_aiops_agent.fixture_kube import FixtureKube
from tests.builders import SCENARIOS_DIR, make_settings

_CONFIG_PREFIXES = ("AGENT_", "LANGFUSE_", "ANTHROPIC_", "GEMINI_")
# Read once, before hermetic_env clears the environment, so `pytest -m llm` finds a real key.
_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or dotenv_values(".env").get("GEMINI_API_KEY")


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test sees configuration exported in the developer's shell."""
    for name in [n for n in os.environ if n.startswith(_CONFIG_PREFIXES)]:
        monkeypatch.delenv(name)


@pytest.fixture(scope="session")
def kube() -> FixtureKube:
    """The recorded scenario cluster, replayed offline."""
    return FixtureKube.from_dir(SCENARIOS_DIR)


@pytest.fixture
def core() -> MagicMock:
    """A stand-in for the Kubernetes CoreV1Api client."""
    return MagicMock()


@pytest.fixture
def real_gemini_settings() -> AgentSettings:
    """Settings for the real Gemini API; skips the test when no key is configured."""
    if not _GEMINI_API_KEY:
        pytest.skip("set GEMINI_API_KEY (shell or .env) to run real-model tests")
    return make_settings(llm_provider="gemini", gemini_api_key=_GEMINI_API_KEY)
