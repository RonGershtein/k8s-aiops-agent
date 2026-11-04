import os
from unittest.mock import MagicMock

import pytest

from k8s_aiops_agent.fixture_kube import FixtureKube
from tests.builders import SCENARIOS_DIR

_CONFIG_PREFIXES = ("AGENT_", "LANGFUSE_", "ANTHROPIC_")


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
