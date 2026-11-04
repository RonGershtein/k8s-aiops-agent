import asyncio
from pathlib import Path

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from k8s_aiops_agent.fixture_kube import FixtureKube
from k8s_aiops_agent.mcp_server import build_mcp_server
from k8s_aiops_agent.tools import TOOL_SPECS


@pytest.fixture(scope="module")
def server():  # type: ignore[no-untyped-def]
    return build_mcp_server(FixtureKube.from_dir(Path("scenarios")), ["scenario-01"], 6_000)


def test_mcp_server_exposes_only_the_read_only_agent_tools(server) -> None:  # type: ignore[no-untyped-def]
    tools = asyncio.run(server.list_tools())

    assert sorted(t.name for t in tools) == sorted(s.name for s in TOOL_SPECS)
    assert all(t.annotations.read_only_hint for t in tools)


def test_mcp_server_tool_call_returns_guarded_output(server) -> None:  # type: ignore[no-untyped-def]
    result = asyncio.run(server.call_tool("list_pods", {"namespace": "scenario-01"}))

    assert "CrashLoopBackOff" in result.content[0].text


def test_mcp_server_namespace_outside_allowlist_is_refused(server) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ToolError, match="not allowed"):
        asyncio.run(server.call_tool("list_pods", {"namespace": "kube-system"}))
