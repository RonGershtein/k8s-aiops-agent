"""The agent's read-only tools as an MCP server, for other agents or IDEs.

Same tool specs, guards and identity as the agent itself: the allowlist, redaction, truncation and
RBAC apply unchanged, and there is no write tool here (the selector fix stays approval-gated in the
agent).
"""

import inspect
import itertools
from collections.abc import Callable, Collection
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from k8s_aiops_agent.kube import KubeGateway
from k8s_aiops_agent.tools import TOOL_SPECS, ToolSpec, run_tool


def build_mcp_server(
    kube: KubeGateway, allowed_namespaces: Collection[str], max_output_chars: int
) -> MCPServer:
    server = MCPServer("k8s-aiops-readonly")
    ids = itertools.count(1)
    for spec in TOOL_SPECS:
        server.add_tool(
            _tool_function(spec, kube, allowed_namespaces, max_output_chars, ids),
            name=spec.name,
            description=spec.description,
            annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
        )
    return server


def _tool_function(
    spec: ToolSpec,
    kube: KubeGateway,
    allowed_namespaces: Collection[str],
    max_output_chars: int,
    ids: Any,
) -> Callable[..., str]:
    def call(**arguments: Any) -> str:
        record = run_tool(
            kube, f"m{next(ids)}", spec.name, arguments, allowed_namespaces, max_output_chars
        )
        if record.error:
            raise ToolError(record.error)
        return record.output

    # MCP derives each tool's input schema from the signature; expose the spec's typed fields.
    fields = spec.args.model_fields
    call.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=inspect.Parameter.empty if field.is_required() else field.default,
                annotation=Annotated[field.annotation, field],
            )
            for name, field in fields.items()
        ],
        return_annotation=str,
    )
    call.__annotations__ = {n: Annotated[f.annotation, f] for n, f in fields.items()} | {
        "return": str
    }
    call.__name__ = spec.name
    return call
