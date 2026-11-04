"""The agent's tool surface: typed, read-only, guarded, and recorded in an evidence ledger."""

import json
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from k8s_aiops_agent.guards import ToolForbiddenError, ensure_namespace_allowed, redact, truncate
from k8s_aiops_agent.kube import DEFAULT_LOG_LINES, KubeError, KubeGateway
from k8s_aiops_agent.schemas import Diagnosis

SUBMIT_DIAGNOSIS = "submit_diagnosis"
MAX_LOG_LINES = 500


class NamespaceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    namespace: str


class NamedArgs(NamespaceArgs):
    name: str


class LogArgs(NamedArgs):
    container: str | None = Field(default=None, description="Only needed for multi-container pods.")
    previous: bool = Field(
        default=False, description="Logs of the previous (crashed) container instance."
    )
    tail_lines: int = Field(default=DEFAULT_LOG_LINES, ge=1, le=MAX_LOG_LINES)


class EventArgs(NamespaceArgs):
    involved_object: str | None = Field(
        default=None, description="Only events about this object name, e.g. a pod name."
    )


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args: type[NamespaceArgs]
    run: Callable[[KubeGateway, Any], Any]


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "list_pods",
        "List pods in a namespace with phase, readiness, restarts, waiting reasons and labels.",
        NamespaceArgs,
        lambda k, a: k.list_pods(a.namespace),
    ),
    ToolSpec(
        "list_deployments",
        "List deployments in a namespace with desired/ready replicas and images.",
        NamespaceArgs,
        lambda k, a: k.list_deployments(a.namespace),
    ),
    ToolSpec(
        "list_services",
        "List services in a namespace with their type and label selector.",
        NamespaceArgs,
        lambda k, a: k.list_services(a.namespace),
    ),
    ToolSpec(
        "describe_pod",
        "Pod details: conditions, container states and last termination, restart counts, spec.",
        NamedArgs,
        lambda k, a: k.describe_pod(a.namespace, a.name),
    ),
    ToolSpec(
        "get_pod_logs",
        "Tail of a pod's container logs. Use previous=true for a container that crashed.",
        LogArgs,
        lambda k, a: k.get_pod_logs(a.namespace, a.name, a.container, a.previous, a.tail_lines),
    ),
    ToolSpec(
        "get_events",
        "Recent namespace events (scheduling, image pulls, probe failures, config errors).",
        EventArgs,
        lambda k, a: k.get_events(a.namespace, a.involved_object),
    ),
    ToolSpec(
        "get_deployment",
        "Deployment spec and status: replicas, selector, pod template labels, containers, probes.",
        NamedArgs,
        lambda k, a: k.get_deployment(a.namespace, a.name),
    ),
    ToolSpec(
        "get_service_and_endpoints",
        "A Service's selector and ports plus its ready and not-ready endpoints.",
        NamedArgs,
        lambda k, a: k.get_service_and_endpoints(a.namespace, a.name),
    ),
    ToolSpec(
        "get_configmap",
        "A ConfigMap's data. Secrets are never readable.",
        NamedArgs,
        lambda k, a: k.get_configmap(a.namespace, a.name),
    ),
)
_SPECS_BY_NAME = {spec.name: spec for spec in TOOL_SPECS}


def anthropic_tool_definitions() -> list[dict[str, Any]]:
    tools = [
        {"name": s.name, "description": s.description, "input_schema": s.args.model_json_schema()}
        for s in TOOL_SPECS
    ]
    tools.append(
        {
            "name": SUBMIT_DIAGNOSIS,
            "description": (
                "Submit the final structured diagnosis. Call exactly once, when the evidence "
                "supports a conclusion (or supports that no fault exists)."
            ),
            "input_schema": Diagnosis.model_json_schema(),
        }
    )
    return tools


@dataclass(frozen=True)
class ToolRecord:
    id: str
    name: str
    args: dict[str, Any]
    output: str
    error: str | None
    latency_ms: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def run_tool(
    kube: KubeGateway,
    record_id: str,
    name: str,
    raw_args: dict[str, Any],
    allowed_namespaces: Collection[str],
    max_output_chars: int,
) -> ToolRecord:
    """Never raises: every failure becomes an error record the model (and the trace) can see."""
    started = time.monotonic()
    output, error = "", None
    try:
        spec = _SPECS_BY_NAME.get(name)
        if spec is None:
            raise ToolForbiddenError(f"unknown tool '{name}'")
        args = spec.args.model_validate(raw_args)
        ensure_namespace_allowed(args.namespace, allowed_namespaces)
        result = spec.run(kube, args)
        text = result if isinstance(result, str) else json.dumps(result, default=str)
        output = truncate(redact(text), max_output_chars)
    except ValidationError as exc:
        error = f"invalid arguments: {exc.errors(include_url=False)}"
    except (ToolForbiddenError, KubeError) as exc:
        error = str(exc)
    return ToolRecord(
        id=record_id,
        name=name,
        args=raw_args,
        output=output,
        error=error,
        latency_ms=int((time.monotonic() - started) * 1000),
    )


def shown_output(error: str | None, output: str) -> str:
    """The tool output exactly as the model sees it, so its quotes can be checked against it."""
    return f"ERROR: {error}" if error else output


def tool_result_text(record: ToolRecord) -> str:
    body = shown_output(record.error, record.output)
    # Tool output is untrusted cluster data; the wrapper tells the model where it starts and ends.
    return f"evidence_id: {record.id}\n<tool_output>\n{body}\n</tool_output>"
