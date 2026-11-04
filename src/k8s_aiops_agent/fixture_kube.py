"""Record a namespace through any KubeGateway, and replay it offline with the same interface."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from k8s_aiops_agent.kube import (
    DEFAULT_LOG_LINES,
    PREVIOUS_LOGS_UNAVAILABLE,
    Json,
    KubeError,
    KubeGateway,
)

FIXTURE_FILE = "fixture.json"
RECORDED_LOG_LINES = 50
_ERROR_KEY = "__error__"


def record_namespace(kube: KubeGateway, namespace: str) -> Json:
    pod_list = kube.list_pods(namespace)
    deployment_list = kube.list_deployments(namespace)
    service_list = kube.list_services(namespace)
    pods = [p["name"] for p in pod_list]
    deployments = [d["name"] for d in deployment_list]
    services = [s["name"] for s in service_list]
    deployment_details = {n: _record_call(kube.get_deployment, namespace, n) for n in deployments}
    return {
        "namespace": namespace,
        "pods": pod_list,
        "deployments": deployment_list,
        "services": service_list,
        "pod_details": {n: _record_call(kube.describe_pod, namespace, n) for n in pods},
        "logs": {
            n: {
                kind: _record_call(
                    kube.get_pod_logs,
                    namespace,
                    n,
                    previous=kind == "previous",
                    tail_lines=RECORDED_LOG_LINES,
                )
                for kind in ("current", "previous")
            }
            for n in pods
        },
        "events": kube.get_events(namespace),
        "deployment_details": deployment_details,
        "service_details": {
            n: _record_call(kube.get_service_and_endpoints, namespace, n) for n in services
        },
        "configmaps": {
            n: _record_call(kube.get_configmap, namespace, n)
            for n in sorted(_referenced_configmaps(deployment_details))
        },
    }


class FixtureKube:
    def __init__(self, recordings: list[Json]) -> None:
        self._by_namespace = {r["namespace"]: r for r in recordings}

    @classmethod
    def from_dir(cls, fixtures_dir: Path) -> "FixtureKube":
        files = sorted(fixtures_dir.glob(f"*/{FIXTURE_FILE}"))
        return cls([json.loads(f.read_text()) for f in files])

    def list_pods(self, namespace: str) -> list[Json]:
        return list(self._namespace_data(namespace).get("pods", []))

    def list_deployments(self, namespace: str) -> list[Json]:
        return list(self._namespace_data(namespace).get("deployments", []))

    def list_services(self, namespace: str) -> list[Json]:
        return list(self._namespace_data(namespace).get("services", []))

    def describe_pod(self, namespace: str, name: str) -> Json:
        return dict(self._recorded_object(namespace, "pod_details", "pods", name))

    def get_pod_logs(
        self,
        namespace: str,
        name: str,
        container: str | None = None,
        previous: bool = False,
        tail_lines: int = DEFAULT_LOG_LINES,
    ) -> str:
        # shortcut: `container` is ignored because every scenario pod has one container;
        # record per container if a multi-container scenario is added.
        logs = self._recorded_object(namespace, "logs", "pods", name)
        text = str(_replay_entry(logs["previous" if previous else "current"]))
        note = PREVIOUS_LOGS_UNAVAILABLE if text.startswith(PREVIOUS_LOGS_UNAVAILABLE) else ""
        lines = text.removeprefix(note).splitlines()[-tail_lines:]
        return note + "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    def get_events(self, namespace: str, involved_object: str | None = None) -> list[Json]:
        events: list[Json] = self._namespace_data(namespace).get("events", [])
        if involved_object is None:
            return list(events)
        return [e for e in events if e["object"].split("/", 1)[-1] == involved_object]

    def get_deployment(self, namespace: str, name: str) -> Json:
        return dict(
            self._recorded_object(namespace, "deployment_details", "deployments.apps", name)
        )

    def get_service_and_endpoints(self, namespace: str, name: str) -> Json:
        return dict(self._recorded_object(namespace, "service_details", "services", name))

    def get_configmap(self, namespace: str, name: str) -> Json:
        return dict(self._recorded_object(namespace, "configmaps", "configmaps", name))

    def _namespace_data(self, namespace: str) -> Json:
        return self._by_namespace.get(namespace, {})

    def _recorded_object(self, namespace: str, section: str, kind: str, name: str) -> Any:
        entry = self._namespace_data(namespace).get(section, {}).get(name)
        if entry is None:
            raise KubeError(404, f'{kind} "{name}" not found')
        return _replay_entry(entry)


def _record_call(read: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Records API errors too, so replay fails exactly where the live cluster failed."""
    try:
        return read(*args, **kwargs)
    except KubeError as error:
        return {_ERROR_KEY: {"status": error.status, "message": error.message}}


def _replay_entry(entry: Any) -> Any:
    if isinstance(entry, dict) and _ERROR_KEY in entry:
        raise KubeError(entry[_ERROR_KEY]["status"], entry[_ERROR_KEY]["message"])
    return entry


def _referenced_configmaps(deployment_details: Json) -> set[str]:
    names: set[str] = set()

    def collect_refs(node: Any) -> None:
        if isinstance(node, dict):
            for key in ("configMapRef", "configMapKeyRef"):
                if isinstance(node.get(key), dict) and "name" in node[key]:
                    names.add(node[key]["name"])
            for value in node.values():
                collect_refs(value)
        elif isinstance(node, list):
            for value in node:
                collect_refs(value)

    collect_refs(deployment_details)
    return names
