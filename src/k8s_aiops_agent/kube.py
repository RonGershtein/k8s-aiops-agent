"""Read-only access to the cluster, returned as compact JSON-ready dicts the model can read."""

import json
from pathlib import Path
from typing import Any, Protocol

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from urllib3.exceptions import HTTPError

Json = dict[str, Any]

REQUEST_TIMEOUT_SECONDS = 10
MAX_EVENTS = 30
DEFAULT_LOG_LINES = 100
_LOGS_GONE = "unable to retrieve container logs"
PREVIOUS_LOGS_UNAVAILABLE = "[previous container logs unavailable; showing current container]\n"


class KubeError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"kubernetes API error {status}: {message}")
        self.status = status
        self.message = message


class KubeGateway(Protocol):
    def list_pods(self, namespace: str) -> list[Json]: ...
    def list_deployments(self, namespace: str) -> list[Json]: ...
    def list_services(self, namespace: str) -> list[Json]: ...
    def describe_pod(self, namespace: str, name: str) -> Json: ...
    def get_pod_logs(
        self,
        namespace: str,
        name: str,
        container: str | None = None,
        previous: bool = False,
        tail_lines: int = DEFAULT_LOG_LINES,
    ) -> str: ...
    def get_events(self, namespace: str, involved_object: str | None = None) -> list[Json]: ...
    def get_deployment(self, namespace: str, name: str) -> Json: ...
    def get_service_and_endpoints(self, namespace: str, name: str) -> Json: ...
    def get_configmap(self, namespace: str, name: str) -> Json: ...


class LiveKube:
    def __init__(
        self, core: client.CoreV1Api, apps: client.AppsV1Api, discovery: client.DiscoveryV1Api
    ) -> None:
        self._core = core
        self._apps = apps
        self._discovery = discovery
        self._plain = client.ApiClient().sanitize_for_serialization

    @classmethod
    def from_kubeconfig(cls, kubeconfig: Path) -> "LiveKube":
        api = config.new_client_from_config(config_file=str(kubeconfig))
        return cls(client.CoreV1Api(api), client.AppsV1Api(api), client.DiscoveryV1Api(api))

    def list_pods(self, namespace: str) -> list[Json]:
        pods = call_api(self._core.list_namespaced_pod, namespace)
        return [self._pod_summary(pod) for pod in pods.items]

    def list_deployments(self, namespace: str) -> list[Json]:
        deployments = call_api(self._apps.list_namespaced_deployment, namespace)
        return [
            {
                "name": d.metadata.name,
                "replicas": d.spec.replicas,
                "ready_replicas": d.status.ready_replicas or 0,
                "images": [c.image for c in d.spec.template.spec.containers],
            }
            for d in deployments.items
        ]

    def list_services(self, namespace: str) -> list[Json]:
        services = call_api(self._core.list_namespaced_service, namespace)
        return [
            {"name": s.metadata.name, "type": s.spec.type, "selector": s.spec.selector or {}}
            for s in services.items
        ]

    def describe_pod(self, namespace: str, name: str) -> Json:
        pod = call_api(self._core.read_namespaced_pod, name, namespace)
        return {
            **self._pod_summary(pod),
            "conditions": self._plain(pod.status.conditions or []),
            "container_statuses": self._plain(pod.status.container_statuses or []),
            "containers": [self._container_spec(c) for c in pod.spec.containers],
        }

    def get_pod_logs(
        self,
        namespace: str,
        name: str,
        container: str | None = None,
        previous: bool = False,
        tail_lines: int = DEFAULT_LOG_LINES,
    ) -> str:
        def read_logs(previous: bool) -> str:
            logs: str = call_api(
                self._core.read_namespaced_pod_log,
                name,
                namespace,
                container=container,
                previous=previous,
                tail_lines=tail_lines,
            )
            return logs

        if not previous:
            return read_logs(previous=False)
        try:
            logs = read_logs(previous=True)
            # The kubelet sometimes answers 200 with this message instead of an error status.
            if not logs.startswith(_LOGS_GONE):
                return logs
        except KubeError as error:
            # The kubelet garbage-collects old containers fast; a crash-looping pod's current
            # (already exited) container still holds the crash output.
            if error.status in (403, 404):
                raise
        return PREVIOUS_LOGS_UNAVAILABLE + read_logs(previous=False)

    def get_events(self, namespace: str, involved_object: str | None = None) -> list[Json]:
        selector = f"involvedObject.name={involved_object}" if involved_object else None
        events = call_api(self._core.list_namespaced_event, namespace, field_selector=selector)
        items = sorted(events.items, key=_event_time)[-MAX_EVENTS:]
        return [
            {
                "type": e.type,
                "reason": e.reason,
                "object": f"{e.involved_object.kind.lower()}/{e.involved_object.name}",
                "message": e.message,
                "count": e.count or 1,
                "last_seen": _event_time(e),
            }
            for e in items
        ]

    def get_deployment(self, namespace: str, name: str) -> Json:
        d = call_api(self._apps.read_namespaced_deployment, name, namespace)
        return {
            "name": d.metadata.name,
            "replicas": d.spec.replicas,
            "ready_replicas": d.status.ready_replicas or 0,
            "available_replicas": d.status.available_replicas or 0,
            "selector": d.spec.selector.match_labels,
            "template_labels": d.spec.template.metadata.labels,
            "containers": [self._container_spec(c) for c in d.spec.template.spec.containers],
            "conditions": self._plain(d.status.conditions or []),
        }

    def get_service_and_endpoints(self, namespace: str, name: str) -> Json:
        svc = call_api(self._core.read_namespaced_service, name, namespace)
        slices = call_api(
            self._discovery.list_namespaced_endpoint_slice,
            namespace,
            label_selector=f"kubernetes.io/service-name={name}",
        )
        ready: list[str] = []
        not_ready: list[str] = []
        for endpoint in (e for s in slices.items for e in (s.endpoints or [])):
            target = (
                endpoint.target_ref.name if endpoint.target_ref else ",".join(endpoint.addresses)
            )
            (ready if endpoint.conditions and endpoint.conditions.ready else not_ready).append(
                target
            )
        return {
            "name": svc.metadata.name,
            "type": svc.spec.type,
            "selector": svc.spec.selector or {},
            "ports": self._plain(svc.spec.ports or []),
            "ready_endpoints": ready,
            "not_ready_endpoints": not_ready,
        }

    def get_configmap(self, namespace: str, name: str) -> Json:
        cm = call_api(self._core.read_namespaced_config_map, name, namespace)
        return {"name": cm.metadata.name, "data": cm.data or {}}

    def _pod_summary(self, pod: Any) -> Json:
        statuses = pod.status.container_statuses or []
        return {
            "name": pod.metadata.name,
            "labels": pod.metadata.labels or {},
            "phase": pod.status.phase,
            "ready": f"{sum(1 for s in statuses if s.ready)}/{len(pod.spec.containers)}",
            "restarts": sum(s.restart_count for s in statuses),
            "waiting_reasons": [
                s.state.waiting.reason for s in statuses if s.state and s.state.waiting
            ],
            "node": pod.spec.node_name,
        }

    def _container_spec(self, container: Any) -> Json:
        spec = {
            "name": container.name,
            "image": container.image,
            "command": container.command,
            "args": container.args,
            "ports": self._plain(container.ports or []),
            "env": self._plain(container.env or []),
            "env_from": self._plain(container.env_from or []),
            "readiness_probe": self._plain(container.readiness_probe),
            "liveness_probe": self._plain(container.liveness_probe),
            "resources": self._plain(container.resources),
        }
        return {key: value for key, value in spec.items() if value}


def call_api(method: Any, *args: Any, **kwargs: Any) -> Any:
    """Calls the Kubernetes client with a timeout; every failure becomes a KubeError."""
    try:
        return method(*args, _request_timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
    except ApiException as error:
        raise KubeError(error.status or 0, _api_message(error)) from error
    except HTTPError as error:
        raise KubeError(0, f"cluster unreachable: {error}") from error


def _api_message(error: ApiException) -> str:
    try:
        return str(json.loads(error.body or "")["message"])
    except (TypeError, ValueError, KeyError):
        return str(error.reason)


def _event_time(event: Any) -> str:
    moment = event.last_timestamp or event.event_time or event.metadata.creation_timestamp
    return moment.isoformat() if moment else ""
