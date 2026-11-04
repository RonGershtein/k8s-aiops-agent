"""The one supported write action: replace a Service's label selector, after human approval."""

from pathlib import Path
from typing import Any, Protocol

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

from k8s_aiops_agent.kube import Json, KubeError, KubeGateway
from k8s_aiops_agent.schemas import RemediationProposal, ServiceSelectorFix


class SelectorPatcher(Protocol):
    def patch_service_selector(
        self, namespace: str, service: str, selector: dict[str, str]
    ) -> None: ...


class LiveSelectorPatcher:
    """Uses the remediator identity, which RBAC limits to patching Services in one namespace."""

    def __init__(self, core: Any) -> None:
        self._core = core

    @classmethod
    def from_kubeconfig(cls, kubeconfig: Path) -> "LiveSelectorPatcher":
        api = config.new_client_from_config(config_file=str(kubeconfig))
        return cls(client.CoreV1Api(api))

    def patch_service_selector(
        self, namespace: str, service: str, selector: dict[str, str]
    ) -> None:
        # JSON patch "replace" swaps the whole selector, so stale keys cannot survive a merge.
        body = [{"op": "replace", "path": "/spec/selector", "value": selector}]
        try:
            self._core.patch_namespaced_service(service, namespace, body, _request_timeout=10)
        except ApiException as error:
            raise KubeError(error.status or 0, str(error.reason)) from error


def build_proposal(
    kube: KubeGateway, namespace: str, fix: ServiceSelectorFix
) -> RemediationProposal | str:
    """A proposal, or the reason no safe one exists. Preconditions are checked on fresh reads."""
    if not fix.proposed_selector:
        return "the proposed selector is empty; an empty selector would match nothing"
    try:
        service = kube.get_service_and_endpoints(namespace, fix.service)
        pods = kube.list_pods(namespace)
    except KubeError as error:
        return f"could not read current state: {error}"
    current: dict[str, str] = service["selector"]
    if current == fix.proposed_selector:
        return "the Service already uses the proposed selector"
    matching = [p["name"] for p in pods if _selects(fix.proposed_selector, p) and _is_ready(p)]
    if not matching:
        return f"selector {fix.proposed_selector} matches no Ready pods, so patching would not help"
    return RemediationProposal(
        namespace=namespace,
        service=fix.service,
        current_selector=current,
        proposed_selector=fix.proposed_selector,
        matching_ready_pods=matching,
        expected_effect=(
            f"Service {fix.service} will route to {len(matching)} Ready pod(s) "
            f"({', '.join(matching)}) and its ready endpoints will no longer be empty."
        ),
    )


def endpoints_cover(service: Json, pods: list[str]) -> bool:
    return set(pods) <= set(service.get("ready_endpoints", []))


def _selects(selector: dict[str, str], pod: Json) -> bool:
    labels = pod.get("labels") or {}
    return all(labels.get(key) == value for key, value in selector.items())


def _is_ready(pod: Json) -> bool:
    ready, _, total = str(pod.get("ready", "0/0")).partition("/")
    return ready == total and total not in ("", "0")
