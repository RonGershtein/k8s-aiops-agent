import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from kubernetes.client.exceptions import ApiException
from urllib3.exceptions import MaxRetryError

from k8s_aiops_agent.kube import PREVIOUS_LOGS_UNAVAILABLE, KubeError, LiveKube

READONLY_KUBECONFIG = Path(".kube/agent-readonly.kubeconfig")


def live_kube_with_core(core: Any) -> LiveKube:
    return LiveKube(core, MagicMock(), MagicMock())


def api_error(status: int, message: str) -> ApiException:
    error = ApiException(status=status, reason="Error")
    error.body = json.dumps({"message": message})
    return error


def test_get_pod_logs_previous_unavailable_falls_back_to_current_logs() -> None:
    core = MagicMock()
    core.read_namespaced_pod_log.side_effect = [
        api_error(400, "unable to retrieve container logs"),
        "FATAL config",
    ]

    logs = live_kube_with_core(core).get_pod_logs("ns", "pod", previous=True)

    assert logs == PREVIOUS_LOGS_UNAVAILABLE + "FATAL config"


def test_get_pod_logs_previous_forbidden_raises_instead_of_falling_back() -> None:
    core = MagicMock()
    core.read_namespaced_pod_log.side_effect = api_error(403, "forbidden")

    with pytest.raises(KubeError) as raised:
        live_kube_with_core(core).get_pod_logs("ns", "pod", previous=True)

    assert raised.value.status == 403
    assert core.read_namespaced_pod_log.call_count == 1


def test_live_kube_api_error_surfaces_server_message() -> None:
    core = MagicMock()
    core.read_namespaced_config_map.side_effect = api_error(404, 'configmaps "x" not found')

    with pytest.raises(KubeError, match='configmaps "x" not found'):
        live_kube_with_core(core).get_configmap("ns", "x")


def test_live_kube_unreachable_cluster_raises_kube_error() -> None:
    core = MagicMock()
    core.list_namespaced_pod.side_effect = MaxRetryError(None, "https://127.0.0.1:6443")

    with pytest.raises(KubeError, match="cluster unreachable"):
        live_kube_with_core(core).list_pods("ns")


@pytest.fixture
def live_kube() -> LiveKube:
    if not READONLY_KUBECONFIG.exists():
        pytest.skip("run `make cluster-up` first")
    return LiveKube.from_kubeconfig(READONLY_KUBECONFIG)


@pytest.mark.live
def test_live_crashloop_scenario_logs_show_config_error(live_kube: LiveKube) -> None:
    pod = live_kube.list_pods("scenario-01")[0]["name"]

    logs = live_kube.get_pod_logs("scenario-01", pod, previous=True)

    assert 'PAYMENT_TIMEOUT_MS="30s"' in logs


@pytest.mark.live
def test_live_selector_mismatch_service_has_no_ready_endpoints(live_kube: LiveKube) -> None:
    service = live_kube.get_service_and_endpoints("scenario-03", "frontend")

    assert service["selector"] == {"app": "frontend-v2"}
    assert service["ready_endpoints"] == []


@pytest.mark.live
def test_live_discovery_lists_workloads_without_knowing_names(live_kube: LiveKube) -> None:
    deployments = live_kube.list_deployments("scenario-02")
    services = live_kube.list_services("scenario-02")

    assert [d["name"] for d in deployments] == ["catalog"]
    assert [s["name"] for s in services] == ["catalog"]


@pytest.mark.live
def test_live_secrets_are_forbidden_by_rbac(live_kube: LiveKube) -> None:
    with pytest.raises(ApiException) as raised:
        live_kube._core.list_namespaced_secret("scenario-01")

    assert raised.value.status == 403
