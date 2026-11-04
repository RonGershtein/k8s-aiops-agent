from unittest.mock import MagicMock

import pytest
from kubernetes.client.exceptions import ApiException
from urllib3.exceptions import MaxRetryError

from k8s_aiops_agent.kube import KubeError
from k8s_aiops_agent.remediation import LiveSelectorPatcher


def test_patch_service_selector_replaces_the_whole_selector() -> None:
    core = MagicMock()

    LiveSelectorPatcher(core).patch_service_selector("ns", "frontend", {"app": "frontend"})

    args = core.patch_namespaced_service.call_args.args
    assert args[:2] == ("frontend", "ns")
    assert args[2] == [{"op": "replace", "path": "/spec/selector", "value": {"app": "frontend"}}]


@pytest.mark.parametrize(
    "failure",
    [ApiException(status=403, reason="Forbidden"), MaxRetryError(None, "https://127.0.0.1:6443")],
    ids=["api-error", "unreachable"],
)
def test_patch_service_selector_failure_raises_kube_error(failure: Exception) -> None:
    core = MagicMock()
    core.patch_namespaced_service.side_effect = failure

    with pytest.raises(KubeError):
        LiveSelectorPatcher(core).patch_service_selector("ns", "frontend", {"app": "frontend"})
