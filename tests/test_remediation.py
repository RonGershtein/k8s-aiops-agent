from unittest.mock import MagicMock

import pytest
from urllib3.exceptions import MaxRetryError

from k8s_aiops_agent.kube import REQUEST_TIMEOUT_SECONDS, KubeError
from k8s_aiops_agent.remediation import LiveSelectorPatcher
from tests.builders import api_error


def test_patch_selector_replaces_only_if_unchanged(
    core: MagicMock,
) -> None:
    patcher = LiveSelectorPatcher(core)

    patcher.patch_service_selector("ns", "frontend", {"app": "frontend-v2"}, {"app": "frontend"})

    core.patch_namespaced_service.assert_called_once_with(
        "frontend",
        "ns",
        [
            {"op": "test", "path": "/spec/selector", "value": {"app": "frontend-v2"}},
            {"op": "replace", "path": "/spec/selector", "value": {"app": "frontend"}},
        ],
        _request_timeout=REQUEST_TIMEOUT_SECONDS,
    )


@pytest.mark.parametrize(
    "failure",
    [api_error(403, "forbidden"), MaxRetryError(None, "https://127.0.0.1:6443")],
    ids=["api-error", "unreachable"],
)
def test_patch_service_selector_failure_raises_kube_error(
    core: MagicMock, failure: Exception
) -> None:
    core.patch_namespaced_service.side_effect = failure

    with pytest.raises(KubeError):
        LiveSelectorPatcher(core).patch_service_selector("ns", "frontend", {}, {"app": "frontend"})
