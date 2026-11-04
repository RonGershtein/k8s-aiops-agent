import json
from pathlib import Path

import pytest

from k8s_aiops_agent.fixture_kube import FIXTURE_FILE, FixtureKube, record_namespace
from k8s_aiops_agent.kube import KubeError
from tests.builders import SCENARIOS_DIR

RECORDED = sorted(SCENARIOS_DIR.glob(f"*/{FIXTURE_FILE}"))


def first_pod(kube: FixtureKube, namespace: str) -> str:
    return str(kube.list_pods(namespace)[0]["name"])


@pytest.mark.parametrize("fixture_file", RECORDED, ids=lambda p: p.parent.name)
def test_record_from_replay_reproduces_recording(kube: FixtureKube, fixture_file: Path) -> None:
    original = json.loads(fixture_file.read_text())

    rerecorded = record_namespace(kube, original["namespace"])

    assert json.loads(json.dumps(rerecorded, default=str)) == original


def test_fixture_kube_unknown_pod_raises_not_found(kube: FixtureKube) -> None:
    with pytest.raises(KubeError) as raised:
        kube.describe_pod("scenario-01", "no-such-pod")

    assert raised.value.status == 404


def test_fixture_kube_recorded_api_error_is_replayed(kube: FixtureKube) -> None:
    with pytest.raises(KubeError, match='"orders-config" not found'):
        kube.get_configmap("scenario-05", "orders-config")


def test_fixture_kube_unknown_namespace_looks_empty(kube: FixtureKube) -> None:
    assert kube.list_pods("team-x") == []


def test_fixture_events_filter_by_object(
    kube: FixtureKube,
) -> None:
    pod = first_pod(kube, "scenario-02")

    events = kube.get_events("scenario-02", involved_object=pod)

    assert events
    assert {e["object"] for e in events} == {f"pod/{pod}"}


def test_fixture_kube_logs_respect_tail_lines(kube: FixtureKube) -> None:
    pod = first_pod(kube, "scenario-00")
    tail_lines = 3

    logs = kube.get_pod_logs("scenario-00", pod, tail_lines=tail_lines)

    assert len(logs.splitlines()) == tail_lines


def test_fixture_crashloop_logs_show_config_error(kube: FixtureKube) -> None:
    pod = first_pod(kube, "scenario-01")

    logs = kube.get_pod_logs("scenario-01", pod, previous=True)

    assert 'PAYMENT_TIMEOUT_MS="30s"' in logs
