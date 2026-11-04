from pathlib import Path
from typing import Any

import pytest

from k8s_aiops_agent import agent as agent_module
from k8s_aiops_agent.agent import Remediation, check_diagnosis, investigate
from k8s_aiops_agent.config import AgentSettings
from k8s_aiops_agent.fixture_kube import FixtureKube
from k8s_aiops_agent.llm import ModelError, ScriptedModel
from k8s_aiops_agent.schemas import FaultCategory, RemediationProposal

SCRIPTS = Path("evals/scripted")


def settings(**overrides: Any) -> AgentSettings:
    values: dict[str, Any] = {"allowed_namespaces": ["scenario-01", "scenario-03"], **overrides}
    return AgentSettings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture(scope="module")
def kube() -> FixtureKube:
    return FixtureKube.from_dir(Path("scenarios"))


def call(tool: str, **args: Any) -> dict[str, Any]:
    return {"type": "tool_use", "name": tool, "input": args}


def submit(evidence: list[dict[str, str]], **overrides: Any) -> dict[str, Any]:
    diagnosis = {
        "fault_found": True,
        "category": "crash_loop",
        "root_cause": "checkout crashes",
        "evidence": evidence,
        "recommended_action": "fix config",
        "action_risk": "low",
        "confidence": 0.8,
        **overrides,
    }
    return call("submit_diagnosis", **diagnosis)


GOOD_EVIDENCE = [{"claim": "crash loop", "tool_call_id": "t1", "excerpt": "CrashLoopBackOff"}]
LIST_PODS = call("list_pods", namespace="scenario-01")


def test_investigate_scripted_crashloop_returns_grounded_diagnosis(kube: FixtureKube) -> None:
    model = ScriptedModel.from_file(SCRIPTS / "01-crashloop-bad-config.yaml")

    result = investigate("Why is checkout unhealthy?", "scenario-01", settings(), kube, model)

    assert result.stats.stop_reason == "diagnosis_submitted"
    assert result.diagnosis.category is FaultCategory.BAD_CONFIGURATION
    assert result.diagnosis.cited_tool_call_ids() <= {r["id"] for r in result.ledger}


def test_investigate_model_that_never_submits_stops_at_max_steps_inconclusive(
    kube: FixtureKube,
) -> None:
    turns = [
        [call("get_pod_logs", namespace="scenario-01", name="p", tail_lines=n)]
        for n in range(1, 20)
    ]

    result = investigate("q", "scenario-01", settings(max_steps=3), kube, ScriptedModel(turns))

    assert result.stats.stop_reason == "max_steps"
    assert result.stats.steps == 4  # the budget plus one finalize turn
    assert result.diagnosis.category is FaultCategory.OTHER
    assert result.diagnosis.confidence == 0.0


def test_investigate_after_budget_accepts_a_final_diagnosis(kube: FixtureKube) -> None:
    turns = [[LIST_PODS], [call("list_services", namespace="scenario-01")], [submit(GOOD_EVIDENCE)]]

    result = investigate("q", "scenario-01", settings(max_steps=2), kube, ScriptedModel(turns))

    assert result.stats.stop_reason == "max_steps"
    assert result.diagnosis.category is FaultCategory.CRASH_LOOP


def test_investigate_repeating_the_same_call_stops_for_repeated_calls(kube: FixtureKube) -> None:
    turns = [[LIST_PODS]] * 10

    result = investigate("q", "scenario-01", settings(), kube, ScriptedModel(turns))

    assert result.stats.stop_reason == "repeated_calls"
    assert result.stats.tool_calls == 1


def test_investigate_token_budget_stops_the_run(kube: FixtureKube) -> None:
    turns = [
        [call("list_pods", namespace="scenario-01")],
        [call("list_services", namespace="scenario-01")],
    ]

    result = investigate(
        "q", "scenario-01", settings(max_total_tokens=1_000), kube, ScriptedModel(turns)
    )

    assert result.stats.stop_reason == "token_budget"


def test_investigate_ungrounded_excerpt_is_repaired_once(kube: FixtureKube) -> None:
    bad = [{"claim": "x", "tool_call_id": "t1", "excerpt": "OOMKilled"}]
    turns = [[LIST_PODS], [submit(bad)], [submit(GOOD_EVIDENCE)]]

    result = investigate("q", "scenario-01", settings(), kube, ScriptedModel(turns))

    assert result.stats.stop_reason == "diagnosis_submitted"
    assert result.diagnosis.evidence[0].excerpt == "CrashLoopBackOff"


def test_investigate_repeatedly_ungrounded_diagnosis_falls_back(kube: FixtureKube) -> None:
    bad = [{"claim": "x", "tool_call_id": "t9", "excerpt": "anything"}]
    turns = [[LIST_PODS], [submit(bad)], [submit(bad)], [submit(bad)]]

    result = investigate("q", "scenario-01", settings(), kube, ScriptedModel(turns))

    assert result.stats.stop_reason == "invalid_diagnosis"
    assert result.diagnosis.category is FaultCategory.OTHER


def test_investigate_model_error_returns_inconclusive_instead_of_crashing(
    kube: FixtureKube,
) -> None:
    class FailingModel:
        name = "failing"

        def create(self, *args: Any) -> Any:
            raise ModelError("Claude API error 529: overloaded")

    result = investigate("q", "scenario-01", settings(), kube, FailingModel())

    assert result.stats.stop_reason == "model_error"
    assert result.diagnosis.confidence == 0.0


def test_investigate_namespace_outside_allowlist_is_recorded_as_tool_error(
    kube: FixtureKube,
) -> None:
    turns = [[call("list_pods", namespace="kube-system")], [submit(GOOD_EVIDENCE)]]

    result = investigate("q", "scenario-01", settings(), kube, ScriptedModel(turns))

    assert "not allowed" in result.ledger[0]["error"]


@pytest.mark.parametrize(
    ("evidence", "problem"),
    [
        ({"tool_call_id": "t7", "excerpt": "x"}, "unknown id 't7'"),
        ({"tool_call_id": "t1", "excerpt": "not in output"}, "not verbatim"),
    ],
)
def test_check_diagnosis_rejects_unsupported_evidence(
    evidence: dict[str, str], problem: str
) -> None:
    ledger = [{"id": "t1", "output": '{"phase": "Running"}', "error": None}]
    raw = submit([{"claim": "c", **evidence}])["input"]

    diagnosis, problems = check_diagnosis(raw, ledger)

    assert diagnosis is None
    assert problem in problems[0]


def test_check_diagnosis_accepts_excerpt_copied_from_json_escaped_output() -> None:
    ledger = [
        {"id": "t1", "output": '{"log": "PAYMENT_TIMEOUT_MS=\\"30s\\" invalid"}', "error": None}
    ]
    raw = submit([{"claim": "c", "tool_call_id": "t1", "excerpt": 'PAYMENT_TIMEOUT_MS="30s"'}])

    diagnosis, problems = check_diagnosis(raw["input"], ledger)

    assert problems == []
    assert diagnosis is not None


class FakeCluster:
    """Just enough of a gateway for the remediation path; patching changes what reads return."""

    def __init__(self, ready: bool = True) -> None:
        self.selector = {"app": "frontend-v2"}
        self.ready = ready
        self.patches: list[dict[str, str]] = []

    def list_pods(self, namespace: str) -> list[dict[str, Any]]:
        status = "1/1" if self.ready else "0/1"
        return [{"name": "frontend-a", "labels": {"app": "frontend"}, "ready": status}]

    def get_service_and_endpoints(self, namespace: str, name: str) -> dict[str, Any]:
        routed = self.selector == {"app": "frontend"} and self.ready
        return {
            "name": name,
            "selector": self.selector,
            "ready_endpoints": ["frontend-a"] if routed else [],
        }

    def patch_service_selector(
        self, namespace: str, service: str, selector: dict[str, str]
    ) -> None:
        self.patches.append(selector)
        self.selector = selector


def selector_fix_turns() -> list[list[dict[str, Any]]]:
    evidence = [{"claim": "no endpoints", "tool_call_id": "t1", "excerpt": "frontend-v2"}]
    fix = {"service": "frontend", "proposed_selector": {"app": "frontend"}}
    return [
        [call("get_service_and_endpoints", namespace="scenario-03", name="frontend")],
        [submit(evidence, category="service_selector_mismatch", proposed_fix=fix)],
    ]


@pytest.fixture
def no_verify_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_module, "VERIFY_INTERVAL_SECONDS", 0)


@pytest.mark.usefixtures("no_verify_wait")
def test_remediation_approved_patches_and_verifies_with_fresh_evidence() -> None:
    cluster = FakeCluster()
    seen: list[RemediationProposal] = []

    def approve(proposal: RemediationProposal) -> bool:
        seen.append(proposal)
        return True

    result = investigate(
        "q",
        "scenario-03",
        settings(),
        cluster,
        ScriptedModel(selector_fix_turns()),  # type: ignore[arg-type]
        Remediation(patcher=cluster, approve=approve),
    )

    assert seen[0].current_selector == {"app": "frontend-v2"}
    assert cluster.patches == [{"app": "frontend"}]
    assert result.remediation is not None
    assert result.remediation.resolved
    assert result.remediation.verification[0].tool_call_id.startswith("v")


@pytest.mark.usefixtures("no_verify_wait")
def test_remediation_declined_changes_nothing() -> None:
    cluster = FakeCluster()

    result = investigate(
        "q",
        "scenario-03",
        settings(),
        cluster,
        ScriptedModel(selector_fix_turns()),  # type: ignore[arg-type]
        Remediation(patcher=cluster, approve=lambda proposal: False),
    )

    assert cluster.patches == []
    assert result.remediation is not None
    assert not result.remediation.approved


def test_remediation_selector_matching_no_ready_pods_is_never_proposed() -> None:
    cluster = FakeCluster(ready=False)
    asked: list[RemediationProposal] = []

    result = investigate(
        "q",
        "scenario-03",
        settings(),
        cluster,
        ScriptedModel(selector_fix_turns()),  # type: ignore[arg-type]
        Remediation(patcher=cluster, approve=lambda p: asked.append(p) or True),
    )

    assert asked == []
    assert cluster.patches == []
    assert result.remediation is not None
    assert "matches no Ready pods" in result.remediation.message
