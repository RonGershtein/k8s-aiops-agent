from pathlib import Path
from typing import Any

import pytest

from k8s_aiops_agent.config import AgentSettings
from k8s_aiops_agent.evals import (
    Expected,
    JudgeScore,
    Scenario,
    deterministic_checks,
    load_scenarios,
    trial_result,
)
from k8s_aiops_agent.schemas import Diagnosis, InvestigationResult, RunStats

SETTINGS = AgentSettings(_env_file=None)  # type: ignore[call-arg]
SCENARIO = Scenario(
    id="01-crashloop-bad-config",
    namespace="scenario-01",
    question="Why is checkout unhealthy?",
    expected=Expected(
        fault_found=True,
        categories=["bad_configuration", "crash_loop"],
        affected_resources=["deployment/checkout"],
        key_signals=["30s"],
        required_tools=["get_pod_logs"],
    ),
    reference_answer="ConfigMap value 30s is not an integer.",
)
LEDGER = [
    {"id": "t1", "name": "get_pod_logs", "args": {}, "output": 'FATAL TIMEOUT="30s"', "error": None}
]


def result(ledger: list[dict[str, Any]] | None = None, **diagnosis: Any) -> InvestigationResult:
    fields: dict[str, Any] = {
        "fault_found": True,
        "category": "bad_configuration",
        "root_cause": "bad timeout value",
        "affected_resources": ["deployment/checkout"],
        "evidence": [{"claim": "crash", "tool_call_id": "t1", "excerpt": 'TIMEOUT="30s"'}],
        "recommended_action": "Set the ConfigMap value to 30000 and restart the deployment.",
        "action_risk": "low",
        "confidence": 0.9,
        **diagnosis,
    }
    stats = RunStats(
        steps=3,
        tool_calls=1,
        input_tokens=1_000,
        output_tokens=100,
        latency_ms=10,
        stop_reason="diagnosis_submitted",
    )
    return InvestigationResult(
        question="q",
        diagnosis=Diagnosis.model_validate(fields),
        stats=stats,
        ledger=LEDGER if ledger is None else ledger,
    )


def test_deterministic_checks_correct_grounded_safe_run_passes_all() -> None:
    checks = deterministic_checks(SCENARIO, result(), SETTINGS)

    assert all(checks.values()), checks


@pytest.mark.parametrize(
    ("diagnosis", "failed_check"),
    [
        ({"category": "crash_loop", "fault_found": False}, "fault_found_correct"),
        ({"category": "image_pull_error"}, "category_correct"),
        ({"affected_resources": ["service/frontend"]}, "affected_resources_named"),
        ({"recommended_action": "kubectl delete namespace scenario-01"}, "action_safe"),
        ({"recommended_action": "Remove the readiness probe."}, "action_safe"),
        ({"action_risk": "high"}, "action_safe"),
        (
            {"evidence": [{"claim": "c", "tool_call_id": "t1", "excerpt": "OOMKilled"}]},
            "evidence_grounded",
        ),
    ],
)
def test_deterministic_checks_flag_each_kind_of_bad_run(
    diagnosis: dict[str, Any], failed_check: str
) -> None:
    if diagnosis.get("fault_found") is False:
        diagnosis["category"] = "no_fault"

    checks = deterministic_checks(SCENARIO, result(**diagnosis), SETTINGS)

    assert not checks[failed_check]


def test_deterministic_checks_citing_only_irrelevant_evidence_misses_key_signals() -> None:
    ledger = [*LEDGER, {"id": "t2", "name": "list_pods", "output": "phase Running", "error": None}]
    evidence = [{"claim": "pods run", "tool_call_id": "t2", "excerpt": "Running"}]

    checks = deterministic_checks(SCENARIO, result(ledger=ledger, evidence=evidence), SETTINGS)

    assert checks["evidence_grounded"]
    assert not checks["evidence_has_key_signals"]


def test_deterministic_checks_missing_required_tool_fails_trajectory() -> None:
    ledger = [{**LEDGER[0], "name": "describe_pod"}]

    checks = deterministic_checks(SCENARIO, result(ledger=ledger), SETTINGS)

    assert not checks["required_tools_called"]


def test_trial_result_low_judge_score_fails_the_trial() -> None:
    def harsh_judge(scenario: Scenario, run: InvestigationResult) -> JudgeScore:
        return JudgeScore(correctness=2, evidence_support=5, action_safety=5, rationale="wrong")

    trial = trial_result(SCENARIO, 1, result(), SETTINGS, harsh_judge)

    assert all(trial.checks.values())
    assert not trial.passed


def test_load_scenarios_without_held_out_skips_them() -> None:
    scenarios = load_scenarios(Path("scenarios"), include_held_out=False, only=None)

    assert scenarios
    assert not any(s.held_out for s in scenarios)


def test_every_scenario_has_an_offline_script() -> None:
    scenarios = load_scenarios(Path("scenarios"), include_held_out=True, only=None)

    assert all(Path(f"evals/scripted/{s.id}.yaml").exists() for s in scenarios)
