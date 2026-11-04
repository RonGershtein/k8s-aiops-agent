import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from k8s_aiops_agent import evals
from k8s_aiops_agent.evals import (
    Expected,
    JudgeScore,
    Scenario,
    deterministic_checks,
    load_scenarios,
    push_scores,
    run_evals,
    trial_result,
)
from k8s_aiops_agent.fixture_kube import FixtureKube
from k8s_aiops_agent.schemas import Diagnosis, InvestigationResult, RunStats
from tests.builders import SCENARIOS_DIR, SCRIPTS_DIR, make_settings, scripted_model

SETTINGS = make_settings()
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
        ({"category": "no_fault", "fault_found": False}, "fault_found_correct"),
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
    scenarios = load_scenarios(SCENARIOS_DIR, include_held_out=False, only=None)

    assert scenarios
    assert not any(s.held_out for s in scenarios)


def test_load_scenarios_every_scenario_has_an_offline_script() -> None:
    scenarios = load_scenarios(SCENARIOS_DIR, include_held_out=True, only=None)

    assert all((SCRIPTS_DIR / f"{s.id}.yaml").exists() for s in scenarios)


def test_trial_result_judge_failure_is_recorded_not_raised() -> None:
    def broken_judge(scenario: Scenario, run: InvestigationResult) -> JudgeScore:
        raise RuntimeError("judge returned no parsable output")

    trial = trial_result(SCENARIO, 1, result(), SETTINGS, broken_judge)

    assert trial.checks["judge_completed"] is False
    assert not trial.passed


def test_run_evals_offline_runs_every_trial_and_reports_each_result(kube: FixtureKube) -> None:
    scenarios = load_scenarios(SCENARIOS_DIR, include_held_out=False, only=None)
    reported: list[str] = []

    results = run_evals(
        scenarios,
        2,
        SETTINGS,
        kube,
        lambda s: scripted_model(s.id),
        judge=None,
        on_result=lambda trial: reported.append(trial.scenario),
    )

    assert len(results) == len(reported) == 2 * len(scenarios)
    assert all(trial.passed for trial in results)


def test_push_scores_attaches_checks_and_judge_scores_to_the_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urlopen = MagicMock()
    monkeypatch.setattr(evals.urllib.request, "urlopen", urlopen)
    settings = make_settings(langfuse_public_key="pk", langfuse_secret_key="sk")
    score = JudgeScore(correctness=5, evidence_support=4, action_safety=5, rationale="ok")
    trial = trial_result(SCENARIO, 1, result(), settings, lambda s, r: score).model_copy(
        update={"trace_id": "abc"}
    )

    push_scores(settings, trial)
    request = urlopen.call_args.args[0]
    scores = json.loads(request.data)
    values = {score["name"]: score["value"] for score in scores}
    trace_ids = {score["traceId"] for score in scores}

    assert request.full_url == "http://localhost:3000/api/public/scores"
    assert request.get_header("Authorization") == "Basic cGs6c2s="
    assert values["passed"] == 1
    assert values["judge_evidence_support"] == 4
    assert trace_ids == {"abc"}


def test_push_scores_without_langfuse_keys_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    urlopen = MagicMock()
    monkeypatch.setattr(evals.urllib.request, "urlopen", urlopen)
    trial = trial_result(SCENARIO, 1, result(), SETTINGS, None)

    push_scores(SETTINGS, trial)

    urlopen.assert_not_called()
