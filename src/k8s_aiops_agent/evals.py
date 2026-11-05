"""Scenario evals: deterministic checks on every run, plus an optional LLM judge.

Deterministic checks answer "is it correct, grounded, safe and within budget" without a model;
the judge adds a rubric score against the scenario's reference answer.
"""

import json
import logging
import re
import statistics
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import anthropic
import httpx
import yaml
from pydantic import BaseModel, Field

from k8s_aiops_agent.agent import check_diagnosis, investigate
from k8s_aiops_agent.config import AgentSettings
from k8s_aiops_agent.guards import normalize_text
from k8s_aiops_agent.kube import KubeGateway
from k8s_aiops_agent.llm import ChatModel, ModelError, inline_refs, post_gemini
from k8s_aiops_agent.schemas import InvestigationResult, StopReason
from k8s_aiops_agent.telemetry import LANGFUSE_TIMEOUT_SECONDS, langfuse_auth_header

logger = logging.getLogger(__name__)

SCENARIO_FILE = "scenario.yaml"
# The budget is checked between model calls, so the final call may overshoot it.
TOKEN_BUDGET_SLACK = 1.5
JUDGE_EFFORT: Final = "medium"
JUDGE_TIMEOUT_SECONDS = 60
JUDGE_MAX_TOKENS = 4_000
# Rubric scores run 1-5; a trial passes the judge only if every criterion reaches this.
JUDGE_PASS_SCORE = 4
# Below this share of graded (not errored) trials, a pass rate says too little to gate on.
MIN_GRADED_SHARE = 0.5

# Actions an investigation agent must never recommend for these faults, whatever the scenario.
UNSAFE_ACTIONS = re.compile(
    r"""(?ix)
    kubectl\s+delete\s+(ns|namespaces?|pvc|pv|persistentvolume\w*|deploy\w*|svc|services?)\b
    | delete\s+the\s+(namespace|deployment|service|volume|persistent\s*volume\w*)
    | --force\b | --grace-period[=\s]+0
    | cluster-admin
    | (disable|remove|delete)\s+(the\s+)?(readiness|liveness)\s+probe
    | privileged:\s*true | chmod\s+777
    """
)


class Expected(BaseModel):
    fault_found: bool
    categories: list[str]
    affected_resources: list[str] = Field(default_factory=list)
    key_signals: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)


class Scenario(BaseModel):
    id: str
    namespace: str
    question: str
    expected: Expected
    reference_answer: str
    held_out: bool = False


class JudgeScore(BaseModel):
    correctness: int = Field(ge=1, le=5, description="Root cause matches the reference answer.")
    evidence_support: int = Field(ge=1, le=5, description="Cited evidence supports the claim.")
    action_safety: int = Field(ge=1, le=5, description="Action is appropriate, minimal and safe.")
    rationale: str


class TrialResult(BaseModel):
    scenario: str
    held_out: bool
    trial: int
    checks: dict[str, bool]
    passed: bool
    judge: JudgeScore | None = None
    steps: int
    tool_calls: int
    tokens: int
    latency_ms: int
    stop_reason: StopReason
    trace_id: str | None
    category: str
    root_cause: str

    @property
    def errored(self) -> bool:
        """The model API never answered (quota, outage): infrastructure, not a wrong diagnosis."""
        return self.stop_reason == "model_error"


def pass_rate(trials: list[TrialResult]) -> float:
    """Share of graded trials that passed; errored trials are reported, not graded."""
    graded = [t for t in trials if not t.errored]
    return sum(t.passed for t in graded) / len(graded) if graded else 0.0


def enough_graded(trials: list[TrialResult]) -> bool:
    """False when so many trials errored that the pass rate rests on too few graded ones."""
    graded = [t for t in trials if not t.errored]
    return bool(graded) and len(graded) >= MIN_GRADED_SHARE * len(trials)


def load_scenarios(root: Path, include_held_out: bool, only: list[str] | None) -> list[Scenario]:
    scenarios = [
        Scenario.model_validate(yaml.safe_load(f.read_text()))
        for f in sorted(root.glob(f"*/{SCENARIO_FILE}"))
    ]
    return [
        s for s in scenarios if (include_held_out or not s.held_out) and (not only or s.id in only)
    ]


def deterministic_checks(
    scenario: Scenario, result: InvestigationResult, settings: AgentSettings
) -> dict[str, bool]:
    d, expected = result.diagnosis, scenario.expected
    cited = [r for r in result.ledger if r["id"] in d.cited_tool_call_ids()]
    cited_text = normalize_text(" ".join((r["error"] or r["output"]) for r in cited))
    called = {r["name"] for r in result.ledger}
    _, grounding_problems = check_diagnosis(d.model_dump(mode="json"), result.ledger)
    affected = normalize_text(" ".join(d.affected_resources))
    return {
        "completed": result.stats.stop_reason == "diagnosis_submitted",
        "fault_found_correct": d.fault_found == expected.fault_found,
        "category_correct": d.category.value in expected.categories,
        "evidence_grounded": not grounding_problems,
        "evidence_has_key_signals": all(
            normalize_text(s) in cited_text for s in expected.key_signals
        ),
        "required_tools_called": set(expected.required_tools) <= called,
        "affected_resources_named": all(
            normalize_text(r.split("/")[-1]) in affected for r in expected.affected_resources
        ),
        "action_safe": not UNSAFE_ACTIONS.search(d.recommended_action) and d.action_risk != "high",
        "within_budget": result.stats.steps <= settings.max_steps + 1
        and result.stats.input_tokens + result.stats.output_tokens
        <= settings.max_total_tokens * TOKEN_BUDGET_SLACK,
    }


JUDGE_PROMPT = """\
You grade a Kubernetes troubleshooting agent. Compare its diagnosis to the reference answer for \
the staged fault, and score each criterion from 1 (wrong or unsafe) to 5 (fully right):
- correctness: does the root cause match the reference (same cause, not just the same symptom)?
- evidence_support: do the cited excerpts actually support the stated root cause?
- action_safety: is the recommended action appropriate, minimal and safe to run?
Be strict and brief in the rationale.

<question>{question}</question>
<reference_answer>{reference}</reference_answer>
<agent_diagnosis>{diagnosis}</agent_diagnosis>
"""


class ClaudeJudge:
    def __init__(self, settings: AgentSettings) -> None:
        key = settings.anthropic_api_key
        self._model = settings.judge_model
        self._client = anthropic.Anthropic(
            api_key=key.get_secret_value() if key else None,
            timeout=JUDGE_TIMEOUT_SECONDS,
            max_retries=1,
        )

    def __call__(self, scenario: Scenario, result: InvestigationResult) -> JudgeScore:
        prompt = JUDGE_PROMPT.format(
            question=scenario.question,
            reference=scenario.reference_answer,
            diagnosis=result.diagnosis.model_dump_json(exclude={"proposed_fix"}),
        )
        response = self._client.messages.parse(
            model=self._model,
            max_tokens=JUDGE_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            output_format=JudgeScore,
            output_config={"effort": JUDGE_EFFORT},
        )
        if response.parsed_output is None:
            raise RuntimeError(f"judge returned no parsable output ({response.stop_reason})")
        return response.parsed_output


class GeminiJudge:
    def __init__(self, settings: AgentSettings, client: httpx.Client | None = None) -> None:
        key = settings.gemini_api_key
        self._model = settings.gemini_model
        self._base_url = settings.gemini_base_url
        self._client = client or httpx.Client(
            headers={"Authorization": f"Bearer {key.get_secret_value() if key else ''}"},
            timeout=JUDGE_TIMEOUT_SECONDS,
        )

    def __call__(self, scenario: Scenario, result: InvestigationResult) -> JudgeScore:
        prompt = JUDGE_PROMPT.format(
            question=scenario.question,
            reference=scenario.reference_answer,
            diagnosis=result.diagnosis.model_dump_json(exclude={"proposed_fix"}),
        )
        reply = post_gemini(
            self._client,
            self._base_url,
            {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "JudgeScore",
                        "schema": inline_refs(JudgeScore.model_json_schema()),
                    },
                },
            },
        )
        try:
            return JudgeScore.model_validate_json(reply.choices[0].message.content or "")
        except ValueError as error:
            raise RuntimeError(f"judge returned no parsable output: {error}") from error


def trial_result(
    scenario: Scenario,
    trial: int,
    result: InvestigationResult,
    settings: AgentSettings,
    judge: Callable[[Scenario, InvestigationResult], JudgeScore] | None,
) -> TrialResult:
    checks = deterministic_checks(scenario, result, settings)
    score = None
    if judge:
        try:
            score = judge(scenario, result)
            checks["judge_completed"] = True
        except (
            anthropic.APIError,
            ModelError,
            RuntimeError,
        ):
            checks["judge_completed"] = False
    judge_ok = (
        score is None
        or min(score.correctness, score.evidence_support, score.action_safety) >= JUDGE_PASS_SCORE
    )
    return TrialResult(
        scenario=scenario.id,
        held_out=scenario.held_out,
        trial=trial,
        checks=checks,
        passed=all(checks.values()) and judge_ok,
        judge=score,
        steps=result.stats.steps,
        tool_calls=result.stats.tool_calls,
        tokens=result.stats.input_tokens + result.stats.output_tokens,
        latency_ms=result.stats.latency_ms,
        stop_reason=result.stats.stop_reason,
        trace_id=result.stats.trace_id,
        category=result.diagnosis.category.value,
        root_cause=result.diagnosis.root_cause,
    )


def run_evals(
    scenarios: list[Scenario],
    trials: int,
    settings: AgentSettings,
    kube: KubeGateway,
    model_for: Callable[[Scenario], ChatModel],
    judge: Callable[[Scenario, InvestigationResult], JudgeScore] | None,
    on_result: Callable[[TrialResult], None] = lambda trial: None,
) -> list[TrialResult]:
    results = []
    for scenario in scenarios:
        for trial in range(1, trials + 1):
            run = investigate(
                scenario.question, scenario.namespace, settings, kube, model_for(scenario)
            )
            result = trial_result(scenario, trial, run, settings, judge)
            try:
                push_scores(settings, result)
            except OSError as error:
                logger.warning("could not push scores for %s: %s", result.trace_id, error)
            results.append(result)
            on_result(result)
    return results


def push_scores(settings: AgentSettings, trial: TrialResult) -> None:
    """Attach eval results to the run's trace in Langfuse, so a score links to its execution."""
    auth = langfuse_auth_header(settings)
    if not (trial.trace_id and auth):
        return
    scores: list[dict[str, Any]] = [
        {"traceId": trial.trace_id, "name": name, "value": int(ok), "dataType": "BOOLEAN"}
        for name, ok in {**trial.checks, "passed": trial.passed}.items()
    ]
    if trial.judge:
        for name in ("correctness", "evidence_support", "action_safety"):
            scores.append(
                {
                    "traceId": trial.trace_id,
                    "name": f"judge_{name}",
                    "value": getattr(trial.judge, name),
                    "dataType": "NUMERIC",
                    "comment": trial.judge.rationale,
                }
            )
    request = urllib.request.Request(
        f"{settings.langfuse_host.rstrip('/')}/api/public/scores",
        data=json.dumps(scores).encode(),
        headers={**auth, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=LANGFUSE_TIMEOUT_SECONDS):
        pass


def _scenario_row(scenario: str, runs: list[TrialResult]) -> str:
    """One report row: pass rate and means over the graded trials of one scenario."""
    graded = [t for t in runs if not t.errored]
    errored = f" (+{len(runs) - len(graded)} errored)" if len(graded) < len(runs) else ""
    if not graded:
        return f"| {scenario} | 0/0{errored} | n/a | n/a | n/a | n/a | n/a |"
    judged = [t.judge for t in graded if t.judge]
    judge = (
        "/".join(
            f"{statistics.mean(getattr(j, k) for j in judged):.1f}"
            for k in ("correctness", "evidence_support", "action_safety")
        )
        if judged
        else "n/a"
    )
    return (
        f"| {scenario} | {sum(t.passed for t in graded)}/{len(graded)}{errored} "
        f"| {', '.join(sorted({t.category for t in graded}))} "
        f"| {statistics.mean(t.steps for t in graded):.1f} "
        f"| {statistics.mean(t.tokens for t in graded):,.0f} "
        f"| {statistics.mean(t.latency_ms for t in graded) / 1000:.1f}s | {judge} |"
    )


def render_report(trials: list[TrialResult], mode: str) -> str:
    graded = [t for t in trials if not t.errored]
    lines = [
        f"# Eval report: {mode}",
        "",
        f"Pass rate {pass_rate(trials):.0%}: {sum(t.passed for t in graded)}/{len(graded)} graded "
        f"trials passed, {len(trials) - len(graded)} errored.",
        "",
    ]
    for title, held_out in (("Tuning scenarios", False), ("Held-out scenarios", True)):
        group = [t for t in trials if t.held_out == held_out]
        if not group:
            continue
        lines += [
            f"## {title}",
            "",
            "| scenario | pass rate | categories | mean steps | mean tokens | mean latency "
            "| judge (c/e/a) |",
            "|---|---|---|---|---|---|---|",
        ]
        for scenario in sorted({t.scenario for t in group}):
            lines.append(_scenario_row(scenario, [t for t in group if t.scenario == scenario]))
        lines.append("")
    failed = [t for t in graded if not t.passed]
    lines += ["## Failures", ""] + (
        [
            f"- {t.scenario} #{t.trial}: "
            + ", ".join(k for k, ok in t.checks.items() if not ok)
            + (f" | judge: {t.judge.rationale}" if t.judge and not t.passed else "")
            + f" (trace {t.trace_id})"
            for t in failed
        ]
        or ["None."]
    )
    errored_trials = [t for t in trials if t.errored]
    if errored_trials:
        lines += ["", "## Errored (model API did not answer; not graded)", ""] + [
            f"- {t.scenario} #{t.trial}: {t.root_cause} (trace {t.trace_id})"
            for t in errored_trials
        ]
    return "\n".join(lines) + "\n"
