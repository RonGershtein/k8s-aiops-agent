"""LangGraph investigation agent: tool loop, budgets, validated diagnosis, approval-gated fix."""

import json
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from operator import add
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from k8s_aiops_agent.config import AgentSettings
from k8s_aiops_agent.kube import KubeError, KubeGateway
from k8s_aiops_agent.llm import ChatModel, ModelError
from k8s_aiops_agent.prompts import FINALIZE_PROMPT, NO_TOOL_CALL_PROMPT, SYSTEM_PROMPT
from k8s_aiops_agent.remediation import SelectorPatcher, build_proposal, endpoints_cover
from k8s_aiops_agent.schemas import (
    Diagnosis,
    Evidence,
    FaultCategory,
    InvestigationResult,
    RemediationOutcome,
    RemediationProposal,
    RunStats,
)
from k8s_aiops_agent.telemetry import current_trace_id, mark_error, observe, set_output
from k8s_aiops_agent.tools import (
    SUBMIT_DIAGNOSIS,
    ToolRecord,
    anthropic_tool_definitions,
    run_tool,
    tool_result_text,
)

MAX_REPEATED_CALLS = 3
MAX_REPAIRS = 1
VERIFY_ATTEMPTS = 10
VERIFY_INTERVAL_SECONDS = 3.0


class AgentState(TypedDict, total=False):
    question: str
    namespace: str
    messages: Annotated[list[dict[str, Any]], add]
    ledger: Annotated[list[dict[str, Any]], add]
    started_at: float
    steps: int
    input_tokens: int
    output_tokens: int
    repeated_calls: int
    repairs: int
    last_turn: dict[str, Any]
    finalizing: str | None
    stop_reason: str | None
    diagnosis: dict[str, Any] | None
    proposal: dict[str, Any] | None
    remediation: dict[str, Any] | None


@dataclass(frozen=True)
class Remediation:
    """Enables the approval-gated fix path. The approver decides; the agent never self-approves."""

    patcher: SelectorPatcher
    approve: Callable[[RemediationProposal], bool]


def build_graph(
    settings: AgentSettings, kube: KubeGateway, model: ChatModel, remediation: Remediation | None
) -> CompiledStateGraph[Any, Any, Any, Any]:
    tools = anthropic_tool_definitions()

    def agent(state: AgentState) -> dict[str, Any]:
        with observe(
            "model.call",
            "generation",
            input_value=state["messages"][-1],
            **{"gen_ai.request.model": model.name, "gen_ai.operation.name": "chat"},
        ) as span:
            try:
                turn = model.create(SYSTEM_PROMPT, state["messages"], tools)
            except ModelError as error:
                mark_error(span, str(error))
                return {"stop_reason": "model_error"}
            span.set_attribute("gen_ai.response.model", turn.model)
            span.set_attribute("gen_ai.usage.input_tokens", turn.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", turn.output_tokens)
            span.set_attribute("gen_ai.response.finish_reasons", [turn.stop_reason])
            set_output(span, turn.content)
        update: dict[str, Any] = {
            "messages": [{"role": "assistant", "content": turn.content}],
            "last_turn": {"content": turn.content, "stop_reason": turn.stop_reason},
            "steps": state.get("steps", 0) + 1,
            "input_tokens": state.get("input_tokens", 0) + turn.input_tokens,
            "output_tokens": state.get("output_tokens", 0) + turn.output_tokens,
        }
        if turn.stop_reason == "refusal":
            update["stop_reason"] = "model_refusal"
        return update

    def route_after_agent(
        state: AgentState,
    ) -> Literal["tools", "validate", "finalize", "nudge", "fallback"]:
        if state.get("stop_reason") in ("model_error", "model_refusal"):
            return "fallback"
        tool_uses = _tool_uses(state)
        if any(t["name"] == SUBMIT_DIAGNOSIS for t in tool_uses):
            return "validate"
        if state.get("finalizing"):
            return "fallback"
        if _budget_exceeded(state, settings):
            return "finalize"
        return "tools" if tool_uses else "nudge"

    def tools_node(state: AgentState) -> dict[str, Any]:
        seen = {_call_key(r["name"], r["args"]): r["id"] for r in state.get("ledger", [])}
        results: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        repeated = state.get("repeated_calls", 0)
        for call in _tool_uses(state):
            key = _call_key(call["name"], call["input"])
            if key in seen:
                repeated += 1
                text = f"Duplicate call: this was already answered as {seen[key]}; reuse it."
                results.append(_tool_result(call["id"], text, is_error=True))
                continue
            record_id = f"t{len(state.get('ledger', [])) + len(records) + 1}"
            with observe(f"tool.{call['name']}", "tool", input_value=call["input"]) as span:
                record = run_tool(
                    kube,
                    record_id,
                    call["name"],
                    call["input"],
                    settings.allowed_namespaces,
                    settings.max_tool_output_chars,
                )
                span.set_attribute("evidence_id", record.id)
                span.set_attribute("tool.latency_ms", record.latency_ms)
                set_output(span, record.error or record.output)
                if record.error:
                    mark_error(span, record.error)
            seen[key] = record.id
            records.append(record.to_dict())
            results.append(_tool_result(call["id"], tool_result_text(record), bool(record.error)))
        return {
            "messages": [{"role": "user", "content": results}],
            "ledger": records,
            "repeated_calls": repeated,
        }

    def nudge(state: AgentState) -> dict[str, Any]:
        return {"messages": [{"role": "user", "content": NO_TOOL_CALL_PROMPT}]}

    def finalize(state: AgentState) -> dict[str, Any]:
        reason = _budget_exceeded(state, settings) or "max_steps"
        skipped = [
            _tool_result(t["id"], "Not executed: investigation budget used up.", is_error=True)
            for t in _tool_uses(state)
        ]
        prompt = {"type": "text", "text": FINALIZE_PROMPT.format(reason=reason)}
        return {"messages": [{"role": "user", "content": [*skipped, prompt]}], "finalizing": reason}

    def validate(state: AgentState) -> dict[str, Any]:
        calls = _tool_uses(state)
        submit = next(t for t in calls if t["name"] == SUBMIT_DIAGNOSIS)
        with observe("validate_diagnosis", "evaluator", input_value=submit["input"]) as span:
            diagnosis, problems = check_diagnosis(submit["input"], state.get("ledger", []))
            set_output(span, problems or "valid")
        if diagnosis is not None:
            return {
                "diagnosis": diagnosis.model_dump(mode="json"),
                "stop_reason": state.get("finalizing") or "diagnosis_submitted",
            }
        feedback = "Diagnosis rejected, fix it and call submit_diagnosis again:\n- " + "\n- ".join(
            problems
        )
        results = [
            _tool_result(t["id"], feedback, is_error=True)
            if t is submit
            else _tool_result(t["id"], "Not executed: a diagnosis was submitted.", is_error=True)
            for t in calls
        ]
        return {
            "messages": [{"role": "user", "content": results}],
            "repairs": state.get("repairs", 0) + 1,
        }

    def route_after_validate(
        state: AgentState,
    ) -> Literal["agent", "fallback", "propose", "__end__"]:
        diagnosis = state.get("diagnosis")
        if diagnosis is None:
            if state.get("finalizing") or state.get("repairs", 0) > MAX_REPAIRS:
                return "fallback"
            return "agent"
        is_selector_fault = diagnosis["category"] == FaultCategory.SERVICE_SELECTOR_MISMATCH
        if remediation and diagnosis.get("proposed_fix") and is_selector_fault:
            return "propose"
        return "__end__"

    def fallback(state: AgentState) -> dict[str, Any]:
        reason = state.get("stop_reason") or state.get("finalizing") or "invalid_diagnosis"
        return {
            "diagnosis": fallback_diagnosis(reason, state.get("ledger", [])).model_dump(
                mode="json"
            ),
            "stop_reason": reason,
        }

    def propose(state: AgentState) -> dict[str, Any]:
        diagnosis = Diagnosis.model_validate(state["diagnosis"])
        assert diagnosis.proposed_fix is not None
        with observe("remediation.propose", "span", input_value=diagnosis.proposed_fix) as span:
            proposal = build_proposal(kube, state["namespace"], diagnosis.proposed_fix)
            set_output(span, proposal if isinstance(proposal, str) else proposal.model_dump())
        if isinstance(proposal, str):
            outcome = RemediationOutcome(
                approved=False,
                applied=False,
                resolved=False,
                message=f"No fix proposed: {proposal}",
            )
            return {"remediation": outcome.model_dump(mode="json")}
        return {"proposal": proposal.model_dump(mode="json")}

    def route_after_propose(state: AgentState) -> Literal["approve", "__end__"]:
        return "approve" if state.get("proposal") else "__end__"

    def approve(state: AgentState) -> Command[Literal["apply", "__end__"]]:
        # Pauses the graph; the caller shows the proposal to a human and resumes with their answer.
        approved = bool(interrupt(state["proposal"]))
        with observe("remediation.approval", "span", input_value=state["proposal"]) as span:
            set_output(span, {"approved": approved})
        if approved:
            return Command(goto="apply")
        outcome = RemediationOutcome(
            approved=False, applied=False, resolved=False, message="Fix declined by the operator."
        )
        return Command(goto="__end__", update={"remediation": outcome.model_dump(mode="json")})

    def apply(state: AgentState) -> dict[str, Any]:
        assert remediation is not None
        proposal = RemediationProposal.model_validate(state["proposal"])
        with observe("remediation.apply", "tool", input_value=proposal.model_dump()) as span:
            try:
                remediation.patcher.patch_service_selector(
                    proposal.namespace, proposal.service, proposal.proposed_selector
                )
            except KubeError as error:
                mark_error(span, str(error))
                outcome = RemediationOutcome(
                    approved=True, applied=False, resolved=False, message=f"Patch failed: {error}"
                )
                return {"remediation": outcome.model_dump(mode="json")}
        return {}

    def verify(state: AgentState) -> dict[str, Any]:
        if state.get("remediation"):
            return {}
        proposal = RemediationProposal.model_validate(state["proposal"])
        args = {"namespace": proposal.namespace, "name": proposal.service}
        ledger_size = len(state.get("ledger", []))
        records: list[ToolRecord] = []
        with observe("remediation.verify", "span", input_value=args) as span:
            for attempt in range(VERIFY_ATTEMPTS):
                record = run_tool(
                    kube,
                    f"v{ledger_size + attempt + 1}",
                    "get_service_and_endpoints",
                    args,
                    settings.allowed_namespaces,
                    settings.max_tool_output_chars,
                )
                records.append(record)
                if not record.error and endpoints_cover(
                    json.loads(record.output), proposal.matching_ready_pods
                ):
                    break
                time.sleep(VERIFY_INTERVAL_SECONDS)
            last = records[-1]
            resolved = not last.error and endpoints_cover(
                json.loads(last.output), proposal.matching_ready_pods
            )
            set_output(span, {"resolved": resolved, "attempts": len(records)})
        outcome = RemediationOutcome(
            approved=True,
            applied=True,
            resolved=resolved,
            verification=[
                Evidence(
                    claim="Fresh read of the Service and its endpoints after the patch.",
                    tool_call_id=last.id,
                    excerpt=(last.error or last.output)[:300],
                )
            ],
            message=(
                f"Resolved: {proposal.service} now has ready endpoints for "
                f"{', '.join(proposal.matching_ready_pods)}."
                if resolved
                else f"Not resolved after {len(records)} checks; endpoints did not converge."
            ),
        )
        return {
            "ledger": [r.to_dict() for r in records],
            "remediation": outcome.model_dump(mode="json"),
        }

    graph = StateGraph(AgentState)
    for name, node in [
        ("agent", agent),
        ("tools", tools_node),
        ("nudge", nudge),
        ("finalize", finalize),
        ("validate", validate),
        ("fallback", fallback),
        ("propose", propose),
        ("approve", approve),
        ("apply", apply),
        ("verify", verify),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent)
    graph.add_edge("tools", "agent")
    graph.add_edge("nudge", "agent")
    graph.add_edge("finalize", "agent")
    graph.add_conditional_edges("validate", route_after_validate)
    graph.add_edge("fallback", END)
    graph.add_conditional_edges("propose", route_after_propose)
    graph.add_edge("apply", "verify")
    graph.add_edge("verify", END)
    return graph.compile(checkpointer=InMemorySaver())


def investigate(
    question: str,
    namespace: str,
    settings: AgentSettings,
    kube: KubeGateway,
    model: ChatModel,
    remediation: Remediation | None = None,
) -> InvestigationResult:
    graph = build_graph(settings, kube, model, remediation)
    config: Any = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        # A hard backstop below the agent's own budgets; each step is a few graph super-steps.
        "recursion_limit": settings.max_steps * 4 + 20,
    }
    started = time.monotonic()
    request = {"question": question, "namespace": namespace}
    with observe("investigate", "agent", input_value=request, namespace=namespace) as span:
        trace_id = current_trace_id()
        state: dict[str, Any] = graph.invoke(
            {
                **request,
                "messages": [{"role": "user", "content": _first_message(question, namespace)}],
                "started_at": time.time(),
            },
            config,
        )
        while state.get("__interrupt__"):
            assert remediation is not None
            proposal = RemediationProposal.model_validate(state["__interrupt__"][0].value)
            state = graph.invoke(Command(resume=remediation.approve(proposal)), config)
        result = InvestigationResult(
            question=question,
            diagnosis=Diagnosis.model_validate(state["diagnosis"]),
            stats=RunStats(
                steps=state.get("steps", 0),
                tool_calls=len([r for r in state.get("ledger", []) if r["id"].startswith("t")]),
                input_tokens=state.get("input_tokens", 0),
                output_tokens=state.get("output_tokens", 0),
                latency_ms=int((time.monotonic() - started) * 1000),
                stop_reason=state.get("stop_reason") or "diagnosis_submitted",
                trace_id=trace_id,
            ),
            remediation=RemediationOutcome.model_validate(state["remediation"])
            if state.get("remediation")
            else None,
            ledger=state.get("ledger", []),
        )
        set_output(span, result.diagnosis.model_dump(mode="json"))
        span.set_attribute("stop_reason", result.stats.stop_reason)
    return result


def check_diagnosis(
    raw: dict[str, Any], ledger: list[dict[str, Any]]
) -> tuple[Diagnosis | None, list[str]]:
    """Schema-valid, and every evidence item quotes the tool output it cites."""
    try:
        diagnosis = Diagnosis.model_validate(raw)
    except ValidationError as error:
        return None, [
            f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors(include_url=False)
        ]
    records = {r["id"]: r for r in ledger}
    problems = []
    for item in diagnosis.evidence:
        record = records.get(item.tool_call_id)
        if record is None:
            problems.append(f"evidence cites unknown id '{item.tool_call_id}'")
        elif _normalize(item.excerpt) not in _normalize(record["error"] or record["output"]):
            problems.append(f"excerpt for {item.tool_call_id} is not verbatim from that output")
    return (None, problems) if problems else (diagnosis, [])


def fallback_diagnosis(reason: str, ledger: list[dict[str, Any]]) -> Diagnosis:
    last = ledger[-1] if ledger else None
    evidence = Evidence(
        claim="Last observation before the investigation stopped.",
        tool_call_id=last["id"] if last else "none",
        excerpt=((last["error"] or last["output"])[:200] if last else ""),
    )
    return Diagnosis(
        fault_found=True,
        category=FaultCategory.OTHER,
        root_cause=f"Inconclusive: the investigation stopped ({reason}) before a supported "
        "conclusion was reached.",
        evidence=[evidence],
        recommended_action="Escalate to an operator and inspect the affected namespace manually.",
        action_risk="none",
        confidence=0.0,
    )


def _first_message(question: str, namespace: str) -> str:
    return f"Namespace: {namespace}\nQuestion: {question}"


def _tool_uses(state: AgentState) -> list[dict[str, Any]]:
    content = state.get("last_turn", {}).get("content", [])
    return [b for b in content if b.get("type") == "tool_use"]


def _tool_result(tool_use_id: str, text: str, is_error: bool = False) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": text,
        "is_error": is_error,
    }


def _call_key(name: str, args: dict[str, Any]) -> str:
    return f"{name}:{json.dumps(args, sort_keys=True)}"


def _budget_exceeded(state: AgentState, settings: AgentSettings) -> str | None:
    if state.get("steps", 0) >= settings.max_steps:
        return "max_steps"
    if state.get("input_tokens", 0) + state.get("output_tokens", 0) >= settings.max_total_tokens:
        return "token_budget"
    if time.time() - state.get("started_at", time.time()) >= settings.timeout_seconds:
        return "timeout"
    if state.get("repeated_calls", 0) >= MAX_REPEATED_CALLS:
        return "repeated_calls"
    return None


_ESCAPES_AND_SPACE = re.compile(r'[\\"\s]+')


def _normalize(text: str) -> str:
    # Compare text, not JSON escaping or whitespace: quotes copied from JSON output still match.
    return _ESCAPES_AND_SPACE.sub(" ", text).strip().lower()
