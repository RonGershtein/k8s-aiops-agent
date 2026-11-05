import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.tree import Tree

from k8s_aiops_agent.agent import Remediation, investigate
from k8s_aiops_agent.config import AgentSettings, KubeBackend, LlmProvider
from k8s_aiops_agent.evals import (
    ClaudeJudge,
    GeminiJudge,
    TrialResult,
    enough_graded,
    load_scenarios,
    pass_rate,
    render_report,
    run_evals,
)
from k8s_aiops_agent.fixture_kube import FixtureKube
from k8s_aiops_agent.guards import ToolForbiddenError, ensure_namespace_allowed
from k8s_aiops_agent.kube import KubeGateway, LiveKube
from k8s_aiops_agent.llm import build_model
from k8s_aiops_agent.mcp_server import build_mcp_server
from k8s_aiops_agent.remediation import LiveSelectorPatcher
from k8s_aiops_agent.schemas import (
    Diagnosis,
    FaultCategory,
    InvestigationResult,
    RemediationProposal,
)
from k8s_aiops_agent.telemetry import current_trace_id, setup_tracing, shutdown_tracing

SCRIPTED_DIR = Path("evals/scripted")

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console(stderr=True)


@app.command("investigate")
def investigate_command(
    question: Annotated[str, typer.Argument(help="Free-text troubleshooting request.")],
    namespace: Annotated[str, typer.Option("--namespace", "-n")],
    backend: Annotated[KubeBackend | None, typer.Option(help="Overrides AGENT_BACKEND.")] = None,
    llm: Annotated[LlmProvider | None, typer.Option(help="Overrides AGENT_LLM_PROVIDER.")] = None,
    script: Annotated[Path | None, typer.Option(help="Turn script for --llm scripted.")] = None,
    allow_remediation: Annotated[
        bool, typer.Option(help="Offer the approval-gated Service selector fix.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
) -> None:
    """Investigate a namespace and print a root cause, evidence and a recommended action."""
    settings = _load_settings(backend, llm)
    try:  # before any client, model or trace is set up
        ensure_namespace_allowed(namespace, settings.allowed_namespaces)
    except ToolForbiddenError as error:
        raise _usage_error(str(error)) from error
    kube = _build_kube(settings)
    try:
        model = build_model(settings, script)
    except ValueError as error:
        raise _usage_error(str(error)) from error
    approval = ApprovalPrompt(settings.trace_dir)
    remediation = _build_remediation(settings, approval) if allow_remediation else None
    setup_tracing(settings)
    try:
        result = investigate(question, namespace, settings, kube, model, remediation)
    finally:
        shutdown_tracing()
    if as_json:
        print(result.model_dump_json(indent=2))
    elif approval.shown_findings:
        _print_outcome(result)  # the investigation and diagnosis were shown before the prompt
    else:
        _print_result(result, settings.trace_dir)


@app.command("eval")
def eval_command(
    trials: Annotated[int, typer.Option(min=1, help="Runs per scenario.")] = 1,
    judge: Annotated[
        bool,
        typer.Option(
            help="Also score each run with an LLM judge (Gemini if provider, else Claude)."
        ),
    ] = False,
    held_out: Annotated[bool, typer.Option(help="Include held-out scenarios.")] = True,
    scenario: Annotated[list[str] | None, typer.Option(help="Only these scenario ids.")] = None,
    backend: Annotated[KubeBackend | None, typer.Option()] = None,
    llm: Annotated[LlmProvider | None, typer.Option()] = None,
    min_pass_rate: Annotated[float, typer.Option(help="Exit 1 below this pass rate.")] = 0.0,
    out_dir: Annotated[Path, typer.Option()] = Path("evals/results"),
) -> None:
    """Run the scenario evals and write a Markdown + JSON report."""
    settings = _load_settings(backend, llm)
    scenarios = load_scenarios(settings.fixtures_dir, held_out, scenario)
    if not scenarios:
        raise _usage_error("no scenarios selected")
    missing = {s.namespace for s in scenarios} - set(settings.allowed_namespaces)
    if missing:
        raise _usage_error(f"add {sorted(missing)} to AGENT_ALLOWED_NAMESPACES")
    gemini = settings.llm_provider is LlmProvider.GEMINI
    if judge and not gemini and settings.anthropic_api_key is None:
        raise _usage_error("--judge needs ANTHROPIC_API_KEY (or AGENT_LLM_PROVIDER=gemini)")
    setup_tracing(settings)
    kube = _build_kube(settings)
    grader = (GeminiJudge(settings) if gemini else ClaudeJudge(settings)) if judge else None

    def print_result(result: TrialResult) -> None:
        mark = (
            "[yellow]ERR[/]"
            if result.errored
            else "[green]pass[/]"
            if result.passed
            else "[red]FAIL[/]"
        )
        console.print(
            f"{mark} {result.scenario} #{result.trial} {result.category} ({result.stop_reason})"
        )

    try:
        results = run_evals(
            scenarios,
            trials,
            settings,
            kube,
            lambda case: build_model(settings, SCRIPTED_DIR / f"{case.id}.yaml"),
            grader,
            print_result,
        )
    finally:
        shutdown_tracing()
    report = _write_reports(
        results, f"{settings.backend.value}-{settings.llm_provider.value}", out_dir
    )
    rate = pass_rate(results)
    console.print(f"pass rate {rate:.0%} -> {report}")
    if not enough_graded(results):
        console.print("[red]too many trials errored (model API) for the pass rate to count[/]")
        raise typer.Exit(code=1)
    if rate < min_pass_rate:
        raise typer.Exit(code=1)


def _write_reports(results: list[TrialResult], mode: str, out_dir: Path) -> Path:
    """Writes <mode>.md for people and <mode>.json for tools; returns the Markdown path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"{mode}.md"
    report.write_text(render_report(results, mode))
    (out_dir / f"{mode}.json").write_text(
        json.dumps([r.model_dump(mode="json") for r in results], indent=2) + "\n"
    )
    return report


@app.command("mcp")
def mcp_command(
    backend: Annotated[KubeBackend | None, typer.Option(help="Overrides AGENT_BACKEND.")] = None,
) -> None:
    """Serve the read-only investigation tools over MCP (stdio)."""
    settings = _load_settings(backend, None)
    server = build_mcp_server(
        _build_kube(settings), settings.allowed_namespaces, settings.max_tool_output_chars
    )
    server.run("stdio")


@app.command("trace")
def trace_command(
    trace_id: str,
    trace_dir: Annotated[Path | None, typer.Option(help="Overrides AGENT_TRACE_DIR.")] = None,
) -> None:
    """Show a recorded trace as a timeline: request -> agent -> model -> tools -> response."""
    directory = trace_dir or _load_settings(None, None).trace_dir
    console.print(_trace_tree(directory / f"{trace_id}.jsonl"))


def _load_settings(backend: KubeBackend | None, llm: LlmProvider | None) -> AgentSettings:
    overrides: dict[str, Any] = {}
    if backend:
        overrides["backend"] = backend
    if llm:
        overrides["llm_provider"] = llm
    try:
        return AgentSettings(**overrides)
    except ValidationError as error:
        raise _usage_error(f"invalid configuration: {error}") from error


def _build_kube(settings: AgentSettings) -> KubeGateway:
    if settings.backend is KubeBackend.FIXTURE:
        return FixtureKube.from_dir(settings.fixtures_dir)
    if settings.kubeconfig is None:
        raise _usage_error("live backend needs AGENT_KUBECONFIG (run `make cluster-up`)")
    try:
        return LiveKube.from_kubeconfig(settings.kubeconfig)
    except OSError as error:
        raise _usage_error(f"cannot read kubeconfig {settings.kubeconfig}: {error}") from error


def _build_remediation(settings: AgentSettings, approval: "ApprovalPrompt") -> Remediation:
    path = settings.remediator_kubeconfig
    if settings.backend is not KubeBackend.LIVE or path is None:
        raise _usage_error(
            "--allow-remediation needs the live backend and AGENT_REMEDIATOR_KUBECONFIG"
        )
    try:
        patcher = LiveSelectorPatcher.from_kubeconfig(path)
    except OSError as error:
        raise _usage_error(f"cannot read kubeconfig {path}: {error}") from error
    return Remediation(patcher=patcher, approve=approval)


@dataclass
class ApprovalPrompt:
    """Asks a human to approve the fix, after showing what was investigated and why."""

    trace_dir: Path
    shown_findings: bool = False

    def __call__(self, proposal: RemediationProposal, diagnosis: Diagnosis) -> bool:
        trace_id = current_trace_id()
        if trace_id:
            console.print(_trace_tree(self.trace_dir / f"{trace_id}.jsonl"))
        console.print(_diagnosis_panel(diagnosis, "Diagnosis"))
        console.print(_proposal_panel(proposal))
        self.shown_findings = True
        try:
            return typer.confirm("Apply this change?", default=False, err=True)
        except typer.Abort:
            return False


def _proposal_panel(proposal: RemediationProposal) -> Panel:
    return Panel(
        f"[bold]Patch Service {proposal.namespace}/{proposal.service} selector[/]\n"
        + escape(
            f"current:  {proposal.current_selector}\n"
            f"proposed: {proposal.proposed_selector}\n"
            f"matches Ready pods: {', '.join(proposal.matching_ready_pods)}\n\n"
            f"Expected effect: {proposal.expected_effect}"
        ),
        title="Proposed fix (requires approval)",
        border_style="yellow",
    )


def _print_result(result: InvestigationResult, trace_dir: Path) -> None:
    if result.stats.trace_id:
        console.print(_trace_tree(trace_dir / f"{result.stats.trace_id}.jsonl"))
    console.print(_diagnosis_panel(result.diagnosis, f"Diagnosis: {result.question}"))
    _print_outcome(result)


def _diagnosis_panel(diagnosis: Diagnosis, title: str) -> Panel:
    evidence = "\n".join(
        escape(f"  [{e.tool_call_id}] {e.claim}\n      “{e.excerpt}”") for e in diagnosis.evidence
    )
    color = {FaultCategory.NO_FAULT: "green", FaultCategory.INCONCLUSIVE: "yellow"}.get(
        diagnosis.category, "red"
    )
    return Panel(
        f"[bold]{diagnosis.category.value}[/] (confidence {diagnosis.confidence:.0%})\n\n"
        f"[bold]Root cause[/]\n  {escape(diagnosis.root_cause)}\n\n"
        f"[bold]Evidence[/]\n{evidence}\n\n"
        f"[bold]Recommended action[/] (risk: {diagnosis.action_risk})\n  "
        f"{escape(diagnosis.recommended_action)}",
        title=escape(title),
        border_style=color,
    )


def _print_outcome(result: InvestigationResult) -> None:
    if result.remediation:
        remediation = result.remediation
        console.print(
            Panel(
                escape(remediation.message),
                title="Remediation",
                border_style="green" if remediation.resolved else "yellow",
            )
        )
    stats = result.stats
    console.print(
        f"[dim]stop={stats.stop_reason} steps={stats.steps} tool_calls={stats.tool_calls} "
        f"tokens={stats.input_tokens}+{stats.output_tokens} latency={stats.latency_ms}ms "
        f"trace={stats.trace_id}[/]"
    )


def _trace_tree(path: Path) -> Tree | str:
    if not path.exists():
        return f"no trace file at {path}"
    spans = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not spans:
        return f"trace file {path} is empty"
    spans.sort(key=lambda s: s["start_ns"])
    by_parent: dict[str | None, list[dict[str, Any]]] = {}
    for span in spans:
        by_parent.setdefault(span["parent_id"], []).append(span)
    roots = [s for s in spans if s["parent_id"] not in {x["span_id"] for x in spans}]
    tree = Tree(f"[bold]trace {spans[0]['trace_id']}[/]")

    def add_span(node: Tree, span: dict[str, Any]) -> None:
        attrs = span["attributes"]
        ms = (span["end_ns"] - span["start_ns"]) // 1_000_000
        label = f"{span['name']} [dim]{ms}ms[/]"
        if "evidence_id" in attrs:
            label += f" [cyan]{attrs['evidence_id']}[/] " + escape(
                str(attrs.get("langfuse.observation.input", ""))
            )
        if "gen_ai.usage.input_tokens" in attrs:
            label += (
                f" [magenta]{attrs['gen_ai.usage.input_tokens']}+"
                f"{attrs['gen_ai.usage.output_tokens']} tok[/]"
            )
        if span["status"] == "ERROR":
            label += " [red]ERROR[/]"
        child = node.add(label)
        for sub in by_parent.get(span["span_id"], []):
            add_span(child, sub)

    for root in roots:
        add_span(tree, root)
    return tree


def _usage_error(message: str) -> typer.Exit:
    console.print(f"[red]error:[/] {message}")
    return typer.Exit(code=2)
