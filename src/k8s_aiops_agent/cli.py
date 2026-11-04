import json
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
from k8s_aiops_agent.fixture_kube import FixtureKube
from k8s_aiops_agent.kube import KubeGateway, LiveKube
from k8s_aiops_agent.llm import build_model
from k8s_aiops_agent.remediation import LiveSelectorPatcher
from k8s_aiops_agent.schemas import InvestigationResult, RemediationProposal
from k8s_aiops_agent.telemetry import setup_tracing, shutdown_tracing

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
    setup_tracing(settings)
    try:
        result = investigate(
            question,
            namespace,
            settings,
            _build_kube(settings),
            build_model(settings, script),
            _build_remediation(settings) if allow_remediation else None,
        )
    except ValueError as error:
        raise _fail(str(error)) from error
    finally:
        shutdown_tracing()
    if as_json:
        print(result.model_dump_json(indent=2))
    else:
        _render(result, settings.trace_dir)


@app.command("trace")
def trace_command(
    trace_id: str,
    trace_dir: Annotated[Path, typer.Option()] = Path("traces"),
) -> None:
    """Show a recorded trace as a timeline: request -> agent -> model -> tools -> response."""
    console.print(_trace_tree(trace_dir / f"{trace_id}.jsonl"))


def _load_settings(backend: KubeBackend | None, llm: LlmProvider | None) -> AgentSettings:
    overrides: dict[str, Any] = {}
    if backend:
        overrides["backend"] = backend
    if llm:
        overrides["llm_provider"] = llm
    try:
        return AgentSettings(**overrides)
    except ValidationError as error:
        raise _fail(f"invalid configuration: {error}") from error


def _build_kube(settings: AgentSettings) -> KubeGateway:
    if settings.backend is KubeBackend.FIXTURE:
        return FixtureKube.from_dir(settings.fixtures_dir)
    if settings.kubeconfig is None or not settings.kubeconfig.exists():
        raise _fail("live backend needs AGENT_KUBECONFIG (run `make cluster-up`)")
    return LiveKube.from_kubeconfig(settings.kubeconfig)


def _build_remediation(settings: AgentSettings) -> Remediation:
    path = settings.remediator_kubeconfig
    if settings.backend is not KubeBackend.LIVE or path is None or not path.exists():
        raise _fail("--allow-remediation needs the live backend and AGENT_REMEDIATOR_KUBECONFIG")
    return Remediation(patcher=LiveSelectorPatcher.from_kubeconfig(path), approve=_ask_approval)


def _ask_approval(proposal: RemediationProposal) -> bool:
    console.print(
        Panel(
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
    )
    try:
        return typer.confirm("Apply this change?", default=False, err=True)
    except typer.Abort:
        return False


def _render(result: InvestigationResult, trace_dir: Path) -> None:
    d = result.diagnosis
    if result.stats.trace_id:
        console.print(_trace_tree(trace_dir / f"{result.stats.trace_id}.jsonl"))
    evidence = "\n".join(
        escape(f"  [{e.tool_call_id}] {e.claim}\n      “{e.excerpt}”") for e in d.evidence
    )
    color = "green" if not d.fault_found else "red"
    console.print(
        Panel(
            f"[bold]{d.category.value}[/] (confidence {d.confidence:.0%})\n\n"
            f"[bold]Root cause[/]\n  {escape(d.root_cause)}\n\n[bold]Evidence[/]\n{evidence}\n\n"
            f"[bold]Recommended action[/] (risk: {d.action_risk})\n  "
            f"{escape(d.recommended_action)}",
            title=escape(f"Diagnosis: {result.question}"),
            border_style=color,
        )
    )
    if result.remediation:
        r = result.remediation
        console.print(
            Panel(
                escape(r.message),
                title="Remediation",
                border_style="green" if r.resolved else "yellow",
            )
        )
    s = result.stats
    console.print(
        f"[dim]stop={s.stop_reason} steps={s.steps} tool_calls={s.tool_calls} "
        f"tokens={s.input_tokens}+{s.output_tokens} latency={s.latency_ms}ms trace={s.trace_id}[/]"
    )


def _trace_tree(path: Path) -> Tree | str:
    if not path.exists():
        return f"no trace file at {path}"
    spans = [json.loads(line) for line in path.read_text().splitlines()]
    spans.sort(key=lambda s: s["start_ns"])
    by_parent: dict[str | None, list[dict[str, Any]]] = {}
    for span in spans:
        by_parent.setdefault(span["parent_id"], []).append(span)
    roots = [s for s in spans if s["parent_id"] not in {x["span_id"] for x in spans}]
    tree = Tree(f"[bold]trace {spans[0]['trace_id']}[/]")

    def add(node: Tree, span: dict[str, Any]) -> None:
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
            add(child, sub)

    for root in roots:
        add(tree, root)
    return tree


def _fail(message: str) -> typer.Exit:
    console.print(f"[red]error:[/] {message}")
    return typer.Exit(code=2)
