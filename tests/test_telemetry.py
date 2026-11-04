import json
from pathlib import Path

from k8s_aiops_agent.agent import investigate
from k8s_aiops_agent.config import AgentSettings
from k8s_aiops_agent.fixture_kube import FixtureKube
from k8s_aiops_agent.llm import ScriptedModel
from k8s_aiops_agent.telemetry import setup_tracing


def test_investigation_trace_links_request_model_calls_and_tools(tmp_path: Path) -> None:
    settings = AgentSettings(  # type: ignore[call-arg]
        _env_file=None, allowed_namespaces=["scenario-01"], trace_dir=tmp_path
    )
    setup_tracing(settings)
    model = ScriptedModel.from_file(Path("evals/scripted/01-crashloop-bad-config.yaml"))

    result = investigate(
        "q", "scenario-01", settings, FixtureKube.from_dir(Path("scenarios")), model
    )

    spans = [json.loads(line) for line in (tmp_path / f"{result.stats.trace_id}.jsonl").open()]
    root = next(s for s in spans if s["name"] == "investigate")
    children = [s for s in spans if s["parent_id"] == root["span_id"]]
    tool_spans = [s for s in children if s["name"].startswith("tool.")]
    assert {s["attributes"]["langfuse.observation.type"] for s in children} >= {
        "generation",
        "tool",
    }
    assert [s["attributes"]["evidence_id"] for s in tool_spans] == ["t1", "t2", "t3", "t4", "t5"]
    assert all(
        s["attributes"]["gen_ai.usage.input_tokens"] > 0
        for s in children
        if s["name"] == "model.call"
    )
