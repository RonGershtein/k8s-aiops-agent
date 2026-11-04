import json
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from k8s_aiops_agent import telemetry
from k8s_aiops_agent.agent import investigate
from k8s_aiops_agent.fixture_kube import FixtureKube
from k8s_aiops_agent.telemetry import JsonlSpanExporter
from tests.builders import make_settings, scripted_model


@pytest.fixture
def trace_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private tracer provider writing JSONL to tmp_path, independent of the global one."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(tmp_path)))
    monkeypatch.setattr(telemetry, "_provider", provider)
    return tmp_path


def summarize_trace(trace_file: Path) -> dict[str, Any]:
    """What the request span's direct children show: observation types, evidence ids, tokens."""
    spans = [json.loads(line) for line in trace_file.open()]
    root = next(s for s in spans if s["name"] == "investigate")
    children = [s for s in spans if s["parent_id"] == root["span_id"]]
    return {
        "types": {s["attributes"]["langfuse.observation.type"] for s in children},
        "evidence_ids": [
            s["attributes"]["evidence_id"] for s in children if "evidence_id" in s["attributes"]
        ],
        "model_input_tokens": [
            s["attributes"]["gen_ai.usage.input_tokens"]
            for s in children
            if s["name"] == "model.call"
        ],
    }


def test_investigation_trace_links_request_model_calls_and_tools(
    trace_dir: Path, kube: FixtureKube
) -> None:
    model = scripted_model("01-crashloop-bad-config")

    result = investigate("q", "scenario-01", make_settings(), kube, model)
    trace = summarize_trace(trace_dir / f"{result.stats.trace_id}.jsonl")

    assert trace["types"] >= {"generation", "tool", "evaluator"}
    assert trace["evidence_ids"] == ["t1", "t2", "t3", "t4", "t5"]
    assert min(trace["model_input_tokens"]) > 0
