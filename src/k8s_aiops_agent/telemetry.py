"""OpenTelemetry tracing: always to local JSONL, and to Langfuse over OTLP when keys are set."""

import base64
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Span, Status, StatusCode

from k8s_aiops_agent.config import AgentSettings

TRACER_NAME = "k8s_aiops_agent"
MAX_ATTRIBUTE_CHARS = 8_000

_provider: TracerProvider | None = None


class JsonlSpanExporter(SpanExporter):
    """One JSON object per finished span, in a file per trace: traces/<trace_id>.jsonl."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self._directory.mkdir(parents=True, exist_ok=True)
        for span in spans:
            context = span.get_span_context()
            if context is None:
                continue
            trace_id = format(context.trace_id, "032x")
            record = {
                "trace_id": trace_id,
                "span_id": format(context.span_id, "016x"),
                "parent_id": format(span.parent.span_id, "016x") if span.parent else None,
                "name": span.name,
                "start_ns": span.start_time,
                "end_ns": span.end_time,
                "status": span.status.status_code.name,
                "attributes": dict(span.attributes or {}),
            }
            with (self._directory / f"{trace_id}.jsonl").open("a") as out:
                out.write(json.dumps(record, default=str) + "\n")
        return SpanExportResult.SUCCESS


def setup_tracing(settings: AgentSettings) -> TracerProvider:
    global _provider
    if _provider is not None:
        return _provider
    provider = TracerProvider(resource=Resource.create({"service.name": "k8s-aiops-agent"}))
    provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(settings.trace_dir)))
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        token = f"{settings.langfuse_public_key}:{settings.langfuse_secret_key.get_secret_value()}"
        exporter = OTLPSpanExporter(
            endpoint=f"{settings.langfuse_host.rstrip('/')}/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {base64.b64encode(token.encode()).decode()}"},
            timeout=5,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _provider = provider
    return provider


def shutdown_tracing() -> None:
    if _provider is not None:
        _provider.force_flush(timeout_millis=5_000)


@contextmanager
def observe(name: str, kind: str, input_value: Any = None, **attributes: Any) -> Iterator[Span]:
    """A span Langfuse renders as an observation of `kind` (agent, generation, tool, span...)."""
    tracer = (_provider or trace.get_tracer_provider()).get_tracer(TRACER_NAME)
    with tracer.start_as_current_span(name, record_exception=False) as span:
        span.set_attribute("langfuse.observation.type", kind)
        if input_value is not None:
            set_json(span, "langfuse.observation.input", input_value)
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as error:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.set_attribute("error.type", type(error).__name__)
            raise


def set_output(span: Span, value: Any) -> None:
    set_json(span, "langfuse.observation.output", value)


def set_json(span: Span, key: str, value: Any) -> None:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    span.set_attribute(key, text[:MAX_ATTRIBUTE_CHARS])


def mark_error(span: Span, message: str) -> None:
    span.set_status(Status(StatusCode.ERROR, message))
    span.set_attribute("langfuse.observation.level", "ERROR")
    span.set_attribute("langfuse.observation.status_message", message)


def current_trace_id() -> str | None:
    context = trace.get_current_span().get_span_context()
    return format(context.trace_id, "032x") if context.is_valid else None
