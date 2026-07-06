"""OTel wiring: native GenAI instrumentation (no Logfire) + the harness.*
custom attribute namespace.

Pydantic AI emits `gen_ai.*` spans itself via `Agent.instrument_all()` once a
TracerProvider is configured — this module's job is just to point that
TracerProvider at the OTel Collector (which fans out to Phoenix + Datadog,
see harness/telemetry/collector/config.yaml), and to pin the GenAI semconv
version so attribute names don't silently shift while the convention is
still "Development" status upstream. Harness-owned attributes (route
decisions, budget state, ablation flags) live in their own `harness.*`
namespace to avoid collisions with future OTel GenAI additions — see
RunContext.stamp() in harness/core/context.py, which is where they're set.
"""

from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic_ai.agent import Agent, InstrumentationSettings

SERVICE_NAME = "fable-harness"
# Pinned InstrumentationSettings schema version. GenAI + MCP semconv are still
# "Development" upstream (expect a migration); bump this deliberately when we
# choose to move, not implicitly on a library upgrade.
SEMCONV_VERSION = 5

_configured = False


def configure_telemetry(otlp_endpoint: str, service_name: str = SERVICE_NAME, include_content: bool = False) -> None:
    """Idempotent — safe to call once at process startup (cli.py, eval
    fixtures). Sets up a TracerProvider exporting to the OTel Collector and
    turns on Pydantic AI's native GenAI instrumentation for every Agent.

    `include_content` controls the first half of the PII double-guard the
    research brief calls for: full prompt/response content
    (`gen_ai.input.messages` etc) is only ever captured on spans when this is
    explicitly True (pre-production/debug environments) — default False, so
    a production deployment doesn't emit conversation content at the source
    at all. The second half — a `redaction` processor stripping/blocking any
    content that slips through anyway — lives in
    harness/telemetry/collector/config.yaml, since content capture being
    disabled here is a harness-level setting, not a guarantee every
    dependency respects.
    """
    global _configured
    if _configured:
        return

    os.environ.setdefault("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental")

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)))
    trace.set_tracer_provider(provider)

    Agent.instrument_all(InstrumentationSettings(version=SEMCONV_VERSION, include_content=include_content))

    _configured = True
