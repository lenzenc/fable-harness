"""CLI entrypoint: run a single turn through the harness end-to-end.

    uv run harness "prep me for my meeting with client X"
    uv run harness --test-model "smoke test — stubs the LLM only"

Wires all the Phase 1-3 seams together (tiered router, tool registry +
permissions, pgvector memory + session persistence, budgets, telemetry).
Phase 4 registers the "orchestrator" flow (routes.yaml's advisor_prep route
already points at it, so it'll resolve as soon as it's registered here).

Memory/session persistence degrade gracefully: if Postgres isn't reachable
(e.g. `docker compose up` hasn't been run), the CLI falls back to
NullMemoryStore/NullSessionStore with a warning rather than failing the run
— a non-essential dependency being down shouldn't take out the harness.

Note: `--test-model` only stubs the LLM (via Pydantic AI's `TestModel`) — it
does NOT stub embeddings. The router and memory store always call OpenAI's
embeddings API (harness/routing/embeddings.py), so `HARNESS_OPENAI_API_KEY`
is required for every run, `--test-model` included.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import yaml

from harness.config.settings import get_harness_config, get_settings
from harness.core.budgets import BudgetTracker
from harness.core.context import AblationFlags, RunContext
from harness.core.loop import AgentLoop
from harness.core.permissions import ApprovalGate
from harness.core.session import NullSessionStore, PostgresSessionStore, RunRecord
from harness.flows.base import HarnessServices
from harness.flows.orchestrator import OrchestratorFlow
from harness.flows.single_agent import SingleAgentFlow
from harness.memory.extraction import FactExtractor
from harness.memory.pgvector_store import PgVectorMemoryStore
from harness.memory.store import MemoryStore, NullMemoryStore
from harness.routing.embeddings import get_encoder
from harness.routing.router import TieredRouter
from harness.routing.triage import LLMTriage
from harness.telemetry.otel import configure_telemetry
from harness.tools.demo_tools import demo_tool_specs
from harness.tools.registry import InMemoryToolRegistry

logger = logging.getLogger("harness.cli")

ROUTES_YAML_PATH = Path(__file__).parent / "routing" / "routes.yaml"


def _build_registry() -> InMemoryToolRegistry:
    registry = InMemoryToolRegistry()
    for spec in demo_tool_specs():
        registry.register(spec)
    return registry


def _build_router(settings, config, test_model) -> TieredRouter:
    routes_config = yaml.safe_load(ROUTES_YAML_PATH.read_text())["routes"]
    triage = LLMTriage(model_id=settings.triage_model_id, tier=2, test_model=test_model)
    return TieredRouter(
        encoder=get_encoder(settings.embedding_model_id),
        routes_config=routes_config,
        triage=triage,
        confidence_threshold=config.routing_confidence_threshold,
    )


async def _connect_memory(settings, encoder_id: str) -> tuple[MemoryStore, PgVectorMemoryStore | None]:
    try:
        store = await PgVectorMemoryStore.connect(settings.database_url, encoder_id=encoder_id)
        return store, store
    except Exception as exc:  # noqa: BLE001 - a down/misconfigured DB should degrade, not crash the CLI
        logger.warning("pgvector memory store unreachable (%s) — falling back to NullMemoryStore", exc)
        return NullMemoryStore(), None


async def _connect_sessions(settings):
    try:
        return await PostgresSessionStore.connect(settings.database_url)
    except Exception as exc:  # noqa: BLE001 - a down/misconfigured DB should degrade, not crash the CLI
        logger.warning("session store unreachable (%s) — falling back to NullSessionStore", exc)
        return NullSessionStore()


async def _run(request: str, use_test_model: bool, session_id: str) -> None:
    settings = get_settings()
    config = get_harness_config()
    configure_telemetry(settings.otel_exporter_otlp_endpoint, include_content=settings.otel_include_content)

    test_model = None
    if use_test_model:
        from pydantic_ai.models.test import TestModel

        test_model = TestModel()

    memory, pgvector_store = await _connect_memory(settings, settings.embedding_model_id)
    sessions = await _connect_sessions(settings)
    ctx = RunContext(
        request=request,
        session_id=session_id,
        ablations=AblationFlags(
            memory_on=settings.memory_on,
            critic_on=settings.critic_on,
            router_tier=settings.router_tier,
        ),
    )
    await sessions.ensure_session(session_id, ctx.user_scope, ctx.org_scope)

    fact_extractor = FactExtractor(model_id=settings.triage_model_id, test_model=test_model)
    flows = {
        "single_agent": SingleAgentFlow(model_id=settings.model_id, test_model=test_model, fact_extractor=fact_extractor),
        "orchestrator": OrchestratorFlow(model_id=settings.model_id, test_model=test_model, fact_extractor=fact_extractor),
    }
    loop = AgentLoop(flows=flows)

    # Demo-only approval callback: auto-approves anything the permission
    # matrix marks require_approval (e.g. the orchestrator's `financial`-tier
    # account_lookup), so the CLI demo runs end-to-end out of the box while
    # still exercising the gate. A real deployment replaces this with a human
    # approval mechanism (Slack prompt, review queue, etc) — the library
    # default (ApprovalGate() with no callback) fails closed, not open.
    services = HarnessServices(
        tools=_build_registry(),
        memory=memory,
        budget=BudgetTracker(config.budget_for("default"), model_id=settings.model_id, pricing=config.pricing),
        router=_build_router(settings, config, test_model),
        approval_gate=ApprovalGate(matrix=config.permission_matrix(), on_approval_needed=lambda call, tier: True),
    )

    try:
        result = await loop.run_turn(ctx, services)
    finally:
        if pgvector_store is not None:
            await pgvector_store.close()

    await sessions.record_run(
        RunRecord(
            session_id=session_id,
            request=request,
            output=str(result.output) if result.output is not None else None,
            stop_reason=result.stop_reason,
            success=result.success,
        )
    )
    if isinstance(sessions, PostgresSessionStore):
        await sessions.close()

    print(f"[{result.stop_reason}] success={result.success}")
    print(result.output)


def run() -> None:
    parser = argparse.ArgumentParser(prog="harness", description="Run one turn through the fable-harness.")
    parser.add_argument("request", help="The user request/query to run through the harness.")
    parser.add_argument(
        "--test-model",
        action="store_true",
        help=(
            "Use Pydantic AI's TestModel instead of a real Anthropic call — no Anthropic API "
            "key needed and zero LLM token spend, but routing/memory still call OpenAI's "
            "embeddings API, so HARNESS_OPENAI_API_KEY is still required."
        ),
    )
    parser.add_argument("--session-id", default="cli-session", help="Session id for memory/session scoping.")
    args = parser.parse_args()
    asyncio.run(_run(args.request, args.test_model, args.session_id))


if __name__ == "__main__":
    run()
