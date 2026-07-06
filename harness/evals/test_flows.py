"""Flow-level evals: drive the single-agent tool loop and the
orchestrator-workers advisor-prep flow end-to-end with Pydantic AI's
`TestModel` — deterministic, zero token spend, no API key required. This is
what makes the eval suite runnable in CI as a release gate (DeepEval/Ragas
would layer real judge metrics on top of real model output; these tests
check the harness's own plumbing behaves correctly regardless of model
quality).
"""

from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel

from harness.core.budgets import Budget, BudgetTracker
from harness.core.context import AblationFlags, RunContext
from harness.core.permissions import ApprovalGate
from harness.flows.base import HarnessServices
from harness.flows.orchestrator import OrchestratorFlow
from harness.flows.single_agent import SingleAgentFlow
from harness.memory.store import NullMemoryStore
from harness.tools.demo_tools import demo_tool_specs
from harness.tools.registry import InMemoryToolRegistry


def _registry() -> InMemoryToolRegistry:
    registry = InMemoryToolRegistry()
    for spec in demo_tool_specs():
        registry.register(spec)
    return registry


def _services(approve_all: bool = True) -> HarnessServices:
    gate = ApprovalGate(on_approval_needed=(lambda call, tier: True) if approve_all else None)
    return HarnessServices(
        tools=_registry(),
        memory=NullMemoryStore(),
        budget=BudgetTracker(Budget(max_total_tokens=20_000, max_tool_calls=10), model_id="test"),
        approval_gate=gate,
    )


@pytest.mark.asyncio
async def test_single_agent_flow_succeeds_with_test_model() -> None:
    flow = SingleAgentFlow(model_id="test", test_model=TestModel())
    ctx = RunContext(request="what time is it?", session_id="eval-single-agent")
    result = await flow.run(ctx.request, ctx, _services())

    assert result.success
    assert result.stop_reason == "success"
    assert result.transcript, "expected a non-empty message transcript"


@pytest.mark.asyncio
async def test_orchestrator_flow_fans_out_and_synthesizes() -> None:
    flow = OrchestratorFlow(model_id="test", test_model=TestModel())
    ctx = RunContext(request="prep me for my meeting with client X", session_id="eval-orchestrator")
    result = await flow.run(ctx.request, ctx, _services())

    assert result.success
    assert result.stop_reason == "success"
    assert isinstance(result.output, dict)
    assert "summary" in result.output and "citations" in result.output


@pytest.mark.asyncio
async def test_orchestrator_degrades_gracefully_when_a_worker_is_denied() -> None:
    """The account_lookup demo tool is tagged `financial` risk tier — with no
    approval callback configured, ApprovalGate fails closed (see
    core/permissions.py). The orchestrator's worker wrapper converts that
    into a "[unavailable: ...]" string rather than aborting the whole flow
    (see flows/orchestrator.py:_worker) — one denied worker shouldn't take
    down account/meeting/retrieval fan-out entirely. Permission enforcement
    itself (the denial actually firing) is covered directly in
    test_harness.py, since TestModel's synthesis output doesn't meaningfully
    reflect input content, only structure."""
    flow = OrchestratorFlow(model_id="test", test_model=TestModel())
    ctx = RunContext(request="prep me for my meeting with client X", session_id="eval-orchestrator-denied")
    result = await flow.run(ctx.request, ctx, _services(approve_all=False))

    assert result.success
    assert result.stop_reason == "success"


@pytest.mark.asyncio
async def test_single_agent_flow_respects_ablated_memory() -> None:
    services = _services()
    flow = SingleAgentFlow(model_id="test", test_model=TestModel())
    ctx = RunContext(request="what time is it?", session_id="eval-memory-off", ablations=AblationFlags(memory_on=False))

    recalled_calls = []
    original_recall = services.memory.recall

    async def spy_recall(*args, **kwargs):
        recalled_calls.append((args, kwargs))
        return await original_recall(*args, **kwargs)

    services.memory.recall = spy_recall  # type: ignore[method-assign]
    await flow.run(ctx.request, ctx, services)

    assert not recalled_calls, "memory.recall should not be called when memory_on=False"
