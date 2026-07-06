"""Ablation harness: run the same request through the orchestrator flow
under different harness feature flags and tabulate behavioral deltas — the
concrete answer to "does this feature actually change anything, and by how
much." Every harness feature is supposed to be independently disable-able
per the design thesis (model upgrades reprice harness strategies); this is
where that claim gets checked.

Real quality-metric ablation (e.g. Ragas faithfulness deltas across
memory-on/off, evaluated against real traffic) needs a real judge model and
real production traffic — out of scope for a TestModel-driven MVP runner.
This measures deltas the harness fully controls instead: agent-call count,
memory read/write calls — deferring semantic-quality ablation to the
"memory benchmarking debt" item in the project's known risks.

Run standalone for a report:
    uv run python -m harness.evals.ablations
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pydantic_ai.models.test import TestModel

from harness.core.budgets import Budget, BudgetTracker
from harness.core.context import AblationFlags, RunContext
from harness.core.permissions import ApprovalGate
from harness.flows.base import HarnessServices
from harness.flows.orchestrator import OrchestratorFlow
from harness.memory.extraction import FactExtractor
from harness.memory.store import MemoryItem, NullMemoryStore
from harness.tools.demo_tools import demo_tool_specs
from harness.tools.registry import InMemoryToolRegistry


@dataclass
class AblationResult:
    config: dict
    agent_calls: int
    memory_recall_calls: int
    memory_remember_calls: int


def _registry() -> InMemoryToolRegistry:
    registry = InMemoryToolRegistry()
    for spec in demo_tool_specs():
        registry.register(spec)
    return registry


class _CountingMemoryStore(NullMemoryStore):
    """Wraps NullMemoryStore just to count calls — behaves identically
    otherwise (no-op reads/writes), so this measures *whether* memory was
    touched, not memory quality."""

    def __init__(self) -> None:
        self.recall_calls = 0
        self.remember_calls = 0

    async def recall(self, query: str, scope: str, k: int = 8) -> list[MemoryItem]:
        self.recall_calls += 1
        return await super().recall(query, scope, k)

    async def remember(self, items: list[MemoryItem], scope: str) -> None:
        self.remember_calls += 1
        return await super().remember(items, scope)


async def _run_with_flags(memory_on: bool, critic_on: bool) -> AblationResult:
    memory = _CountingMemoryStore()
    agent_calls = {"n": 0}

    test_model = TestModel()
    fact_extractor = FactExtractor(model_id="test", test_model=test_model)
    flow = OrchestratorFlow(model_id="test", test_model=test_model, fact_extractor=fact_extractor)
    original_run_agent = flow._run_agent

    async def counted_run_agent(agent, prompt):
        agent_calls["n"] += 1
        return await original_run_agent(agent, prompt)

    flow._run_agent = counted_run_agent  # type: ignore[method-assign]

    services = HarnessServices(
        tools=_registry(),
        memory=memory,
        budget=BudgetTracker(Budget(max_total_tokens=50_000, max_tool_calls=20), model_id="test"),
        approval_gate=ApprovalGate(on_approval_needed=lambda call, tier: True),
    )
    ctx = RunContext(
        request="prep me for my meeting with client X",
        session_id=f"ablation-mem{memory_on}-critic{critic_on}",
        ablations=AblationFlags(memory_on=memory_on, critic_on=critic_on),
    )
    await flow.run(ctx.request, ctx, services)

    return AblationResult(
        config={"memory_on": memory_on, "critic_on": critic_on},
        agent_calls=agent_calls["n"],
        memory_recall_calls=memory.recall_calls,
        memory_remember_calls=memory.remember_calls,
    )


async def run_ablations() -> list[AblationResult]:
    """The full 2x2 sweep."""
    configs = [(m, c) for m in (True, False) for c in (True, False)]
    return [await _run_with_flags(memory_on=m, critic_on=c) for m, c in configs]


def test_critic_ablation_adds_exactly_one_agent_call() -> None:
    """critic_on=True must invoke exactly one more agent call (the critic
    pass) than critic_on=False, all else equal — the concrete, harness-owned
    behavioral delta the flag is supposed to produce."""
    with_critic = asyncio.run(_run_with_flags(memory_on=False, critic_on=True))
    without_critic = asyncio.run(_run_with_flags(memory_on=False, critic_on=False))
    assert with_critic.agent_calls == without_critic.agent_calls + 1


def test_memory_ablation_toggles_recall_and_remember() -> None:
    """memory_on=False must skip both memory.recall and memory.remember
    entirely; memory_on=True must call both at least once."""
    with_memory = asyncio.run(_run_with_flags(memory_on=True, critic_on=True))
    without_memory = asyncio.run(_run_with_flags(memory_on=False, critic_on=True))

    assert without_memory.memory_recall_calls == 0
    assert without_memory.memory_remember_calls == 0
    assert with_memory.memory_recall_calls > 0
    assert with_memory.memory_remember_calls > 0


if __name__ == "__main__":
    results = asyncio.run(run_ablations())
    print(f"{'config':40s} agent_calls  memory_recall  memory_remember")
    for r in results:
        print(f"{str(r.config):40s} {r.agent_calls:11d}  {r.memory_recall_calls:13d}  {r.memory_remember_calls:15d}")
