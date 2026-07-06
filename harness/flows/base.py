"""Shared Flow contract + the service bundle every flow receives.

A `Flow` is anything the router can dispatch a request to beyond a single
direct answer: the default single-agent tool loop (`flows/single_agent.py`)
or a deterministic/agentic workflow like orchestrator-workers
(`flows/orchestrator.py`). Flows are handed a `HarnessServices` bundle rather
than reaching into globals, which is what makes them independently testable
and ablatable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from harness.core.budgets import BudgetTracker
    from harness.core.context import RunContext
    from harness.core.permissions import ApprovalGate
    from harness.memory.store import MemoryStore
    from harness.routing.router import Router
    from harness.tools.registry import ToolRegistry


@dataclass
class FlowResult:
    output: Any
    success: bool
    stop_reason: str  # e.g. "success", "budget_exhausted", "repeated_failure", "error"
    transcript: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class HarnessServices:
    """Everything a Flow needs, threaded explicitly rather than via globals."""

    tools: "ToolRegistry"
    memory: "MemoryStore"
    budget: "BudgetTracker"
    router: "Router | None" = None
    approval_gate: "ApprovalGate | None" = None


@runtime_checkable
class Flow(Protocol):
    async def run(self, request: str, ctx: "RunContext", services: HarnessServices) -> FlowResult: ...
