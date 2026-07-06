"""Router: the harness's front door.

The router decides *mode* (direct answer / single-agent tool loop / agentic
flow), not just a destination name. Production traffic should mostly
terminate at tier 1 (embedding fast-path) or a single tool call — see
`harness/routing/embeddings.py` (tier 1) and `harness/routing/triage.py`
(tiers 2/3) for the escalation chain; `TieredRouter` below composes them.

`StaticRouter` is a placeholder used before routing is wired up (Phase 1) and
in tests — always dispatches to the given target at tier 0/full confidence.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from semantic_router import Route as SemanticRoute
from semantic_router.encoders.base import DenseEncoder
from semantic_router.routers import SemanticRouter as SemanticRouterImpl

if TYPE_CHECKING:
    from harness.core.context import RunContext
    from harness.routing.triage import LLMTriage

logger = logging.getLogger("harness.routing")


class RouteMode(str, Enum):
    DIRECT = "direct"  # router answered it directly; no loop
    TOOL_LOOP = "tool_loop"  # single-agent ReAct-style tool loop
    FLOW = "flow"  # named deterministic/agentic flow (e.g. orchestrator-workers)


@dataclass
class RouteDecision:
    mode: RouteMode
    target: str
    confidence: float
    tier: int  # which routing tier produced this decision (1=embedding, 2=function-call, 3=LLM triage)


@runtime_checkable
class Router(Protocol):
    async def route(self, request: str, ctx: "RunContext") -> RouteDecision: ...


class StaticRouter:
    """Always routes to a fixed (mode, target) at full confidence. Useful for
    Phase 1 bring-up and for flows/tests that want to bypass routing."""

    def __init__(self, mode: RouteMode = RouteMode.TOOL_LOOP, target: str = "single_agent") -> None:
        self.mode = mode
        self.target = target

    async def route(self, request: str, ctx: "RunContext") -> RouteDecision:
        return RouteDecision(mode=self.mode, target=self.target, confidence=1.0, tier=0)


@dataclass
class _RouteMeta:
    mode: RouteMode
    target: str


class TieredRouter:
    """The production router: tier-1 embedding fast-path (semantic-router's
    `SemanticRouter`) escalating to a tier-2/3 LLM triage agent when
    confidence is below threshold. Router-tier ablation
    (`ctx.ablations.router_tier`) caps how far a run is allowed to escalate —
    set to 1 to force embedding-only routing for a cost/latency benchmark.

    Route decision + confidence + tier are logged as span attributes by
    `core/loop.py:AgentLoop` — routing accuracy becomes an eval dataset for
    free (see harness/evals/test_router.py).
    """

    def __init__(
        self,
        encoder: DenseEncoder,
        routes_config: list[dict[str, Any]],
        triage: "LLMTriage | None" = None,
        confidence_threshold: float = 0.35,
        fallback: RouteDecision | None = None,
    ) -> None:
        self._meta: dict[str, _RouteMeta] = {
            r["name"]: _RouteMeta(mode=RouteMode(r["mode"]), target=r["target"]) for r in routes_config
        }
        semantic_routes = [SemanticRoute(name=r["name"], utterances=r["utterances"]) for r in routes_config]
        self._semantic_router = SemanticRouterImpl(encoder=encoder, routes=semantic_routes, auto_sync="local")
        self.triage = triage
        self.confidence_threshold = confidence_threshold
        # Safe default when nothing matches confidently and triage is
        # unavailable/ablated-off: fail toward the bounded tool loop, never
        # toward an unrouted "flow" with a wider blast radius.
        self.fallback = fallback or RouteDecision(mode=RouteMode.TOOL_LOOP, target="single_agent", confidence=0.0, tier=1)

    async def route(self, request: str, ctx: "RunContext") -> RouteDecision:
        # SemanticRouter's __call__ is synchronous; hop to a thread so a slow
        # encoder call never blocks the event loop.
        choice = await asyncio.to_thread(self._semantic_router, request)
        score = choice.similarity_score or 0.0

        if choice.name is not None and (score >= self.confidence_threshold or ctx.ablations.router_tier < 2):
            meta = self._meta[choice.name]
            return RouteDecision(mode=meta.mode, target=meta.target, confidence=score, tier=1)

        if self.triage is not None and ctx.ablations.router_tier >= 2:
            try:
                return await self.triage.route(request)
            except Exception:  # noqa: BLE001 - triage failure should degrade, not crash routing
                logger.exception("tier-2/3 triage failed; falling back to tier-1 best guess")

        if choice.name is not None:
            meta = self._meta[choice.name]
            return RouteDecision(mode=meta.mode, target=meta.target, confidence=score, tier=1)
        return self.fallback
