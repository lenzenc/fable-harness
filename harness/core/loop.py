"""Core agent loop: stop conditions, repeated-failure detection, compaction
triggering, and the router -> flow -> memory orchestration.

Pydantic AI's `Agent.run()` already drives the model/tool-calling turns
internally once tools are registered. What the harness owns on top of that —
and what this module provides as reusable policy, consumed by
flows/single_agent.py and flows/orchestrator.py — is everything Pydantic AI
has no opinion about: a hard stop on repeated identical tool failures,
context-compaction triggering, and the top-level dispatch from a
RouteDecision to the right Flow, with harness-owned control-flow exceptions
(budget exhaustion, repeated failure, permission denial) converted into a
clean `FlowResult` rather than left as uncaught exceptions.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from opentelemetry import trace

from harness.core.context import RunContext
from harness.core.permissions import PermissionDenied
from harness.flows.base import Flow, FlowResult, HarnessServices
from harness.routing.router import RouteMode
from harness.tools.registry import ToolResult

if TYPE_CHECKING:
    from harness.routing.router import RouteDecision

logger = logging.getLogger("harness.loop")
tracer = trace.get_tracer("harness.core")


class RepeatedFailureError(RuntimeError):
    """Raised when the same tool+args+error combination repeats — the
    harness's cheapest guard against a model looping on a broken action."""


class BudgetExhaustedError(RuntimeError):
    """Raised (by flows) when BudgetTracker reports exhaustion on the
    dimensions Pydantic AI's UsageLimits doesn't cover: wall-clock and cost."""


@dataclass
class RepeatedFailureDetector:
    """Hashes (tool_name, args, error) for each failed tool call. If the same
    hash recurs `max_repeats` times, raises so the flow can abort with a
    diagnostic instead of letting the model retry a broken action forever."""

    max_repeats: int = 2
    _seen: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def _key(result: ToolResult, arguments: dict[str, Any]) -> str:
        payload = json.dumps(
            {"tool": result.tool_name, "args": arguments, "error": result.error},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def record(self, result: ToolResult, arguments: dict[str, Any]) -> None:
        if result.ok:
            return
        key = self._key(result, arguments)
        count = self._seen.get(key, 0) + 1
        self._seen[key] = count
        if count >= self.max_repeats:
            raise RepeatedFailureError(
                f"tool '{result.tool_name}' failed identically {count}x with args={arguments!r}: {result.error}"
            )


def estimate_utilization(messages: list[Any], max_context_tokens: int, chars_per_token: int = 4) -> float:
    """Cheap context-utilization estimate (chars/4 heuristic — no tokenizer
    dependency for the MVP). Used to trigger compaction before the model's
    actual context window is exceeded."""
    if not max_context_tokens:
        return 0.0
    total_chars = sum(len(str(m)) for m in messages)
    return (total_chars / chars_per_token) / max_context_tokens


def compact_history(messages: list[Any], threshold: float, max_context_tokens: int) -> list[Any]:
    """MVP compaction policy: once utilization crosses `threshold`, summarize
    (truncate) the oldest message first, preserving recent turns which matter
    most for continuing the conversation. Deliberately the simplest policy
    that helps — Claude Code's five-stage progressive compaction is the
    mature reference to grow into as this becomes a bottleneck."""
    if estimate_utilization(messages, max_context_tokens) < threshold:
        return messages
    compacted = list(messages)
    while len(compacted) > 1 and estimate_utilization(compacted, max_context_tokens) >= threshold:
        oldest = compacted[0]
        text = str(oldest)
        if len(text) > 200:
            compacted[0] = f"[compacted: {text[:200]}... ({len(text)} chars truncated)]"
            break  # oldest entry is now short; re-evaluate utilization on the next call
        compacted.pop(0)
    return compacted


@dataclass
class AgentLoop:
    """Top-level orchestrator: router -> flow -> memory."""

    flows: dict[str, Flow]

    async def run_turn(self, ctx: RunContext, services: HarnessServices) -> FlowResult:
        with tracer.start_as_current_span("invoke_agent") as span:
            ctx.span = span

            decision: "RouteDecision | None" = None
            if services.router is not None:
                decision = await services.router.route(ctx.request, ctx)
                ctx.stamp("harness.route.mode", decision.mode.value)
                ctx.stamp("harness.route.target", decision.target)
                ctx.stamp("harness.route.confidence", decision.confidence)
                ctx.stamp("harness.route.tier", decision.tier)

            if decision is not None and decision.mode == RouteMode.DIRECT:
                return FlowResult(output=decision.target, success=True, stop_reason="direct_answer")

            target = decision.target if decision is not None else "single_agent"
            flow = self.flows.get(target)
            if flow is None:
                return FlowResult(output=None, success=False, stop_reason=f"unknown_flow:{target}")

            try:
                result = await flow.run(ctx.request, ctx, services)
            except RepeatedFailureError as exc:
                result = FlowResult(output=str(exc), success=False, stop_reason="repeated_failure")
            except BudgetExhaustedError as exc:
                result = FlowResult(output=str(exc), success=False, stop_reason="budget_exhausted")
            except PermissionDenied as exc:
                result = FlowResult(output=str(exc), success=False, stop_reason="permission_denied")

            ctx.stamp("harness.flow.stop_reason", result.stop_reason)
            ctx.stamp("harness.flow.success", result.success)
            ctx.stamp("harness.budget.tokens_used", services.budget.status.tokens_used)
            ctx.stamp("harness.budget.tool_calls_used", services.budget.status.tool_calls_used)
            ctx.stamp("harness.budget.cost_usd", round(services.budget.status.cost_usd, 6))
            ctx.stamp("harness.ablation.memory_on", ctx.ablations.memory_on)
            ctx.stamp("harness.ablation.critic_on", ctx.ablations.critic_on)
            ctx.stamp("harness.ablation.router_tier", ctx.ablations.router_tier)
            return result
