"""Budgets: the harness's hard resource ceiling, enforced in code — not
requested of the model.

Pydantic AI's `UsageLimits` natively covers token/request/tool-call counts
and raises `UsageLimitExceeded` when breached. It does *not* cover wall-clock
or dollar cost, and raising is the wrong ergonomics for a harness that wants
a clean, catchable "budget exhausted, here's the partial result" outcome
rather than an uncaught exception. `BudgetTracker` wraps `UsageLimits` for
what PAI covers natively and adds wall-clock + cost tracking on top.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from pydantic_ai.usage import UsageLimits


@dataclass
class Budget:
    max_total_tokens: int | None = None
    max_tool_calls: int | None = None
    max_wall_clock_seconds: float | None = None
    max_cost_usd: float | None = None


@dataclass
class BudgetStatus:
    tokens_used: int = 0
    tool_calls_used: int = 0
    elapsed_seconds: float = 0.0
    cost_usd: float = 0.0
    exhausted_reason: str | None = None

    @property
    def exhausted(self) -> bool:
        return self.exhausted_reason is not None


@dataclass
class ModelPricing:
    """USD per 1K tokens for a given model — used only for the cost budget,
    which Pydantic AI does not track natively."""

    input_per_1k: float = 0.0
    output_per_1k: float = 0.0


class BudgetTracker:
    def __init__(
        self,
        budget: Budget,
        model_id: str,
        pricing: dict[str, ModelPricing] | None = None,
    ) -> None:
        self.budget = budget
        self.model_id = model_id
        self.pricing = pricing or {}
        self._start = time.monotonic()
        self.status = BudgetStatus()

    def usage_limits(self) -> UsageLimits:
        """The native Pydantic AI side of the budget — pass this into
        `agent.run(..., usage_limits=...)`."""
        return UsageLimits(
            total_tokens_limit=self.budget.max_total_tokens,
            tool_calls_limit=self.budget.max_tool_calls,
        )

    def record_usage(self, *, input_tokens: int, output_tokens: int, tool_calls: int) -> None:
        """Call after each model turn to update the harness-side ledger
        (cost + wall-clock) that Pydantic AI doesn't track."""
        self.status.tokens_used += input_tokens + output_tokens
        self.status.tool_calls_used = tool_calls
        price = self.pricing.get(self.model_id)
        if price:
            self.status.cost_usd += (input_tokens / 1000) * price.input_per_1k
            self.status.cost_usd += (output_tokens / 1000) * price.output_per_1k
        self._refresh_wall_clock()
        self._check_exhaustion()

    def check_wall_clock(self) -> bool:
        """Cheap poll the loop can call between steps without a model turn.
        Returns True if the budget is (now) exhausted."""
        self._refresh_wall_clock()
        self._check_exhaustion()
        return self.status.exhausted

    def _refresh_wall_clock(self) -> None:
        self.status.elapsed_seconds = time.monotonic() - self._start

    def _check_exhaustion(self) -> None:
        if self.status.exhausted_reason is not None:
            return  # sticky — first exhaustion reason wins
        b, s = self.budget, self.status
        if b.max_wall_clock_seconds is not None and s.elapsed_seconds > b.max_wall_clock_seconds:
            s.exhausted_reason = f"wall_clock_exceeded:{s.elapsed_seconds:.1f}s>{b.max_wall_clock_seconds}s"
        elif b.max_cost_usd is not None and s.cost_usd > b.max_cost_usd:
            s.exhausted_reason = f"cost_exceeded:${s.cost_usd:.4f}>${b.max_cost_usd}"
