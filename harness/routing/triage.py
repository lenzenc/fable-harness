"""Tier-2/3 routing fallback: when the embedding fast-path (tier 1) can't
confidently classify a request, escalate to a small/cheap LLM that returns a
structured decision via Pydantic AI's typed output — more interpretable and
extensible than prompt-engineered "pick a name from this list" routing, and
it works with small models without fine-tuning. Ablatable via
`ctx.ablations.router_tier` (see harness/routing/router.py:TieredRouter).
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from harness.routing.router import RouteDecision, RouteMode

TRIAGE_INSTRUCTIONS = """You are the harness's routing triage step. Classify the user's
request into exactly one of:
- mode="direct": you can answer directly in `target` (no tools/flow needed) — small talk, trivial asides.
- mode="tool_loop", target="single_agent": needs a single bounded tool-using agent (e.g. a lookup, a calculation).
- mode="flow", target="orchestrator": needs the multi-step advisor-prep flow (pulling together account/meeting/product/market context).
Always return your best guess even under ambiguity, with a calibrated confidence in [0, 1]."""


class TriageOutput(BaseModel):
    mode: Literal["direct", "tool_loop", "flow"]
    target: str = Field(
        description="Direct answer text if mode=direct, otherwise the flow/agent target name "
        "(e.g. 'single_agent', 'orchestrator')."
    )
    confidence: float = Field(ge=0.0, le=1.0)


class LLMTriage:
    """Tier-2/3 fallback: a small Pydantic AI agent with structured output.
    `tier` is stamped onto the returned RouteDecision so the escalation step
    that produced a decision is always visible in telemetry."""

    def __init__(self, model_id: str, tier: int = 2, test_model: Model | None = None) -> None:
        self.model_id = model_id
        self.tier = tier
        self.test_model = test_model

    async def route(self, request: str) -> RouteDecision:
        agent: Agent[None, TriageOutput] = Agent(
            self.model_id,
            output_type=TriageOutput,
            instructions=TRIAGE_INSTRUCTIONS,
            defer_model_check=self.test_model is not None,
        )
        override_ctx = agent.override(model=self.test_model) if self.test_model is not None else nullcontext()
        with override_ctx:
            result = await agent.run(request)
        output: TriageOutput = getattr(result, "output", None) or getattr(result, "data")
        return RouteDecision(mode=RouteMode(output.mode), target=output.target, confidence=output.confidence, tier=self.tier)
