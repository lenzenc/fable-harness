"""Orchestrator-workers flow: the advisor-prep flagship demo from the
research brief. A fixed decomposition (account data / meeting history /
retrieval-backed product+market context) fans out to parallel workers, then
a synthesis step combines them with mandatory citation grounding — every
claim must attribute to a source id, the primary mitigation for synthesis
hallucination (the model blending two unrelated sources into a claim neither
supports). An ablatable critic pass (`ctx.ablations.critic_on`) reviews the
synthesis for uncited/unsupported claims before returning.

The decomposition is deliberately fixed rather than planned by an LLM —
matching the research brief's named flagship use case exactly keeps this
flow's behavior predictable and cheap to eval. A dynamic planner (deciding
*which* workers to invoke per request) is a natural extension point, not
built here: complexity should be earned by evals showing the fixed
decomposition is insufficient, not assumed upfront.
"""

from __future__ import annotations

import asyncio
from contextlib import nullcontext

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model

from harness.core.context import RunContext
from harness.core.permissions import RiskTier
from harness.flows.base import FlowResult, HarnessServices
from harness.memory.extraction import FactExtractor
from harness.tools.mcp_client import build_retrieval_toolset
from harness.tools.registry import PermissionGrants, ToolCall

FLOW_NAME = "orchestrator"

SYNTHESIS_INSTRUCTIONS = """You are preparing a financial advisor for a client meeting.
Combine the account data, meeting history, and retrieved product/market context below
into a concise prep summary. Every factual claim MUST cite a source id in brackets, e.g.
[doc-2] or [account]. If a claim cannot be supported by one of the given sources, omit it
rather than inventing a citation."""

CRITIC_INSTRUCTIONS = """Review this advisor-prep summary against its source material below.
Remove any claim whose citation doesn't actually support it, or that has no citation at
all. Return the corrected summary and citation list — err toward cutting an unsupported
claim rather than keeping it."""


class SynthesisOutput(BaseModel):
    summary: str
    citations: list[str]


class OrchestratorFlow:
    def __init__(
        self,
        model_id: str,
        test_model: Model | None = None,
        client_key: str = "client_x",
        fact_extractor: FactExtractor | None = None,
    ) -> None:
        self.model_id = model_id
        self.test_model = test_model
        self.client_key = client_key
        self.fact_extractor = fact_extractor

    def _agent(self, output_type: type, instructions: str) -> Agent:
        return Agent(
            self.model_id,
            output_type=output_type,
            instructions=instructions,
            defer_model_check=self.test_model is not None,
        )

    async def _run_agent(self, agent: Agent, prompt: str):
        override_ctx = agent.override(model=self.test_model) if self.test_model is not None else nullcontext()
        with override_ctx:
            result = await agent.run(prompt)
        output = getattr(result, "output", None)
        return output if output is not None else getattr(result, "data")

    async def run(self, request: str, ctx: RunContext, services: HarnessServices) -> FlowResult:
        grants = PermissionGrants(gate=services.approval_gate) if services.approval_gate else PermissionGrants()

        async def _worker(tool_name: str, arguments: dict) -> str:
            call = ToolCall(tool_name=tool_name, arguments=arguments, flow=FLOW_NAME)
            result = await services.tools.execute(call, grants)
            return result.output if result.ok else f"[unavailable: {result.error}]"

        async def _retrieval_worker() -> str:
            toolset = build_retrieval_toolset(
                approval_gate=services.approval_gate, risk_tier=RiskTier.READ_ONLY, flow=FLOW_NAME
            )
            agent = Agent(self.model_id, toolsets=[toolset], defer_model_check=self.test_model is not None)
            return await self._run_agent(agent, f"Search for product and market context relevant to: {request}")

        # Parallel fan-out — this is what "orchestrator-workers" means as
        # distinct from prompt chaining: workers are independent until the
        # synthesis step below needs all of them together.
        account, meeting, retrieved = await asyncio.gather(
            _worker("account_lookup", {"client": self.client_key}),
            _worker("meeting_history", {"client": self.client_key}),
            _retrieval_worker(),
        )

        memory_context = ""
        if ctx.ablations.memory_on:
            recalled = await services.memory.recall(request, ctx.user_scope, k=5)
            if recalled:
                memory_context = "\nRemembered client context: " + "; ".join(m.content for m in recalled)

        sources_block = f"""Request: {request}

Account [account]: {account}
Meeting history [meeting]: {meeting}
Retrieved context: {retrieved}{memory_context}"""

        synthesis: SynthesisOutput = await self._run_agent(
            self._agent(SynthesisOutput, SYNTHESIS_INSTRUCTIONS), sources_block
        )

        if ctx.ablations.critic_on:
            critic_prompt = f"Summary: {synthesis.summary}\nCitations: {synthesis.citations}\n\nSources:\n{sources_block}"
            synthesis = await self._run_agent(self._agent(SynthesisOutput, CRITIC_INSTRUCTIONS), critic_prompt)

        if ctx.ablations.memory_on and self.fact_extractor is not None:
            facts = await self.fact_extractor.extract(request, synthesis.summary, source=FLOW_NAME)
            if facts:
                await services.memory.remember(facts, scope=ctx.user_scope)

        return FlowResult(output=synthesis.model_dump(), success=True, stop_reason="success")
