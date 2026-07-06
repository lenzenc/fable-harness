"""Single-agent tool loop — the default execution path most traffic should
terminate at: one Pydantic AI Agent, a bounded tool set scoped from the
ToolRegistry, budgets, and the harness's stop-condition policies layered on
top of Pydantic AI's own tool-calling loop.

Tool functions registered on the Agent are thin typed wrappers that delegate
to `services.tools.execute(...)` — the ToolRegistry stays the single choke
point for permission enforcement, telemetry spans, and audit, regardless of
which flow initiated the call. Arbitrary/dynamic-schema tools (e.g. from an
MCP server) are bridged via Pydantic AI's native MCP client support rather
than through this per-tool wrapper pattern — see harness/tools/mcp_client.py.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models import Model

from harness.core.context import RunContext
from harness.core.loop import BudgetExhaustedError, RepeatedFailureDetector
from harness.flows.base import FlowResult, HarnessServices
from harness.memory.extraction import FactExtractor
from harness.tools.registry import PermissionGrants, ToolCall

FLOW_NAME = "single_agent"
MAX_CONTEXT_TOKENS = 150_000
COMPACTION_THRESHOLD = 0.75


class SingleAgentFlow:
    """Default Flow: bounded ReAct-style tool loop via Pydantic AI."""

    def __init__(self, model_id: str, test_model: Model | None = None, fact_extractor: FactExtractor | None = None) -> None:
        """`test_model`, when set (e.g. Pydantic AI's `TestModel`/`FunctionModel`),
        overrides the real model for the duration of the run — zero token
        spend, deterministic tool-schema-satisfying behavior. Used by the CLI's
        `--test-model` flag and by the eval suite (harness/evals).

        `fact_extractor`, when set, replaces the naive "remember the whole
        Q&A pair" fallback with the proper async post-turn extraction step
        (harness/memory/extraction.py) that pulls out only durable facts."""
        self.model_id = model_id
        self.test_model = test_model
        self.fact_extractor = fact_extractor

    async def run(self, request: str, ctx: RunContext, services: HarnessServices) -> FlowResult:
        # Instrumentation is enabled globally via Agent.instrument_all() in
        # harness/telemetry/otel.py:configure_telemetry() — no per-instance
        # `instrument=` kwarg exists on Agent in this pydantic-ai version.
        # defer_model_check: when a test_model override is in play we never
        # actually touch the real provider, so don't require its credentials
        # (e.g. ANTHROPIC_API_KEY) just to construct the Agent.
        agent: Agent = Agent(self.model_id, defer_model_check=self.test_model is not None)
        detector = RepeatedFailureDetector()
        grants = PermissionGrants(gate=services.approval_gate) if services.approval_gate else PermissionGrants()
        call_count = {"n": 0}

        for spec in services.tools.scoped(FLOW_NAME):
            _wire_demo_tool(agent, spec.name, services, grants, detector, call_count)

        memory_context = ""
        if ctx.ablations.memory_on:
            recalled = await services.memory.recall(request, ctx.user_scope, k=5)
            if recalled:
                memory_context = "\n".join(f"- {m.content}" for m in recalled)
        prompt = request if not memory_context else f"Relevant memory:\n{memory_context}\n\nRequest: {request}"

        override_ctx = agent.override(model=self.test_model) if self.test_model is not None else nullcontext()
        try:
            with override_ctx:
                result = await agent.run(prompt, usage_limits=services.budget.usage_limits())
        except UsageLimitExceeded as exc:
            raise BudgetExhaustedError(str(exc)) from exc

        # `result.usage` is a RunUsage property (not a method) in pydantic-ai 2.5;
        # it already tracks tool_calls natively so our own call_count is a
        # cross-check rather than the source of truth.
        usage = result.usage
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        tool_calls = getattr(usage, "tool_calls", 0) or call_count["n"]
        services.budget.record_usage(input_tokens=input_tokens, output_tokens=output_tokens, tool_calls=tool_calls)
        if services.budget.check_wall_clock():
            raise BudgetExhaustedError(services.budget.status.exhausted_reason or "wall_clock_exceeded")

        output = getattr(result, "output", None)
        if output is None:
            output = getattr(result, "data", None)  # pre-2.x Pydantic AI naming fallback

        if ctx.ablations.memory_on and output:
            if self.fact_extractor is not None:
                facts = await self.fact_extractor.extract(request, str(output), source=FLOW_NAME)
            else:
                from harness.memory.store import MemoryItem

                facts = [MemoryItem(content=f"Q: {request}\nA: {output}", scope=ctx.user_scope, source=FLOW_NAME)]
            if facts:
                await services.memory.remember(facts, scope=ctx.user_scope)

        transcript = [_serialize_message(m) for m in result.all_messages()]
        return FlowResult(output=output, success=True, stop_reason="success", transcript=transcript)


def _serialize_message(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json")
    return {"repr": str(message)}


def _wire_demo_tool(
    agent: Agent,
    tool_name: str,
    services: HarnessServices,
    grants: PermissionGrants,
    detector: RepeatedFailureDetector,
    call_count: dict[str, int],
) -> None:
    """Registers one of the known demo tools (harness/tools/demo_tools.py) on
    the Agent with a concrete typed signature — Pydantic AI needs a real
    Python signature to derive each tool's JSON schema, so this is
    necessarily per-tool rather than a single generic bridge."""

    async def _dispatch(name: str, arguments: dict[str, Any]) -> Any:
        call_count["n"] += 1
        call = ToolCall(tool_name=name, arguments=arguments, flow=FLOW_NAME)
        result = await services.tools.execute(call, grants)
        detector.record(result, arguments)
        if not result.ok:
            raise RuntimeError(result.error)
        return result.output

    if tool_name == "get_current_time":

        @agent.tool_plain(name="get_current_time")
        async def get_current_time() -> str:
            """Return the current UTC time as an ISO-8601 string."""
            return await _dispatch("get_current_time", {})

    elif tool_name == "echo":

        @agent.tool_plain(name="echo")
        async def echo(text: str) -> str:
            """Echo back the given text (smoke-test tool)."""
            return await _dispatch("echo", {"text": text})
