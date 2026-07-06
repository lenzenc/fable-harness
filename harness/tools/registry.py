"""Tool registry: the harness's action-dispatch seam.

Tools are *registered*, not hardcoded into agents/flows — name, typed schema,
risk tier, cost estimate, owner, description. `ToolRegistry` is a Protocol so
alternate backends (e.g. one that proxies to an MCP server, see
harness/tools/mcp_client.py) can be swapped in without touching flows or the
core loop. `scoped(flow)` implements "minimum tool set per step": expose only
what the current flow needs, since large tool catalogs measurably degrade
tool selection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from opentelemetry import trace

from harness.core.permissions import ApprovalGate, PermissionDenied, RiskTier

tracer = trace.get_tracer("harness.tools")


@dataclass
class ToolSpec:
    """Registered tool metadata — the audit substrate for what an agent could
    possibly have done, independent of what it did."""

    name: str
    description: str
    risk_tier: RiskTier
    handler: Callable[..., Awaitable[Any]]
    parameters_schema: dict[str, Any] = field(default_factory=dict)
    cost_estimate: float = 0.0
    owner: str = "unknown"
    flows: frozenset[str] = field(default_factory=frozenset)  # empty == available to all flows


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]
    flow: str = "default"


@dataclass
class ToolResult:
    tool_name: str
    output: Any = None
    error: str | None = None
    risk_tier: RiskTier | None = None
    duration_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class PermissionGrants:
    """Per-run approval wiring handed to `execute` — lets a single registry
    serve many concurrent runs with different approval policies/callbacks."""

    gate: ApprovalGate = field(default_factory=ApprovalGate)


@runtime_checkable
class ToolRegistry(Protocol):
    def scoped(self, flow: str) -> list[ToolSpec]: ...

    async def execute(self, call: ToolCall, grants: PermissionGrants) -> ToolResult: ...


class InMemoryToolRegistry:
    """MVP ToolRegistry: an in-process dict of ToolSpecs. Wraps every call in
    a `harness.tool` span (name + risk tier as attributes) and enforces the
    permission matrix/approval gate *before* dispatch — the registry is the
    single choke point tool execution passes through, so this is where
    audit + enforcement live regardless of which flow initiated the call.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def scoped(self, flow: str) -> list[ToolSpec]:
        return [t for t in self._tools.values() if not t.flows or flow in t.flows]

    async def execute(self, call: ToolCall, grants: PermissionGrants) -> ToolResult:
        spec = self._tools.get(call.tool_name)
        if spec is None:
            return ToolResult(tool_name=call.tool_name, error=f"unknown tool '{call.tool_name}'")

        with tracer.start_as_current_span(f"execute_tool {spec.name}") as span:
            span.set_attribute("gen_ai.tool.name", spec.name)
            span.set_attribute("harness.tool.risk_tier", spec.risk_tier.value)
            span.set_attribute("harness.tool.flow", call.flow)
            start = time.monotonic()
            try:
                await grants.gate.check(call, spec.risk_tier)
            except PermissionDenied as exc:
                span.set_attribute("error.type", "PermissionDenied")
                return ToolResult(
                    tool_name=spec.name,
                    error=str(exc),
                    risk_tier=spec.risk_tier,
                    duration_seconds=time.monotonic() - start,
                )
            try:
                output = await spec.handler(**call.arguments)
            except Exception as exc:  # noqa: BLE001 - tool failures are data, not harness bugs
                span.set_attribute("error.type", type(exc).__name__)
                return ToolResult(
                    tool_name=spec.name,
                    error=str(exc),
                    risk_tier=spec.risk_tier,
                    duration_seconds=time.monotonic() - start,
                )
            return ToolResult(
                tool_name=spec.name,
                output=output,
                risk_tier=spec.risk_tier,
                duration_seconds=time.monotonic() - start,
            )
