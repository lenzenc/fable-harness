"""MCP client: bridges MCP-provided tools (the retrieval stub today, a
Salesforce server later) into a Pydantic AI Agent.

Pydantic AI's `MCPToolset` handles arbitrary-schema tool bridging natively —
that's the whole reason MCP is the integration standard here rather than a
hand-rolled generic JSON-schema-to-python-function adapter (which in-process
tools need, see flows/single_agent.py, but MCP tools don't). What this
module adds on top is the harness's own choke point: every MCP tool call is
routed through the `ApprovalGate` via `process_tool_call` *before* it reaches
the server, so an MCP-provided tool gets the same risk-tiered
permission/audit treatment as an in-process ToolRegistry tool.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext as ToolRunContext
from pydantic_ai.mcp import CallToolFunc, FastMCPClient, MCPToolset, StdioTransport

from harness.core.permissions import ApprovalGate, RiskTier
from harness.tools.registry import ToolCall


def build_retrieval_toolset(
    approval_gate: ApprovalGate | None = None,
    risk_tier: RiskTier = RiskTier.READ_ONLY,
    flow: str = "orchestrator",
) -> MCPToolset:
    """Launches harness/retrieval/mcp_server as a stdio subprocess (via `uv
    run`, so it shares this project's environment) and wraps it as a
    Pydantic AI toolset ready to hand to `Agent(..., toolsets=[...])`.
    """
    transport = StdioTransport(command="uv", args=["run", "python", "-m", "harness.retrieval.mcp_server"])
    client = FastMCPClient(transport)

    async def process_tool_call(
        ctx: ToolRunContext[Any],
        call_tool: CallToolFunc,
        name: str,
        tool_args: dict[str, Any],
    ) -> Any:
        if approval_gate is not None:
            await approval_gate.check(ToolCall(tool_name=name, arguments=tool_args, flow=flow), risk_tier)
        return await call_tool(name, tool_args)

    return MCPToolset(client=client, process_tool_call=process_tool_call)
