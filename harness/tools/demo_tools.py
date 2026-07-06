"""Demo tools for bring-up and tests — deliberately trivial, deterministic,
and dependency-free, so the harness's plumbing (registry -> permissions ->
telemetry -> repeated-failure detection) can be exercised end-to-end without
needing a live external system. Real integrations (Salesforce, retrieval)
plug in the same way: a ToolSpec with a risk tier and an async handler.
"""

from __future__ import annotations

from datetime import UTC, datetime

from harness.core.permissions import RiskTier
from harness.tools.registry import ToolSpec


async def _get_current_time_impl() -> str:
    return datetime.now(UTC).isoformat()


async def _echo_impl(text: str) -> str:
    return text


# Demo CRM-style data — stands in for a real Salesforce MCP server (the
# research brief's named "second MCP surface") until one is wired up. The
# account lookup is deliberately tagged `financial` risk tier so the
# orchestrator flow exercises the approval-gate path, not just read_only.
_ACCOUNT_DATA = {"client_x": "Client X: $2.4M AUM, moderate risk profile, onboarded 2019."}
_MEETING_HISTORY = {"client_x": "Last meeting: Q1 portfolio review, discussed rebalancing toward fixed income."}


async def _account_lookup_impl(client: str) -> str:
    return _ACCOUNT_DATA.get(client, "no account data on file")


async def _meeting_history_impl(client: str) -> str:
    return _MEETING_HISTORY.get(client, "no meeting history on file")


def demo_tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="get_current_time",
            description="Return the current UTC time as an ISO-8601 string.",
            risk_tier=RiskTier.READ_ONLY,
            handler=_get_current_time_impl,
            owner="harness-core",
        ),
        ToolSpec(
            name="echo",
            description="Echo back the given text (smoke-test tool).",
            risk_tier=RiskTier.READ_ONLY,
            handler=_echo_impl,
            owner="harness-core",
        ),
        ToolSpec(
            name="account_lookup",
            description="Look up account/AUM data for a client by key (e.g. 'client_x').",
            risk_tier=RiskTier.FINANCIAL,
            handler=_account_lookup_impl,
            owner="crm-demo",
            flows=frozenset({"orchestrator"}),
        ),
        ToolSpec(
            name="meeting_history",
            description="Look up recent meeting history for a client by key (e.g. 'client_x').",
            risk_tier=RiskTier.READ_ONLY,
            handler=_meeting_history_impl,
            owner="crm-demo",
            flows=frozenset({"orchestrator"}),
        ),
    ]
