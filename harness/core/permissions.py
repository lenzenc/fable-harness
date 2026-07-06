"""Risk-tiered tool permissions.

Every tool the harness can call declares a `RiskTier`. The `PermissionMatrix`
maps tiers to an enforcement policy (auto-allow vs. require-approval), and
`ApprovalGate` is the seam a caller plugs a real approval mechanism into
(Slack prompt, human-in-the-loop UI, etc.) — the MVP default just denies
anything above read_only unless a callback says otherwise.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.tools.registry import ToolCall


class RiskTier(str, Enum):
    """Ascending risk order. Financial services default: only read_only is
    auto-approved; everything else requires an explicit approval gate."""

    READ_ONLY = "read_only"
    WRITES_INTERNAL = "writes_internal"
    FINANCIAL = "financial"
    DESTRUCTIVE = "destructive"


class Policy(str, Enum):
    AUTO_ALLOW = "auto_allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


DEFAULT_MATRIX: dict[RiskTier, Policy] = {
    RiskTier.READ_ONLY: Policy.AUTO_ALLOW,
    RiskTier.WRITES_INTERNAL: Policy.AUTO_ALLOW,
    RiskTier.FINANCIAL: Policy.REQUIRE_APPROVAL,
    RiskTier.DESTRUCTIVE: Policy.REQUIRE_APPROVAL,
}

# Type of an approval callback: given the tool call and its risk tier, return
# whether it's approved. Sync or async both work — ApprovalGate normalizes.
ApprovalCallback = Callable[["ToolCall", RiskTier], "bool | Awaitable[bool]"]


@dataclass
class PermissionMatrix:
    """Per-flow (or global) mapping of risk tier -> enforcement policy."""

    policy: dict[RiskTier, Policy] = field(default_factory=lambda: dict(DEFAULT_MATRIX))

    def policy_for(self, tier: RiskTier) -> Policy:
        return self.policy.get(tier, Policy.REQUIRE_APPROVAL)


class PermissionDenied(RuntimeError):
    """Raised when a tool call is blocked by the permission matrix / approval gate."""


@dataclass
class ApprovalGate:
    """Enforces a PermissionMatrix, escalating to an approval callback when the
    matrix requires it. No callback configured => require_approval tiers are
    denied by default (fail closed, the safe default for financial services).
    """

    matrix: PermissionMatrix = field(default_factory=PermissionMatrix)
    on_approval_needed: ApprovalCallback | None = None

    async def check(self, call: "ToolCall", tier: RiskTier) -> None:
        policy = self.matrix.policy_for(tier)
        if policy == Policy.AUTO_ALLOW:
            return
        if policy == Policy.DENY:
            raise PermissionDenied(f"tool '{call.tool_name}' denied: risk tier {tier.value} is not allowed")
        # REQUIRE_APPROVAL
        if self.on_approval_needed is None:
            raise PermissionDenied(
                f"tool '{call.tool_name}' requires approval (risk tier {tier.value}) "
                "but no approval gate is configured — failing closed"
            )
        result = self.on_approval_needed(call, tier)
        if hasattr(result, "__await__"):
            approved = await result  # type: ignore[assignment]
        else:
            approved = result
        if not approved:
            raise PermissionDenied(f"tool '{call.tool_name}' (risk tier {tier.value}) was not approved")
