"""Configuration loading: `.env` (secrets, deployment-specific values) via
pydantic-settings, plus the versioned `harness.yaml` policy overlay
(budgets, risk matrix, thresholds, pricing) that's meant to live in git and
be reviewed like code — a deliberate split so rotating a key never touches a
file a compliance reviewer needs to read.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from harness.core.budgets import Budget, ModelPricing
from harness.core.permissions import PermissionMatrix, Policy, RiskTier

CONFIG_DIR = Path(__file__).parent
HARNESS_YAML_PATH = CONFIG_DIR / "harness.yaml"


class Settings(BaseSettings):
    """Everything sourced from `.env` / the process environment, prefixed
    `HARNESS_` (see .env.example at the repo root)."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="HARNESS_", extra="ignore")

    anthropic_api_key: SecretStr
    model_id: str = "anthropic:claude-sonnet-5"
    triage_model_id: str = "anthropic:claude-haiku-4-5-20251001"
    # Backs both tier-1 semantic routing and pgvector memory embeddings —
    # one embedding space (see harness/routing/embeddings.py).
    openai_api_key: SecretStr
    embedding_model_id: str = "text-embedding-3-small"
    database_url: str = "postgresql://harness:harness@localhost:5432/harness"
    otel_exporter_otlp_endpoint: str = "http://localhost:4318"
    # PII double-guard, half 1 (see telemetry/otel.py): only capture full
    # prompt/response content on spans in debug environments, never by default.
    otel_include_content: bool = False

    # Default ablation flags — per-run RunContext.ablations can override any of these.
    memory_on: bool = True
    critic_on: bool = True
    router_tier: int = 3


class HarnessConfig(BaseModel):
    """The `harness.yaml` policy overlay."""

    budgets: dict[str, Budget]
    compaction_threshold: float = 0.75
    risk_matrix: dict[str, str] = Field(
        default_factory=lambda: {
            "read_only": "auto_allow",
            "writes_internal": "auto_allow",
            "financial": "require_approval",
            "destructive": "require_approval",
        }
    )
    routing_confidence_threshold: float = 0.7
    pricing: dict[str, ModelPricing] = Field(default_factory=dict)

    def permission_matrix(self) -> PermissionMatrix:
        return PermissionMatrix(policy={RiskTier(tier): Policy(policy) for tier, policy in self.risk_matrix.items()})

    def budget_for(self, name: str = "default") -> Budget:
        return self.budgets.get(name, self.budgets["default"])


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    # Every Agent(...) call site passes a bare "anthropic:model-id" string —
    # Pydantic AI's AnthropicProvider reads the unprefixed ANTHROPIC_API_KEY
    # env var directly, not our HARNESS_-prefixed Settings field, so it has
    # to be propagated here once rather than at every call site.
    # setdefault: don't clobber a key the user already exported directly.
    os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key.get_secret_value())
    # OpenAIEncoder (harness/routing/embeddings.py) reads OPENAI_API_KEY via
    # os.getenv directly and raises if empty — same bridge, same reasoning.
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key.get_secret_value())
    return settings


@lru_cache
def get_harness_config() -> HarnessConfig:
    raw = yaml.safe_load(HARNESS_YAML_PATH.read_text())
    return HarnessConfig.model_validate(raw)
