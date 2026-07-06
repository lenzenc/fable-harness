# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Python tooling is `uv` exclusively — never bare `pip`/`python`.

```bash
cp .env.example .env        # then fill in HARNESS_ANTHROPIC_API_KEY + HARNESS_OPENAI_API_KEY
uv sync                     # install deps
uv sync --extra eval        # + deepeval/ragas/pytest/pytest-asyncio

docker compose up -d        # postgres (pgvector), otel-collector, phoenix

uv run harness "prep me for my meeting with client X"   # run one turn end-to-end
uv run harness --test-model "..."                        # stubs the LLM only (not embeddings — see Embeddings below)

uv run pytest                                             # full eval suite
uv run pytest harness/evals/test_flows.py -v              # single file
uv run pytest harness/evals/test_flows.py::test_single_agent_flow_succeeds_with_test_model  # single test

uv run python -m harness.evals.ablations   # standalone ablation-delta report (also runs as pytest tests)
```

No linter/formatter is configured in this repo.

### Resetting the DB after a schema change

`init.sql` runs only on first volume creation (`docker-entrypoint-initdb.d`). If you change `harness/memory/init.sql` (e.g. the embedding vector dimension), you must reset the volume — `CREATE TABLE IF NOT EXISTS` will not alter an existing column:

```bash
docker compose down -v && docker compose up -d
```

### Datadog dual-export (opt-in)

The default `otel-collector` service exports to Phoenix only. A separate `otel-collector-datadog` service (Compose profile `datadog`) exports to both; it requires `DD_API_KEY` in `.env` and must not run alongside the default collector (both bind host ports 4318/4319):

```bash
docker compose stop otel-collector
docker compose --profile datadog up -d otel-collector-datadog
```

## Architecture

### Design thesis

The harness — not the model — is where behavior lives. Every cross-cutting concern (routing, memory, tools, retrieval, telemetry) is expressed as a Python `Protocol` so it can be swapped without touching the core loop, and every feature (`memory_on`, `critic_on`, `router_tier`) is an ablation flag on `RunContext` so its actual effect can be measured, not assumed (`harness/evals/ablations.py` asserts exact behavioral deltas, e.g. `critic_on` adds exactly one agent call).

### Request flow

`harness/cli.py` is the single wiring point. One CLI invocation:

1. `get_settings()` / `get_harness_config()` load config (see Config split below).
2. `_build_router()` constructs a `TieredRouter` (`harness/routing/router.py`): tier-1 is `semantic-router`'s `SemanticRouter` over a local embedding fast-path; tier-2 is `LLMTriage` (`harness/routing/triage.py`), an LLM fallback when tier-1 confidence is below `routing_confidence_threshold`. `RouteDecision.mode` is `direct | tool_loop | flow`, not just a destination name.
3. `AgentLoop.run_turn()` (`harness/core/loop.py`) dispatches the decision to a registered `Flow` — `SingleAgentFlow` (bounded ReAct tool loop) or `OrchestratorFlow` (parallel-worker advisor-prep fan-out with a citation-grounded synthesis + ablatable critic pass). The loop owns budget/repeated-failure/compaction policy that Pydantic AI has no opinion about; it converts `RepeatedFailureError`/`BudgetExhaustedError`/`PermissionDenied` into a clean `FlowResult` rather than letting them escape uncaught.
4. Flows call tools exclusively through `ToolRegistry.execute()` (`harness/tools/registry.py`) — the single choke point for risk-tiered permission enforcement (`RiskTier`: `read_only|writes_internal|financial|destructive`) and `harness.tool` telemetry spans, regardless of whether the tool is in-process or MCP-backed.
5. Memory reads/writes go through `MemoryStore` (`harness/memory/store.py`); the real backend is `PgVectorMemoryStore` (temporal `valid_from`/`valid_to`, never hard-deletes — `invalidate()` closes rows out instead).

### Config split

`harness/config/settings.py` (`.env`, secrets/deployment values, `HARNESS_`-prefixed) vs. `harness/config/harness.yaml` (budgets, risk matrix, thresholds, pricing — versioned in git, reviewable like code). `get_settings()` also bridges `HARNESS_ANTHROPIC_API_KEY`/`HARNESS_OPENAI_API_KEY` into the unprefixed `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` env vars those SDKs read directly — any new provider secret needs the same `os.environ.setdefault(...)` bridge, since nothing else performs it.

### Permissions

`ApprovalGate` (`harness/core/permissions.py`) fails **closed** by default: a `require_approval` tier with no `on_approval_needed` callback configured is denied, not allowed. The CLI wires a demo-only auto-approve callback (`lambda call, tier: True`) so the out-of-the-box demo doesn't get blocked on the orchestrator's `financial`-tier `account_lookup` — a real deployment replaces that callback, not the fail-closed default.

### Embeddings — one shared space

`harness/routing/embeddings.py:get_encoder()` is the single embedding factory, consumed by both tier-1 routing and `PgVectorMemoryStore` — there is exactly one embedding space in the system, not two. It wraps OpenAI's `text-embedding-3-small` (1536-dim, `EMBEDDING_DIM` constant) via `semantic-router`'s `OpenAIEncoder`, which is network-backed (requires `OPENAI_API_KEY`) — **`--test-model` only stubs the LLM, not embeddings**, so routing/memory always call OpenAI regardless of that flag. `pgvector_store.py`'s `_embed()` asserts the returned vector length equals `EMBEDDING_DIM`; changing `embedding_model_id` to a different-dimension model requires updating `harness/memory/init.sql`'s `VECTOR(n)` column and resetting the DB volume (see Commands). pgvector's HNSW/IVFFlat indexes cap at 2000 dimensions — `text-embedding-3-large` (3072-dim) would need `halfvec` or no index, not handled here.

`@lru_cache` on `get_encoder()` keys on the literal call signature, not the resolved default value — `get_encoder()` and `get_encoder("text-embedding-3-small")` are different cache entries even though they resolve to the same client. All call sites must pass the identical `settings.embedding_model_id` string (they currently do, in `cli.py`) or the model silently loads/authenticates twice.

### MCP integration pattern

`harness/tools/mcp_client.py` bridges an MCP server (e.g. `harness/retrieval/mcp_server`, a stub keyword-search server standing in for a future LlamaIndex-backed one) into a Pydantic AI `Agent` via `MCPToolset` — Pydantic AI handles arbitrary-schema tool bridging natively, so this is the mechanism for dynamic/external tool schemas, as opposed to in-process tools which are typed Python functions registered on the `Agent` directly (see `flows/single_agent.py`'s per-tool `@agent.tool_plain` wrappers). The harness's own permission/audit choke point is preserved for MCP tools via the `process_tool_call` hook, which runs the same `ApprovalGate` check before the MCP server is called.

### Telemetry

Pydantic AI emits `gen_ai.*` spans natively via `Agent.instrument_all()` (`harness/telemetry/otel.py`) — no Logfire dependency. PII protection is a double guard: `HARNESS_OTEL_INCLUDE_CONTENT` (default `false`) disables prompt/response content capture at the source, and the OTel Collector's `redaction` processor (`harness/telemetry/collector/config.yaml`) strips PII-shaped values and the content attributes from anything that reaches it regardless. Custom harness attributes live under a `harness.*` namespace (route decisions, budget state, ablation flags) to avoid colliding with evolving `gen_ai.*` semconv, which is still "Development" status upstream — `SEMCONV_VERSION` in `otel.py` is pinned deliberately.

### Eval suite

`harness/evals/` runs almost entirely on Pydantic AI's `TestModel`/`FunctionModel` (zero token spend, no API key) by constructing flows directly and bypassing the router — `test_flows.py`, `test_harness.py`, `ablations.py` never touch a real model or a real embedding call. The one exception is `test_router.py` (real OpenAI embeddings calls, real cost): it skips itself dynamically by probing the encoder and catching failures at runtime, not by checking env-var presence at collection time — a presence check is unreliable here because `.env` ships a non-empty placeholder and at least one installed pytest plugin auto-loads `.env` unconditionally (verified: `uv run pytest` sees `HARNESS_OPENAI_API_KEY` from `.env` even though a plain `uv run python` does not).

`pyproject.toml`'s `python_files` config includes `ablations.py` explicitly (it doesn't match the `test_*.py` pattern) so its `test_*` functions run under `uv run pytest` in addition to being runnable standalone.
