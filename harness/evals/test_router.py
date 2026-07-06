"""Routing accuracy eval: labeled utterance -> expected route, no LLM
involved. `ctx.ablations.router_tier = 1` forces the TieredRouter to accept
its tier-1 (embedding) best guess regardless of confidence — see
TieredRouter.route in harness/routing/router.py — so this exercises exactly
the semantic-router + OpenAI-encoder path, matching the research brief's
"router gets its own golden dataset for free" plan.

This is the one eval in the suite that is NOT free/offline: the encoder
(harness/routing/embeddings.py) makes a real OpenAI embeddings API call, so
this module needs a real key and incurs a (tiny) real cost — every other
eval in this suite uses Pydantic AI's TestModel and touches NullMemoryStore,
never a real router.

The skip check below is intentionally *dynamic* (probe the encoder, skip on
failure) rather than checking env-var presence at collection time: `.env`
ships a non-empty placeholder value (`sk-REPLACE-ME`) for discoverability,
and at least one installed pytest plugin auto-loads `.env` unconditionally
(verified — `HARNESS_OPENAI_API_KEY` is present in-process under `pytest`
even though a plain `python` invocation doesn't see it), so a presence check
can't distinguish a placeholder from a real key. Probing and catching the
actual failure is the only reliable signal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from harness.config.settings import get_settings
from harness.core.context import AblationFlags, RunContext
from harness.routing.embeddings import get_encoder
from harness.routing.router import TieredRouter

ROUTES_YAML = Path(__file__).parent.parent / "routing" / "routes.yaml"
GOLDEN_PATH = Path(__file__).parent / "golden" / "routes.jsonl"

# Empirically, tier-1-only accuracy on held-out paraphrases sits below what a
# tier-2 LLM triage fallback would recover — this is the floor the fast path
# alone should clear, not the full-system target.
MIN_ACCURACY = 0.6


def _load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN_PATH.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def tier1_router() -> TieredRouter:
    # get_settings() bridges HARNESS_OPENAI_API_KEY -> OPENAI_API_KEY (the
    # unprefixed name OpenAIEncoder reads) — without this, a real key in
    # .env still wouldn't reach the encoder, since this module never
    # otherwise touches settings.
    try:
        get_settings()
    except Exception as exc:  # noqa: BLE001 - e.g. anthropic_api_key missing entirely in a stripped-down test env
        pytest.skip(f"tier-1 routing eval could not load settings ({exc!r}); skipping")

    routes_config = yaml.safe_load(ROUTES_YAML.read_text())["routes"]
    try:
        encoder = get_encoder()
    except Exception as exc:  # noqa: BLE001 - e.g. OpenAIEncoder raises ValueError with no key configured at all
        pytest.skip(f"tier-1 routing eval could not construct an OpenAI encoder ({exc!r}); skipping")
    return TieredRouter(encoder=encoder, routes_config=routes_config, triage=None)


@pytest.mark.asyncio
async def test_tier1_routing_accuracy(tier1_router: TieredRouter) -> None:
    cases = _load_golden()
    ctx = RunContext(request="", session_id="eval", ablations=AblationFlags(router_tier=1))

    # Dynamic skip: probe with a throwaway call rather than trusting env-var
    # presence (see module docstring) — any failure here (missing/invalid
    # key, no network) means this eval can't run, not that routing is broken.
    try:
        await tier1_router.route("__probe__", ctx)
    except Exception as exc:  # noqa: BLE001 - any failure means "can't reach OpenAI embeddings", not a code bug
        pytest.skip(f"tier-1 routing eval could not reach OpenAI embeddings ({exc!r}); skipping")

    correct = 0
    misses = []
    for case in cases:
        decision = await tier1_router.route(case["utterance"], ctx)
        mode_ok = decision.mode.value == case["expected_mode"]
        # direct-mode targets are free-text answers, not a stable id — only
        # tool_loop/flow routes have a meaningful target to compare.
        target_ok = "expected_target" not in case or decision.target == case["expected_target"]
        if mode_ok and target_ok:
            correct += 1
        else:
            misses.append((case["utterance"], case.get("expected_mode"), decision.mode.value, decision.target, decision.confidence))

    accuracy = correct / len(cases)
    assert accuracy >= MIN_ACCURACY, f"tier-1 routing accuracy {accuracy:.2f} below floor {MIN_ACCURACY}; misses={misses}"
