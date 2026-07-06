"""fable-harness: a thin-harness-with-thick-seams AI agent runtime.

Every boundary between components (router, memory, tools, retrieval,
telemetry) is a swappable `typing.Protocol`. See harness/core/loop.py for
the orchestration heart, and the repo README for the phase-by-phase build.
"""
