"""LangSmith trace wiring (Stage 10d, BRD §14, SPEC §1 Q4).

LangSmith SaaS auto-instruments LangGraph/LangChain — there is no manual
span code to write. Tracing turns on when ``LANGSMITH_TRACING`` is truthy
and ``LANGSMITH_API_KEY`` is present, both read from the environment
(``load_dotenv`` populates them from ``.env``; the key is NEVER hardcoded
or committed). What this module adds is the project-specific TAGGING so a
trace is findable and correlatable:

- :func:`trace_config` augments the ``RunnableConfig`` passed to a graph
  ``astream`` at each execution-entry boundary (the same ``run_context``
  sites Stage 10c wraps) with:
    * ``run_id`` — the LangSmith ROOT run id, set to the SAME nested
      ``run_id`` Stage 10c mints (parsed back to its ``uuid.UUID``). The
      trace and its structlog lines therefore share ONE id; no second,
      independent id is minted (the operator's locked 10d decision).
    * ``metadata`` — ``{run_id (hex), strategy_id, thread_id, stage}`` so
      the trace cross-references the structlog lines byte-for-byte on
      ``run_id`` and is searchable by ``strategy_id`` / ``stage``.
    * ``tags`` — ``strategy_id:<…>`` / ``stage:<…>`` / ``run_id:<…>`` for
      the LangSmith tag filter (BRD §14: "Tag traces with strategy_id and
      stage").

Graceful degradation: :func:`trace_config` only BUILDS config — whether a
trace is actually emitted is langchain's decision from ``LANGSMITH_TRACING``.
With tracing unset/false (or no API key) the augmented config is inert and
the graph runs identically; there is NO hard dependency (this module
imports only ``os`` + ``uuid``, never ``langsmith``).
"""

from __future__ import annotations

import os
import uuid
from typing import Any

_TRUTHY = {"1", "true", "yes", "on"}

# SPEC §1 Q4 — the LangSmith project all traces land in by default.
_DEFAULT_PROJECT = "ai-trading-agent"


def tracing_enabled() -> bool:
    """True iff ``LANGSMITH_TRACING`` is truthy in the environment.

    This MIRRORS langchain's own on/off decision; it is exposed for the
    startup status log and tests, not as a gate (langchain consults the
    env var directly when it decides whether to emit a trace).
    """
    return os.environ.get("LANGSMITH_TRACING", "").strip().lower() in _TRUTHY


def langsmith_project() -> str:
    """Return ``LANGSMITH_PROJECT`` or the SPEC §1 Q4 default."""
    return os.environ.get("LANGSMITH_PROJECT") or _DEFAULT_PROJECT


def trace_config(
    config: dict[str, Any],
    *,
    strategy_id: str,
    thread_id: str,
    run_id: str,
    stage: str | None = None,
) -> dict[str, Any]:
    """Return a COPY of ``config`` augmented with LangSmith trace fields.

    Pure + side-effect-free: the input ``config`` is never mutated, and
    ``configurable`` (the checkpointer's ``thread_id``) plus any
    pre-existing ``metadata`` / ``tags`` are preserved. Safe to call whether
    or not tracing is enabled — with tracing off the extra fields are inert.

    ``run_id`` is the Stage 10c nested execution id (``uuid4().hex``); it is
    set as the LangSmith ROOT ``run_id`` (parsed to ``uuid.UUID``) so the
    trace shares one identifier with the structlog lines, and ALSO carried
    in ``metadata.run_id`` in hex form for byte-exact log↔trace correlation.
    """
    out: dict[str, Any] = dict(config)

    # Mirror the nested run_id as the LangSmith root run id (no second id).
    out["run_id"] = uuid.UUID(run_id)

    metadata: dict[str, Any] = dict(out.get("metadata") or {})
    metadata["run_id"] = run_id
    metadata["strategy_id"] = strategy_id
    metadata["thread_id"] = thread_id
    if stage is not None:
        metadata["stage"] = stage
    out["metadata"] = metadata

    tags: list[str] = list(out.get("tags") or [])
    tags.append(f"strategy_id:{strategy_id}")
    tags.append(f"run_id:{run_id}")
    if stage is not None:
        tags.append(f"stage:{stage}")
    out["tags"] = tags

    return out
