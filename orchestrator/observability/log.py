"""Structured logging substrate (Stage 10c, BRD §14).

BRD §14 requires that *every node* emit
``{strategy_id, thread_id, node, event, payload}`` as JSON to stdout.
Stage 10c adds a fifth binding — ``run_id`` — and wires the whole thing
on structlog.

Three public pieces:

- :func:`configure_logging` — installs the structlog processor chain.
  The prod default renders **JSON to stdout**; setting ``AIT_LOG_CONSOLE``
  swaps in structlog's human ``ConsoleRenderer`` for local dev. Call once
  at process startup (FastAPI lifespan / scheduler boot) before any node
  logs.

- :func:`run_context` — a context manager that mints a **fresh
  ``run_id``** and binds ``{run_id, strategy_id, thread_id}`` into
  structlog contextvars for its scope, unbinding on exit.

  ``run_id`` scope is **NESTED / execution-scoped** (operator decision,
  Stage 10c): one fresh id per *graph execution* — the supervisor spawn
  invoke, each 6h wake-resume, each HITL/kill resume — NOT per strategy
  lifecycle. It is therefore **execution-scoped state**, bound only here
  at the execution-entry boundary and cleared on exit. It is a plain
  ``str`` (so Stage 10d can mirror it as the LangSmith run key) and is
  **never** a field on :class:`orchestrator.state.StrategyState` — if it
  were checkpointed it would survive across wakes and silently become
  outermost-scoped (BRD §5.7 HARD CONSTRAINT).

- :func:`get_logger` — returns a logger pre-bound with its ``node`` name,
  so a node calls ``get_logger("paper_monitor").info("wake", payload=...)``
  and the contextvars (``run_id`` / ``strategy_id`` / ``thread_id``) are
  merged in automatically by the processor chain.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import structlog
from structlog.typing import FilteringBoundLogger, Processor

_TRUTHY = {"1", "true", "yes", "on"}


def configure_logging() -> None:
    """Install the structlog processor chain (idempotent; call at startup).

    Processor chain: merge contextvars (injects ``run_id`` /
    ``strategy_id`` / ``thread_id``) → add log level → ISO timestamp →
    renderer. The renderer is :class:`structlog.processors.JSONRenderer`
    by default (prod: one JSON object per line to stdout); when
    ``AIT_LOG_CONSOLE`` is truthy it is the human
    :class:`structlog.dev.ConsoleRenderer` (``colors=False`` so it stays
    dependency-free and clean in piped dev logs).
    """
    console = os.environ.get("AIT_LOG_CONSOLE", "").strip().lower() in _TRUTHY
    renderer: Processor = (
        structlog.dev.ConsoleRenderer(colors=False)
        if console
        else structlog.processors.JSONRenderer()
    )
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        renderer,
    ]
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(node: str) -> FilteringBoundLogger:
    """Return a logger pre-bound with ``node=<node>`` (BRD §14).

    The remaining contract keys — ``run_id`` / ``strategy_id`` /
    ``thread_id`` — are supplied by :func:`run_context` via contextvars
    and merged in by the processor chain; ``event`` and ``payload`` are
    passed at the call site (``log.info(event, payload=...)``).
    """
    logger: FilteringBoundLogger = structlog.get_logger()
    return logger.bind(node=node)


@contextmanager
def run_context(*, strategy_id: str, thread_id: str) -> Iterator[str]:
    """Bind a fresh execution-scoped ``run_id`` for the duration of a graph
    execution, yielding the id and clearing all three bindings on exit.

    Wrap every execution-entry / resume site (supervisor spawn invoke,
    ``/wake`` resume, ``/approve`` resume, kill-subscription direct
    resume) so each execution mints exactly one ``run_id``. Because the
    binding is reset on ``__exit__``, the id cannot leak into the next
    execution — preserving the "one fresh id per execution, never across
    wakes" invariant.
    """
    run_id = uuid.uuid4().hex
    tokens = structlog.contextvars.bind_contextvars(
        run_id=run_id,
        strategy_id=strategy_id,
        thread_id=thread_id,
    )
    try:
        yield run_id
    finally:
        structlog.contextvars.reset_contextvars(**tokens)
