"""Typed wrappers over PostgresStore for the BRD §5.9 namespaces.

The researcher agent (BRD §5.3) consults two long-term-Store partitions
before proposing a new hypothesis:

  - ``("failures", regime)`` — past strategies that lost money or
    failed a gate in this regime. Lets the researcher avoid repeating
    a known mistake.
  - ``("wins", regime)`` — past strategies that completed a live cycle
    profitably in this regime. Lets the researcher anchor a new
    proposal to a working pattern.

These helpers are pure wrappers over the LangGraph Store API and accept
any ``BaseStore`` instance — unit tests pass an ``InMemoryStore``;
production wires the ``AsyncPostgresStore`` opened in the FastAPI
``lifespan`` (BRD §6.5). Keeping the API ``BaseStore``-typed (not
PostgresStore-typed) means tests don't need a live Postgres.

The @tool wrappers that bind these to a ContextVar-held store live in
:mod:`orchestrator.agents.researcher`, alongside the agent that uses them.
"""

from __future__ import annotations

from typing import Any

from langgraph.store.base import BaseStore

# Default item count. The researcher's prompt budget caps practical use
# well below 50; 10 is enough to surface representative cases without
# burying the agent in repetitive failures from the same archived stretch.
DEFAULT_LIMIT = 10


async def aget_failures(
    store: BaseStore,
    regime: str,
    *,
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """Fetch up to ``limit`` past failures in ``regime`` (BRD §5.9).

    Returns a list of plain dicts (key + value spread) so the @tool
    wrapper can JSON-serialize them straight to the agent without
    leaking ``Item`` objects across the boundary.
    """
    items = await store.asearch(("failures", regime), limit=limit)
    return [{"key": item.key, **(item.value or {})} for item in items]


async def aget_wins(
    store: BaseStore,
    regime: str,
    *,
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """Fetch up to ``limit`` past wins in ``regime`` (BRD §5.9)."""
    items = await store.asearch(("wins", regime), limit=limit)
    return [{"key": item.key, **(item.value or {})} for item in items]


async def aput_failure(
    store: BaseStore,
    *,
    regime: str,
    strategy_id: str,
    payload: dict[str, Any],
) -> None:
    """Persist a failure record. Called from any ``archive`` node that
    has a regime tag in context (BRD §5.9 writer column)."""
    await store.aput(("failures", regime), strategy_id, payload)


async def aput_win(
    store: BaseStore,
    *,
    regime: str,
    strategy_id: str,
    payload: dict[str, Any],
) -> None:
    """Persist a win record. Called from the post-live archive node when
    live metrics were positive (BRD §5.9 writer column)."""
    await store.aput(("wins", regime), strategy_id, payload)
