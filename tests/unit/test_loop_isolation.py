"""Stage 11a (D-15) regression guard — per-test event-loop isolation.

Pins the contract that ``tests/conftest.py``'s ``_isolate_module_task_registries``
fixture neutralises the cross-loop task leak that made D-13/D-14
collection-order-dependent. pytest-asyncio gives each test a fresh
function-scoped loop, but the module-level fire-and-forget task registries
(``kill_subscription._KILL_RESUME_TASKS``,
``supervisor._BACKGROUND_SPAWN_TASKS``) OUTLIVE that loop. Without the fixture, a
task left in a registry by one test — bound to its now-closed loop — is gathered
by a later test on a DIFFERENT loop, raising ``RuntimeError: ... got Future ...
attached to a different loop``; whether that happens depends on pytest
COLLECTION ORDER (the measurement-integrity bug).

The two tests below run in definition order (pytest executes intra-file tests
top-to-bottom): the first LEAKS an un-drained task into each registry on its
loop; the second asserts both registries are EMPTY at the start of its (fresh)
loop — i.e. the fixture drained the leak — and that gathering them raises
nothing. This guard is itself order-independent: the fixture clears the
registries after EVERY test, so the second test sees empty registries no matter
what ran before it. If someone removes the isolation fixture, the second test
fails loudly.
"""

from __future__ import annotations

import asyncio

import orchestrator.kill_subscription as kill_sub
import orchestrator.supervisor as supervisor


async def test_a_leaks_undrained_background_tasks() -> None:
    """Leak a pending fire-and-forget task into BOTH module-level registries and
    deliberately do NOT drain it — the isolation fixture must clean up."""

    async def _never() -> None:
        await asyncio.sleep(3600)

    kill_sub._KILL_RESUME_TASKS.add(asyncio.create_task(_never()))
    supervisor._BACKGROUND_SPAWN_TASKS.add(asyncio.create_task(_never()))


async def test_b_registries_are_clean_on_a_fresh_loop() -> None:
    """The previous test's leaked tasks must be gone, so nothing here is bound to
    a foreign (closed) loop — the cross-loop RuntimeError D-13 hit cannot fire."""
    assert kill_sub._KILL_RESUME_TASKS == set()
    assert supervisor._BACKGROUND_SPAWN_TASKS == set()
    # Empty gathers on THIS loop — proves no stale, foreign-loop future survives.
    await asyncio.gather(*list(kill_sub._KILL_RESUME_TASKS))
    await asyncio.gather(*list(supervisor._BACKGROUND_SPAWN_TASKS))
