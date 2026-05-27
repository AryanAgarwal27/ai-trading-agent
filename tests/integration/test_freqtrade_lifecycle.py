"""Stage 7b integration test — paper-container spawn / stop lifecycle.

Two tests:

* :func:`test_spawn_and_stop_against_real_container` — boots a real
  Freqtrade paper container against BTC/USDT for ~30 seconds, asserts
  the API responds, then stops and verifies cleanup. Marked
  ``integration`` + ``freqtrade``; skipped by the default CI invocation
  ``pytest -m "not freqtrade"``.

* :func:`test_next_free_paper_port_returns_first_unused` — pure unit
  test for the port allocator. No markers; runs in every default
  invocation. Three cases per the Stage 7b handoff: empty list,
  contiguous-low used, full-range used.

The integration test skips gracefully if the operator hasn't completed
the Stage 7a env setup (no Binance paper subaccount keys yet). The
goal is that ``pytest -m "not freqtrade"`` always passes on a fresh
clone — the freqtrade-marked tests are opt-in for operators who have
configured a real paper subaccount.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from orchestrator.tools.freqtrade_lifecycle import (
    WORKERS_ROOT,
    next_free_paper_port,
    spawn_paper_container,
    stop_paper_container,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MEAN_REVERSION_TEMPLATE = REPO_ROOT / "strategy_templates" / "mean_reversion_template.py"


# ───────────────────────── unit test (no markers) ─────────────────────────


def test_next_free_paper_port_returns_first_unused() -> None:
    """Three cases for the pure port allocator.

    Not marked ``integration`` — runs in the default CI invocation and
    contributes to the baseline test count.
    """
    # Empty list → first port in the [8100, 8200) range.
    assert next_free_paper_port([]) == 8100

    # Lowest two taken → 8102 is the first free.
    assert next_free_paper_port([8100, 8101]) == 8102

    # Order shouldn't matter — allocator skips taken regardless of order.
    assert next_free_paper_port([8101, 8100]) == 8102

    # Holes are filled lowest-first.
    assert next_free_paper_port([8100, 8102, 8103]) == 8101

    # Full range → RuntimeError. Use list(range(...)) to materialize.
    with pytest.raises(RuntimeError, match="v1 capacity exceeded"):
        next_free_paper_port(list(range(8100, 8200)))


# ───────────────────────── integration test ─────────────────────────


def _skip_if_missing_prereqs() -> None:
    """Skip the integration test unless every prerequisite is in place.

    Default outcome on a fresh clone with no Binance paper subaccount is
    *skip*, not fail. The error string for each branch points the
    operator at the exact setup doc.
    """
    if not os.environ.get("BINANCE_PAPER_API_KEY"):
        pytest.skip(
            "BINANCE_PAPER_API_KEY not set; spawn-test requires real paper "
            "API credentials. See SPEC §1 Q1 and stage 7a's .env.example."
        )
    if not os.environ.get("BINANCE_PAPER_API_SECRET"):
        pytest.skip("BINANCE_PAPER_API_SECRET not set; see stage 7a .env.example")
    if not os.environ.get("PAPER_API_PASSWORD"):
        pytest.skip(
            "PAPER_API_PASSWORD not set; generate via "
            "`python -c \"import secrets; print(secrets.token_urlsafe(24))\"` "
            "and add to .env (see stage 7a .env.example)"
        )

    if shutil.which("docker") is None:
        pytest.skip("docker not on PATH; spawn-test requires Docker")

    # Image presence — avoid an unannounced ~1.5 GB pull mid-test.
    inspect = subprocess.run(
        ["docker", "image", "inspect", "freqtradeorg/freqtrade:stable_freqai"],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspect.returncode != 0:
        pytest.skip(
            "freqtradeorg/freqtrade:stable_freqai image not found locally; "
            "run `docker pull freqtradeorg/freqtrade:stable_freqai` first"
        )

    if not MEAN_REVERSION_TEMPLATE.exists():
        pytest.skip(f"strategy template missing at {MEAN_REVERSION_TEMPLATE}")


def _docker_container_exists(name: str) -> bool:
    """Return True if a container with ``name`` is present (any state)."""
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return name in result.stdout.splitlines()


@pytest.mark.integration
@pytest.mark.freqtrade
async def test_spawn_and_stop_against_real_container() -> None:
    """End-to-end: spawn a paper container, ping it, stop, verify gone.

    Uses a UUID-suffixed strategy_id to avoid collisions with any
    operator-managed paper containers that might be running. Port is
    picked from the high end of the range for the same reason. Cleanup
    runs in ``finally`` so a mid-test assertion failure doesn't leak
    an orphan container.
    """
    _skip_if_missing_prereqs()

    strategy_id = f"smoke-{uuid.uuid4().hex[:8]}"
    container_name = f"ait-paper-{strategy_id}"
    # 8199 is the top of the paper-port range — lowest chance of clashing
    # with a real paper thread that picked from the bottom.
    port = 8199

    worker_dir = WORKERS_ROOT / strategy_id

    try:
        # Spawn — returns when /api/v1/ping has 200'd, raises on timeout.
        api_url = await spawn_paper_container(
            strategy_id=strategy_id,
            pair_whitelist=["BTC/USDT"],
            stake_amount=10.0,
            strategy_module_path=MEAN_REVERSION_TEMPLATE,
            port=port,
        )

        assert api_url == f"http://127.0.0.1:{port}"

        # Config file written at the expected location.
        config_path = worker_dir / "config-paper.json"
        assert config_path.exists(), f"resolved config not at {config_path}"

        # Sidecar port file for stop_paper_container.
        port_file = worker_dir / ".paper-port"
        assert port_file.exists() and port_file.read_text().strip() == str(port)

        # Strategy module copied into the worker dir's strategies/.
        copied_strategy = worker_dir / "strategies" / MEAN_REVERSION_TEMPLATE.name
        assert copied_strategy.exists()

        # Container present.
        assert _docker_container_exists(container_name), (
            f"container {container_name} not found after spawn returned success"
        )

    finally:
        # Cleanup — always run, even if the spawn asserts failed.
        try:
            await stop_paper_container(strategy_id)
        except Exception as exc:  # noqa: BLE001 — log and proceed
            # Last-ditch: force-remove the container if the helper failed.
            # Wrapped in to_thread because the surrounding test is async;
            # a raw subprocess.run would block the event loop (ruff ASYNC221).
            await asyncio.to_thread(
                subprocess.run,
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                check=False,
            )
            pytest.fail(f"stop_paper_container failed: {exc}")

        # Container gone.
        assert not _docker_container_exists(container_name), (
            f"container {container_name} still present after stop"
        )

        # Worker dir cleanup — not strictly required by the helper contract
        # (the resolved config + sqlite trades DB might be useful for
        # operator forensics in production), but keep test-scoped runs
        # tidy. The check above already passed, so the container is gone.
        if worker_dir.exists():
            shutil.rmtree(worker_dir, ignore_errors=True)
