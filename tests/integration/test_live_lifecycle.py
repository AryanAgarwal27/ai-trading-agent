"""Stage 8b tests — live-container spawn / stop lifecycle.

Mirrors :mod:`tests.integration.test_freqtrade_lifecycle` (paper) but for the
LIVE path. A separate module keeps the live-lifecycle tests cohesive and
parallel to the paper module rather than bloating a paper-focused file.

Test tiers:

* Pure unit tests (no markers, run every CI invocation): the live port
  allocator, the live/paper port-range non-overlap invariant, the config
  write (``prepare_live_worker`` — render + on-disk), and ``spawn_live_container``
  with its docker/ping calls stubbed out. None require Docker or Postgres.

* :func:`test_spawn_and_stop_live_against_real_container` — boots a REAL
  live-keyed Freqtrade container. Gated behind a DEDICATED opt-in flag
  ``AIT_RUN_REAL_LIVE_SPAWN_TESTS`` (NOT the paper ``AIT_RUN_REAL_SPAWN_TESTS``):
  a live container runs with ``dry_run: false`` and will place REAL orders the
  moment it finds a signal, so the operator's routine paper smoke must never
  boot it as a side effect. Marked ``integration`` + ``freqtrade``; skipped by
  default.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pytest

import orchestrator.tools.freqtrade_lifecycle as fl
from orchestrator.tools.freqtrade_lifecycle import (
    LIVE_PORT_RANGE_END,
    LIVE_PORT_RANGE_START,
    LIVE_WORKERS_ROOT,
    PAPER_PORT_RANGE_END,
    PAPER_PORT_RANGE_START,
    WORKERS_ROOT,
    LiveSpawnTimeout,
    next_free_live_port,
    prepare_live_worker,
    spawn_live_container,
    stop_live_container,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MEAN_REVERSION_TEMPLATE = REPO_ROOT / "strategy_templates" / "mean_reversion_template.py"


def _set_live_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_LIVE_API_KEY", "live-key-aaa")
    monkeypatch.setenv("BINANCE_LIVE_API_SECRET", "live-secret-aaa")
    monkeypatch.setenv("BINANCE_LIVE_API_PASSWORD", "live-rest-pw")


def _set_paper_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_PAPER_API_KEY", "paper-key-zzz")
    monkeypatch.setenv("BINANCE_PAPER_API_SECRET", "paper-secret-zzz")
    monkeypatch.setenv("PAPER_API_PASSWORD", "paper-rest-pw")


# ───────────────────────── unit: port allocator ─────────────────────────


def test_next_free_live_port_returns_first_unused() -> None:
    """Live allocator mirrors the paper one, over the live range."""
    assert next_free_live_port([]) == LIVE_PORT_RANGE_START  # 8200
    assert next_free_live_port([8200, 8201]) == 8202
    assert next_free_live_port([8201, 8200]) == 8202  # order-independent
    assert next_free_live_port([8200, 8202, 8203]) == 8201  # fills holes
    with pytest.raises(RuntimeError, match="v1 capacity exceeded"):
        next_free_live_port(list(range(LIVE_PORT_RANGE_START, LIVE_PORT_RANGE_END)))


def test_live_port_range_does_not_overlap_paper() -> None:
    """The live and paper port windows MUST be disjoint — a collision would
    land a live container on a port a paper container already holds."""
    assert LIVE_PORT_RANGE_START >= PAPER_PORT_RANGE_END, (
        f"live range [{LIVE_PORT_RANGE_START}, {LIVE_PORT_RANGE_END}) overlaps "
        f"paper range [{PAPER_PORT_RANGE_START}, {PAPER_PORT_RANGE_END})"
    )
    paper_ports = set(range(PAPER_PORT_RANGE_START, PAPER_PORT_RANGE_END))
    live_ports = set(range(LIVE_PORT_RANGE_START, LIVE_PORT_RANGE_END))
    assert paper_ports.isdisjoint(live_ports)


# ───────────────────────── unit: config write ───────────────────────────


def test_prepare_live_worker_writes_config_distinct_from_paper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The on-disk live config uses the live userdir, live port, and live keys
    — all distinct from paper's equivalents (BRD §1.1 rule 5 + userdir
    isolation, BRD §7.1)."""
    _set_live_creds(monkeypatch)
    _set_paper_creds(monkeypatch)

    strategy_id = f"live-unit-{uuid.uuid4().hex[:8]}"
    port = LIVE_PORT_RANGE_START  # 8200 — in the live range, not paper's
    worker_dir = LIVE_WORKERS_ROOT / strategy_id
    try:
        config_path = prepare_live_worker(
            strategy_id,
            ["BTC/USDT"],
            125.0,
            MEAN_REVERSION_TEMPLATE,
            port,
        )

        # userdir path: under _live_workers, never paper's _workers.
        assert LIVE_WORKERS_ROOT != WORKERS_ROOT
        assert config_path == worker_dir / "config-live.json"
        assert "_live_workers" in str(config_path)
        assert str(WORKERS_ROOT / strategy_id) not in str(config_path)

        # port: in the live range, recorded in the sidecar (not paper's).
        sidecar = worker_dir / ".live-port"
        assert sidecar.read_text().strip() == str(port)
        assert LIVE_PORT_RANGE_START <= port < LIVE_PORT_RANGE_END
        assert not (worker_dir / ".paper-port").exists()

        # key references: live key, never the paper key. dry_run hard false.
        cfg = json.loads(config_path.read_text())
        assert cfg["dry_run"] is False
        assert cfg["exchange"]["key"] == "live-key-aaa"
        assert cfg["exchange"]["key"] != "paper-key-zzz"
        assert cfg["exchange"]["secret"] == "live-secret-aaa"
        assert "${" not in json.dumps(cfg)  # all placeholders resolved
    finally:
        if worker_dir.exists():
            shutil.rmtree(worker_dir, ignore_errors=True)


# ───────────────────────── unit: spawn (docker stubbed) ─────────────────


async def test_spawn_live_container_returns_api_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """spawn_live_container returns the loopback api_url when the compose-up
    and health-check succeed (both stubbed — no Docker)."""
    _set_live_creds(monkeypatch)

    async def _stub_subprocess(*_a: Any, **_k: Any) -> tuple[bytes, bytes, int]:
        return (b"", b"", 0)  # compose-up success

    async def _stub_ping(*_a: Any, **_k: Any) -> None:
        return None  # health-check passes immediately

    monkeypatch.setattr(fl, "_run_subprocess", _stub_subprocess)
    monkeypatch.setattr(fl, "_await_live_ping", _stub_ping)

    strategy_id = f"live-spawn-{uuid.uuid4().hex[:8]}"
    port = 8299
    worker_dir = LIVE_WORKERS_ROOT / strategy_id
    try:
        api_url = await spawn_live_container(
            strategy_id,
            ["BTC/USDT"],
            125.0,
            MEAN_REVERSION_TEMPLATE,
            port,
        )
        assert api_url == f"http://127.0.0.1:{port}"
        # The resolved config was written before the (stubbed) compose-up.
        assert (worker_dir / "config-live.json").exists()
    finally:
        if worker_dir.exists():
            shutil.rmtree(worker_dir, ignore_errors=True)


# ───────────────────────── integration: real container ───────────────────


def _skip_if_missing_live_prereqs() -> None:
    """Skip unless the operator has DELIBERATELY opted into a real live boot.

    Dedicated flag ``AIT_RUN_REAL_LIVE_SPAWN_TESTS`` — distinct from the paper
    smoke flag ``AIT_RUN_REAL_SPAWN_TESTS``. A live container runs with
    ``dry_run: false`` and places REAL orders; the routine paper pre-flight
    (which sets ``AIT_RUN_REAL_SPAWN_TESTS=1``) must NEVER boot it as a side
    effect.
    """
    if not os.environ.get("AIT_RUN_REAL_LIVE_SPAWN_TESTS"):
        pytest.skip(
            "AIT_RUN_REAL_LIVE_SPAWN_TESTS not set; the real LIVE-container "
            "spawn test boots a dry_run:false container that places REAL "
            "orders. It requires its own explicit opt-in, separate from the "
            "paper AIT_RUN_REAL_SPAWN_TESTS smoke, so a paper pre-flight can "
            "never boot a live trading container by accident."
        )
    for var in ("BINANCE_LIVE_API_KEY", "BINANCE_LIVE_API_SECRET", "BINANCE_LIVE_API_PASSWORD"):
        if not os.environ.get(var):
            pytest.skip(
                f"{var} not set; live spawn-test requires real live creds (see .env.example)"
            )
    if shutil.which("docker") is None:
        pytest.skip("docker not on PATH; spawn-test requires Docker")
    inspect = subprocess.run(
        ["docker", "image", "inspect", "freqtradeorg/freqtrade:stable_freqai"],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspect.returncode != 0:
        pytest.skip("freqtradeorg/freqtrade:stable_freqai image not found locally")
    if not MEAN_REVERSION_TEMPLATE.exists():
        pytest.skip(f"strategy template missing at {MEAN_REVERSION_TEMPLATE}")


def _docker_container_exists(name: str) -> bool:
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return name in result.stdout.splitlines()


@pytest.mark.integration
@pytest.mark.freqtrade
async def test_spawn_and_stop_live_against_real_container() -> None:
    """End-to-end: spawn a REAL live container, ping it, stop, verify gone.

    DANGER: this boots a dry_run:false container with real live keys. Only
    runs under the dedicated AIT_RUN_REAL_LIVE_SPAWN_TESTS opt-in. Cleanup
    runs in ``finally`` so a mid-test failure doesn't leak a live container.
    """
    _skip_if_missing_live_prereqs()

    strategy_id = f"livesmoke-{uuid.uuid4().hex[:8]}"
    container_name = f"ait-live-{strategy_id}"
    port = LIVE_PORT_RANGE_END - 1  # 8299 — top of the live range
    worker_dir = LIVE_WORKERS_ROOT / strategy_id

    try:
        api_url = await spawn_live_container(
            strategy_id,
            ["BTC/USDT"],
            10.0,
            MEAN_REVERSION_TEMPLATE,
            port,
        )
        assert api_url == f"http://127.0.0.1:{port}"
        assert (worker_dir / "config-live.json").exists()
        assert (worker_dir / ".live-port").read_text().strip() == str(port)
        assert _docker_container_exists(
            container_name
        ), f"container {container_name} not found after spawn returned success"
    finally:
        try:
            await stop_live_container(strategy_id)
        except Exception as exc:  # noqa: BLE001
            # to_thread: raw subprocess.run would block the event loop (ASYNC221).
            await asyncio.to_thread(
                subprocess.run,
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                check=False,
            )
            pytest.fail(f"stop_live_container failed: {exc}")
        assert not _docker_container_exists(
            container_name
        ), f"container {container_name} still present after stop"
        if worker_dir.exists():
            shutil.rmtree(worker_dir, ignore_errors=True)


def test_live_spawn_timeout_is_exported() -> None:
    """LiveSpawnTimeout mirrors PaperSpawnTimeout — referenced by 8c's node."""
    assert issubclass(LiveSpawnTimeout, Exception)
