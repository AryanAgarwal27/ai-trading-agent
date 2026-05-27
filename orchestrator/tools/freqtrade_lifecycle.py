"""Stage 7 paper-container lifecycle helpers.

These are the orchestrator's side of the Freqtrade paper-trading boundary.
``paper_spawn`` (the LangGraph node that lands in 7c) calls
:func:`spawn_paper_container` on entry and :func:`stop_paper_container` on
teardown. The third public helper, :func:`next_free_paper_port`, is a pure
function the spawn node uses to pick a host port that doesn't collide with
other live paper threads.

Why this is a separate module from :mod:`orchestrator.tools.backtest_runner`:

* ``backtest_runner`` drives short-lived subprocess Freqtrade invocations
  (minutes) that produce a zip and exit. Per-worker userdirs are torn down
  after each run.
* This module manages long-lived Compose-orchestrated containers (days to
  weeks). The userdir at ``freqtrade/user_data/_workers/<strategy_id>``
  persists across wakes; the container is the unit of lifecycle.

The subprocess pattern (``asyncio.to_thread(subprocess.run, ...)``) is
shared between the two — see :mod:`orchestrator.tools.backtest_runner` and
SPEC §6 (2026-05-27 Stage 3c) for why ``create_subprocess_exec`` is unsafe
on Windows in this project.

Config substitution path (SPEC §6, 2026-05-27 Stage 7a): the ``${VAR}``
strings in ``freqtrade/user_data/configs/paper-base.json`` are resolved
Python-side here, at config-write time, NOT by Freqtrade's own env
substitution. The container env vars set by Compose are belt-and-braces
fallback only.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx

from orchestrator.tools.freqtrade_api import FreqtradeAPI, FreqtradeAPIError, FreqtradeCredentials

logger = logging.getLogger(__name__)


# ───────────────────────── paths + constants ─────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_BASE_CONFIG = REPO_ROOT / "freqtrade" / "user_data" / "configs" / "paper-base.json"
COMPOSE_FILE = REPO_ROOT / "docker-compose.freqtrade.yml"
WORKERS_ROOT = REPO_ROOT / "freqtrade" / "user_data" / "_workers"

# Port range for paper containers. 100 slots is comfortably above v1's
# ~5-active-strategies cap (BRD §12 cost-budget implication) and far
# enough from common dev-stack ports (5432 Postgres, 6379 Redis, 8000
# orchestrator, 8501 Streamlit, 8080-8090 typical app range).
PAPER_PORT_RANGE_START = 8100
PAPER_PORT_RANGE_END = 8200  # exclusive — yields 100 ports [8100, 8199]

# Ping budget. The freqtrade image's import + Postgres-trades-DB init +
# pair-cache warmup runs ~25-40s on a warm laptop; 120s is generous
# without being silly. Polled every 2s → up to 60 attempts.
_SPAWN_PING_TIMEOUT_S = 120.0
_SPAWN_PING_INTERVAL_S = 2.0

# Subprocess timeouts. compose-up is fast (the image is local; the
# container's slow path is its own startup, polled separately above);
# compose-down can drag while Freqtrade flushes its trades DB.
_COMPOSE_UP_TIMEOUT_S = 60
_COMPOSE_DOWN_TIMEOUT_S = 30

# Best-effort REST stop before forcing the container down. Short — if
# Freqtrade is hung, we don't want this to delay the compose-down.
_REST_STOP_TIMEOUT_S = 5.0


# ───────────────────────── exceptions ─────────────────────────


class PaperLifecycleError(RuntimeError):
    """Base class for spawn/stop failures."""


class PaperSpawnError(PaperLifecycleError):
    """Raised when spawn cannot proceed (missing env, bad config, IO).

    Distinct from :class:`PaperSpawnTimeout` so the LangGraph paper_spawn
    node can route fail-fast errors to ``archive`` immediately, while
    timeout errors may warrant a retry on the next wake.
    """


class PaperSpawnTimeout(PaperLifecycleError):
    """Raised when /api/v1/ping doesn't return 200 within the budget."""


class PaperStopError(PaperLifecycleError):
    """Raised when ``docker compose down`` fails.

    A REST-stop failure is NOT this — it's logged as a warning and the
    compose-down still runs. This class is reserved for the case where
    we cannot guarantee the container is gone, which is an operational
    concern (orphaned process holding a port, leaking dry-run state).
    """


# ───────────────────────── pure helpers ─────────────────────────


def next_free_paper_port(used_ports: list[int]) -> int:
    """Return the lowest port in [8100, 8200) not present in ``used_ports``.

    Pure function — the caller (``paper_spawn`` node) queries
    ``strategy_registry`` for ports already assigned to active paper
    threads and passes the list in. Keeping the port allocator out of the
    DB layer keeps it trivially testable.

    Raises :class:`RuntimeError` if every port in the range is taken.
    This is not expected — v1 caps active strategies at ~5 — but the
    error is the right behaviour: silently reusing a port would land a
    new container on top of a live one, which Docker would refuse and
    we'd waste a wake debugging an opaque compose-up failure.
    """
    used = set(used_ports)
    for port in range(PAPER_PORT_RANGE_START, PAPER_PORT_RANGE_END):
        if port not in used:
            return port
    raise RuntimeError(
        f"all {PAPER_PORT_RANGE_END - PAPER_PORT_RANGE_START} paper ports "
        f"in [{PAPER_PORT_RANGE_START}, {PAPER_PORT_RANGE_END}) are in use; "
        "v1 capacity exceeded"
    )


# ───────────────────────── config loading + substitution ─────────────────────────


# Canonical JSONC line-comment regex — duplicated from freqtrade/README.md's
# documented pattern. Block comments are not used in this project.
_JSONC_LINE_COMMENT = re.compile(r"//.*")


def _load_paper_base_config(path: Path = PAPER_BASE_CONFIG) -> dict[str, Any]:
    """Load and parse the JSONC paper-base config.

    Strips ``//`` line comments per the convention documented in
    freqtrade/README.md. Block comments are not used; if the file ever
    starts using them this function will need updating in lockstep.
    """
    raw = path.read_text(encoding="utf-8")
    stripped = _JSONC_LINE_COMMENT.sub("", raw)
    return dict(json.loads(stripped))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a deep copy of ``base``.

    Dict-typed values merge key-by-key; everything else (lists, scalars)
    is replaced wholesale. List-replacement is the right semantics for
    Freqtrade's ``pair_whitelist`` and ``protections`` — per-strategy
    overrides describe the *complete* desired list, not a delta.
    """
    out = deepcopy(base)
    for key, val in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(val, dict)
        ):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _substitute_placeholders(text: str, mapping: dict[str, str]) -> str:
    """Replace every ``${KEY}`` in ``text`` with ``mapping[KEY]``.

    Operates on the serialized JSON string (post-merge) rather than
    walking the dict, because placeholders may appear inside any string
    value at any nesting depth. Unknown ``${...}`` placeholders are left
    intact — a future config addition won't silently blank out.
    """
    for key, value in mapping.items():
        text = text.replace(f"${{{key}}}", value)
    return text


def _required_env(name: str) -> str:
    """Read an env var or raise a :class:`PaperSpawnError` pointing at setup docs."""
    val = os.environ.get(name)
    if not val:
        raise PaperSpawnError(
            f"required env var {name} is not set; "
            "see SPEC §1 Q1 and .env.example (Stage 7a) for setup"
        )
    return val


def _strategy_class_name(strategy_module_path: Path) -> str:
    """Return the first ``IStrategy`` subclass name in the strategy module.

    Freqtrade's ``--config`` path requires ``"strategy"`` in the JSON to
    name the class to instantiate. We don't ask the caller to pass it
    separately because the module file is the authoritative source and
    a mismatched class name would surface as an opaque Freqtrade error
    later. AST-walk avoids importing the file (which would pull in
    Freqtrade's dependencies).
    """
    tree = ast.parse(strategy_module_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            # Match `class X(IStrategy)` or `class X(freqtrade.IStrategy)`.
            base_name = (
                base.id if isinstance(base, ast.Name)
                else base.attr if isinstance(base, ast.Attribute)
                else None
            )
            if base_name == "IStrategy":
                return node.name
    raise PaperSpawnError(
        f"no IStrategy subclass found in {strategy_module_path}; "
        "Freqtrade's --config strategy field cannot be filled"
    )


# ───────────────────────── spawn ─────────────────────────


async def spawn_paper_container(
    strategy_id: str,
    pair_whitelist: list[str],
    stake_amount: float,
    strategy_module_path: Path,
    port: int,
) -> str:
    """Boot one paper-trading Freqtrade container; return its API URL.

    Pipeline:

    1. Validate required env (``BINANCE_PAPER_API_KEY``,
       ``BINANCE_PAPER_API_SECRET``, ``PAPER_API_PASSWORD``). Fail fast
       with a clear error if any are missing.
    2. Load + JSONC-parse ``paper-base.json``.
    3. Deep-merge per-strategy overrides (whitelist, stake_amount,
       dry_run_wallet = 5 * stake_amount, bot_name, strategy class).
    4. Serialize and substitute ``${VAR}`` placeholders with real values.
    5. Write the resolved config to the per-strategy worker dir, copy
       the strategy module into ``strategies/``, and write the assigned
       port to a sidecar file (``stop_paper_container`` reads it back).
    6. ``docker compose -f docker-compose.freqtrade.yml -p paper-<sid>
       up -d`` with ``STRATEGY_ID`` / ``PAPER_PORT`` / ``PAPER_CONFIG_PATH``
       set in the subprocess env (Compose substitutes them into the
       container_name + mount paths + port mapping).
    7. Poll ``GET /api/v1/ping`` every 2s for up to 120s.

    Returns ``http://127.0.0.1:<port>`` on success. The orchestrator's
    ``StrategyState.freqtrade_api_url`` field is set from this return value.

    Raises:
        PaperSpawnError: missing env, bad config, IO failure.
        PaperSpawnTimeout: container did not respond to /ping within 120s.
    """
    # 1. Env validation — do this first so a misconfigured operator sees
    # the error before we write anything to disk.
    api_key = _required_env("BINANCE_PAPER_API_KEY")
    api_secret = _required_env("BINANCE_PAPER_API_SECRET")
    api_password = _required_env("PAPER_API_PASSWORD")

    # 2. Load base config (JSONC).
    base = _load_paper_base_config()

    # 3. Per-strategy overrides — deep-merged into the base. 5x dry-run
    # wallet leaves cushion for unrealized P&L drawdown without tripping
    # Freqtrade's stake-vs-wallet check.
    overrides: dict[str, Any] = {
        "stake_amount": stake_amount,
        "dry_run_wallet": stake_amount * 5,
        "bot_name": strategy_id,
        "strategy": _strategy_class_name(strategy_module_path),
        "exchange": {"pair_whitelist": pair_whitelist},
    }
    merged = _deep_merge(base, overrides)

    # 4. Serialize + substitute. STRATEGY_ID and PAPER_PORT are substituted
    # too even though deep-merge already filled bot_name — covers any
    # future placeholders the base config grows.
    serialized = json.dumps(merged, indent=4)
    resolved = _substitute_placeholders(
        serialized,
        {
            "BINANCE_PAPER_API_KEY": api_key,
            "BINANCE_PAPER_API_SECRET": api_secret,
            "PAPER_API_PASSWORD": api_password,
            "STRATEGY_ID": strategy_id,
            "PAPER_PORT": str(port),
        },
    )

    # 5. Disk: per-strategy worker dir + strategies/ subdir + sidecar port.
    worker_dir = WORKERS_ROOT / strategy_id
    strategies_dir = worker_dir / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)
    (worker_dir / "logs").mkdir(parents=True, exist_ok=True)

    config_path = worker_dir / "config-paper.json"
    config_path.write_text(resolved, encoding="utf-8")

    strategy_dest = strategies_dir / strategy_module_path.name
    shutil.copy2(strategy_module_path, strategy_dest)

    # Sidecar — stop_paper_container reads this to build the REST URL
    # without re-parsing the resolved config.
    (worker_dir / ".paper-port").write_text(str(port), encoding="utf-8")

    # 6. compose up. The compose file's ${PAPER_CONFIG_PATH} resolves
    # relative to the compose file's directory (the repo root), so we
    # pass a relative path from there.
    config_relpath = config_path.relative_to(REPO_ROOT).as_posix()
    compose_env = {
        **os.environ,
        "STRATEGY_ID": strategy_id,
        "PAPER_PORT": str(port),
        "PAPER_CONFIG_PATH": config_relpath,
    }
    project_name = f"paper-{strategy_id}"
    up_cmd = [
        "docker",
        "compose",
        "-f",
        str(COMPOSE_FILE),
        "-p",
        project_name,
        "up",
        "-d",
    ]
    stdout, stderr, returncode = await _run_subprocess(
        up_cmd, _COMPOSE_UP_TIMEOUT_S, env=compose_env
    )
    if returncode != 0:
        raise PaperSpawnError(
            f"docker compose up failed for strategy_id={strategy_id} "
            f"(exit {returncode}): stderr_tail={_tail(stderr)!r}"
        )

    # 7. Poll /ping. On timeout, dump container logs to aid diagnosis
    # before the caller archives the thread.
    base_url = f"http://127.0.0.1:{port}"
    container_name = f"ait-paper-{strategy_id}"

    deadline = asyncio.get_event_loop().time() + _SPAWN_PING_TIMEOUT_S
    last_error: str = ""
    async with httpx.AsyncClient(timeout=2.0) as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                resp = await client.get(f"{base_url}/api/v1/ping")
                if resp.status_code == 200:
                    logger.info(
                        "paper container ready strategy_id=%s url=%s",
                        strategy_id,
                        base_url,
                    )
                    return base_url
                last_error = f"status={resp.status_code}"
            except httpx.HTTPError as exc:
                last_error = f"transport={exc}"
            await asyncio.sleep(_SPAWN_PING_INTERVAL_S)

    # Timeout — dump logs for the operator.
    logs_cmd = ["docker", "logs", "--tail", "100", container_name]
    log_stdout, log_stderr, _rc = await _run_subprocess(
        logs_cmd, 10, env=os.environ
    )
    logger.error(
        "paper container failed to ping within %.0fs strategy_id=%s "
        "last_error=%s container_logs_tail=%r",
        _SPAWN_PING_TIMEOUT_S,
        strategy_id,
        last_error,
        _tail(log_stdout + b"\n" + log_stderr),
    )
    raise PaperSpawnTimeout(
        f"paper container {container_name} did not respond to /api/v1/ping "
        f"within {_SPAWN_PING_TIMEOUT_S:.0f}s (last_error={last_error}); "
        "see orchestrator logs for container output"
    )


# ───────────────────────── stop ─────────────────────────


async def stop_paper_container(strategy_id: str) -> None:
    """Graceful stop of a paper container.

    Order of operations:

    1. Read the sidecar port file written by ``spawn_paper_container``.
       If absent (spawn never completed, or worker dir was manually
       cleaned), skip the REST stop and go straight to compose down.
    2. ``POST /api/v1/stop`` with a 5-second timeout. Best-effort; if
       Freqtrade is hung we don't want this to delay the compose-down.
    3. ``docker compose -f docker-compose.freqtrade.yml -p paper-<sid>
       down --remove-orphans`` with a 30-second timeout.

    A REST-stop failure logs a warning and falls through. A compose-down
    failure raises :class:`PaperStopError` — the orphan container is a
    real operational concern (port held, dry-run state leaking).
    """
    worker_dir = WORKERS_ROOT / strategy_id
    port_sidecar = worker_dir / ".paper-port"

    # 1+2. Best-effort REST stop.
    if port_sidecar.exists():
        try:
            port = int(port_sidecar.read_text(encoding="utf-8").strip())
            api_password = os.environ.get("PAPER_API_PASSWORD", "")
            if api_password:
                creds = FreqtradeCredentials(
                    username="freqtrader", password=api_password
                )
                async with FreqtradeAPI(
                    base_url=f"http://127.0.0.1:{port}",
                    credentials=creds,
                    timeout_s=_REST_STOP_TIMEOUT_S,
                ) as client:
                    await client.stop()
                    logger.info(
                        "rest stop succeeded strategy_id=%s port=%d",
                        strategy_id,
                        port,
                    )
            else:
                logger.warning(
                    "PAPER_API_PASSWORD unset; skipping REST stop "
                    "strategy_id=%s and going straight to compose down",
                    strategy_id,
                )
        except (FreqtradeAPIError, ValueError, OSError) as exc:
            logger.warning(
                "rest stop failed strategy_id=%s exc=%s; "
                "falling through to compose down",
                strategy_id,
                exc,
            )
    else:
        logger.info(
            "no port sidecar found strategy_id=%s; skipping REST stop",
            strategy_id,
        )

    # 3. compose down — authoritative teardown.
    project_name = f"paper-{strategy_id}"
    down_cmd = [
        "docker",
        "compose",
        "-f",
        str(COMPOSE_FILE),
        "-p",
        project_name,
        "down",
        "--remove-orphans",
    ]
    stdout, stderr, returncode = await _run_subprocess(
        down_cmd, _COMPOSE_DOWN_TIMEOUT_S, env=os.environ
    )
    if returncode != 0:
        raise PaperStopError(
            f"docker compose down failed for strategy_id={strategy_id} "
            f"(exit {returncode}): stderr_tail={_tail(stderr)!r}"
        )
    logger.info("paper container removed strategy_id=%s", strategy_id)


# ───────────────────────── subprocess plumbing ─────────────────────────


async def _run_subprocess(
    cmd: list[str], timeout_s: int, *, env: dict[str, str] | os._Environ[str]
) -> tuple[bytes, bytes, int]:
    """Run ``cmd`` in a thread; capture stdout/stderr; enforce ``timeout_s``.

    Same Windows-event-loop reasoning as
    :func:`orchestrator.tools.backtest_runner._run_subprocess` — see
    SPEC §6 (2026-05-27 Stage 3c).
    """
    return await asyncio.to_thread(_run_subprocess_sync, cmd, timeout_s, env)


def _run_subprocess_sync(
    cmd: list[str], timeout_s: int, env: dict[str, str] | os._Environ[str]
) -> tuple[bytes, bytes, int]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout_s,
            check=False,
            env=dict(env),
        )
    except subprocess.TimeoutExpired as exc:
        return exc.stdout or b"", exc.stderr or b"timeout", -1
    return result.stdout, result.stderr, result.returncode


def _tail(data: bytes, max_bytes: int = 2000) -> str:
    """Return the last ``max_bytes`` of ``data`` decoded loosely for logs."""
    if not data:
        return ""
    return data[-max_bytes:].decode("utf-8", errors="replace")
