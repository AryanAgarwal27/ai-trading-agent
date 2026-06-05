"""Unit tests for :mod:`orchestrator.observability.log` (Stage 10c, BRD §14).

The Stage 10c structured-logging substrate must satisfy four contracts,
each pinned by a test here:

1. **JSON contract keys** — an emitted record is a single JSON object on
   stdout carrying the full Stage 10c key set
   ``{strategy_id, thread_id, run_id, node, event, payload}`` (BRD §14
   five-key contract + the execution-scoped ``run_id`` added by 10c).
2. **Distinct run_ids** — two entries of :func:`run_context` mint
   DISTINCT ``run_id`` values. ``run_id`` is execution-scoped (one per
   graph execution), so re-entering the helper must NOT reuse the id.
3. **run_id is NOT on StrategyState** — the HARD CONSTRAINT from the
   Stage 10c brief: ``run_id`` lives in structlog contextvars only, never
   as a checkpointed state field (BRD §5.7). If it were a state field it
   would survive across 6h wakes and silently become outermost-scoped.
4. **Clearing on exit** — :func:`run_context` unbinds ``run_id`` when its
   scope exits, so the id cannot leak into a later execution.

A fifth test pins the prod default: JSON renderer is the default; the
human ``ConsoleRenderer`` is gated behind ``AIT_LOG_CONSOLE``.
"""

from __future__ import annotations

import json

import pytest
import structlog

from orchestrator.observability import log as log_mod

_CONTRACT_KEYS = ("strategy_id", "thread_id", "run_id", "node", "event", "payload")


@pytest.fixture(autouse=True)
def _clean_contextvars() -> None:
    """Clear structlog contextvars before each test so a leaked binding
    from another test can never mask a real propagation bug."""
    structlog.contextvars.clear_contextvars()


# ─── (a) emitted record is JSON with the contract keys ─────────────────


def test_emitted_record_is_json_with_the_contract_keys(
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_mod.configure_logging()

    with log_mod.run_context(strategy_id="strategy_abc", thread_id="strategy_abc"):
        log_mod.get_logger("paper_monitor").info("wake", payload={"open_trades": 2})

    line = capsys.readouterr().out.strip().splitlines()[-1]
    record = json.loads(line)  # must be a single JSON object — raises otherwise

    for key in _CONTRACT_KEYS:
        assert key in record, f"missing contract key {key!r} in {record!r}"

    assert record["strategy_id"] == "strategy_abc"
    assert record["thread_id"] == "strategy_abc"
    assert record["node"] == "paper_monitor"
    assert record["event"] == "wake"
    assert record["payload"] == {"open_trades": 2}
    assert record["run_id"], "run_id must be a non-empty execution id"


# ─── (b) two entries mint DISTINCT run_ids ─────────────────────────────


def test_two_entries_of_the_helper_mint_distinct_run_ids() -> None:
    with log_mod.run_context(strategy_id="s", thread_id="t") as first:
        pass
    with log_mod.run_context(strategy_id="s", thread_id="t") as second:
        pass

    assert first != second, "each run_context entry must mint a fresh run_id"


# ─── (c) run_id is NOT a field on StrategyState (HARD CONSTRAINT) ───────


def test_run_id_is_not_a_field_on_strategy_state() -> None:
    from orchestrator.state import StrategyState

    assert "run_id" not in StrategyState.__annotations__, (
        "run_id is execution-scoped and MUST NOT be checkpointed on StrategyState "
        "(BRD §5.7) — it lives only in structlog contextvars"
    )


# ─── (d) clearing on exit ──────────────────────────────────────────────


def test_run_context_binds_inside_and_clears_run_id_on_exit() -> None:
    with log_mod.run_context(strategy_id="s", thread_id="t") as run_id:
        bound = structlog.contextvars.get_contextvars()
        assert bound.get("run_id") == run_id
        assert bound.get("strategy_id") == "s"
        assert bound.get("thread_id") == "t"

    after = structlog.contextvars.get_contextvars()
    assert "run_id" not in after, "run_id must not survive the run_context scope"
    assert "strategy_id" not in after
    assert "thread_id" not in after


# ─── (e) JSON is the prod default; console is env-gated ────────────────


def test_console_renderer_is_env_gated_and_json_is_the_default(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AIT_LOG_CONSOLE", "1")
    log_mod.configure_logging()

    log_mod.get_logger("n").info("hello", payload={})

    line = capsys.readouterr().out.strip().splitlines()[-1]
    with pytest.raises(json.JSONDecodeError):
        json.loads(line)  # console renderer output is NOT a JSON object
