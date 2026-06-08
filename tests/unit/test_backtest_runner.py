"""Unit tests for orchestrator.tools.backtest_runner.

No Docker / no network: the ``_run_subprocess`` seam is monkeypatched to return
a synthetic ``(stdout, stderr, returncode)`` triple, and ``WORKERS_DIR`` is
redirected to ``tmp_path`` so worker dirs land in pytest's scratch space.

Covers:
- BacktestError surfaces the stderr tail in its MESSAGE (debuggability fix —
  str(exc) is what the logged failure_reason / manual-inject _drive() shows).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.tools import backtest_runner as br

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MEAN_REVERSION = REPO_ROOT / "strategy_templates" / "mean_reversion_template.py"


async def test_backtest_error_message_includes_stderr_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-zero Freqtrade exit must put the real stderr error INTO the
    BacktestError message — not just an unread attribute. Before the fix the
    message was only ``exited with code 2`` and the cause was swallowed."""
    sentinel = "freqAI is not enabled. Please enable it in your config to use this strategy."

    async def _fake_run(cmd: list[str], timeout_s: int) -> tuple[bytes, bytes, int]:
        return (b"backtest progress 100%", f"2026-... freqtrade - ERROR - {sentinel}".encode(), 2)

    monkeypatch.setattr(br, "WORKERS_DIR", tmp_path / "_workers")
    monkeypatch.setattr(br, "_run_subprocess", _fake_run)

    with pytest.raises(br.BacktestError) as excinfo:
        await br.run_backtest(
            MEAN_REVERSION,
            pairs=["BTC/USDT"],
            timeframe="5m",
            timerange="20250101-20250108",
        )

    exc = excinfo.value
    assert exc.returncode == 2
    # The load-bearing assertion: str(exc) now carries Freqtrade's real error,
    # so the logged failure_reason is debuggable.
    assert sentinel in str(exc), f"stderr cause missing from message: {exc}"
    assert "exited with code 2" in str(exc)
    # Attribute still populated for any structured consumer.
    assert sentinel in exc.stderr_tail


async def test_backtest_error_message_falls_back_to_stdout_when_stderr_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When Freqtrade prints the error to stdout and stderr is empty, the
    message falls back to the stdout tail rather than showing nothing."""
    sentinel = "OperationalException: some failure on stdout"

    async def _fake_run(cmd: list[str], timeout_s: int) -> tuple[bytes, bytes, int]:
        return (sentinel.encode(), b"", 2)

    monkeypatch.setattr(br, "WORKERS_DIR", tmp_path / "_workers")
    monkeypatch.setattr(br, "_run_subprocess", _fake_run)

    with pytest.raises(br.BacktestError) as excinfo:
        await br.run_backtest(
            MEAN_REVERSION,
            pairs=["BTC/USDT"],
            timeframe="5m",
            timerange="20250101-20250108",
        )

    assert sentinel in str(excinfo.value)
