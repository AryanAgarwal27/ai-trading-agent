"""Unit tests for small pure helpers in orchestrator.main.

``_find_stderr_tail`` is what makes a manual-inject backtest failure debuggable
from the orchestrator log: it digs the Freqtrade stderr out of a BacktestError
even when LangGraph re-raises it wrapped inside another exception.
"""

from __future__ import annotations

from orchestrator.main import _find_stderr_tail


class _BacktestLike(Exception):
    """Duck-typed stand-in for BacktestError (carries a stderr_tail attr)."""

    def __init__(self, message: str, stderr_tail: str = "") -> None:
        super().__init__(message)
        self.stderr_tail = stderr_tail


def test_find_stderr_tail_direct() -> None:
    exc = _BacktestLike("exited with code 2", "freqAI is not enabled.")
    assert _find_stderr_tail(exc) == "freqAI is not enabled."


def test_find_stderr_tail_through_cause_chain() -> None:
    """LangGraph can re-raise the worker error wrapped — the stderr tail must
    still be recovered from the __cause__."""
    inner = _BacktestLike("exited with code 2", "real freqtrade traceback tail")
    try:
        try:
            raise inner
        except _BacktestLike as e:
            raise RuntimeError("graph run crashed in a Send worker") from e
    except RuntimeError as outer:
        assert _find_stderr_tail(outer) == "real freqtrade traceback tail"


def test_find_stderr_tail_absent_returns_empty() -> None:
    assert _find_stderr_tail(RuntimeError("plain error, no stderr_tail")) == ""


def test_find_stderr_tail_ignores_blank_tail() -> None:
    assert _find_stderr_tail(_BacktestLike("x", "   ")) == ""
