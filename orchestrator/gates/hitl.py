"""HITL primitives — interrupt payload builder + autoresume helper (Stage 6b).

Two pure functions and one decision type. No FastAPI / Streamlit imports —
those land in 6c (resume endpoint) and 6d (dashboard). Keeping this module
dependency-light lets the gate nodes (which import from here) stay testable
without the web layer.

Contract:
- :func:`build_interrupt_payload` assembles the dict passed to
  ``langgraph.types.interrupt(...)`` inside a gate node. The shape is the
  SPEC §4.1 dashboard contract — rationale source as the primary surface,
  metrics as the secondary surface — so the dashboard renderer can rely on
  a stable layout across all three HITL gates.
- :func:`autoresume_for_test` is the test helper used by unit + integration
  tests to drive a parked thread past an ``interrupt()`` without spinning
  up FastAPI. It refuses to resume a thread that is NOT parked at an
  interrupt — that's the most common test bug (asserting a "resume worked"
  when the graph already ran to END).

BRD §6.2 mandates the dynamic ``interrupt()`` form; this module is the
canonical place to assemble the payload it carries.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict, cast

from langgraph.types import Command

# ─── Decision types ─────────────────────────────────────────────────────


class ApprovalDecision(TypedDict):
    """The ``Command(resume=...)`` payload for ``paper_gate`` / ``live_gate``.

    Both approval gates carry the same shape: a boolean and free-form
    operator notes. ``live_pause_review`` will need a richer
    ``PauseDecision`` (resume-after-fix vs archive vs hold) when the live
    subgraph lands in Stage 8 — not added here to avoid prematurely
    fixing a shape we haven't tested end-to-end.
    """

    approved: bool
    notes: str


# ─── Interrupt payload kinds ────────────────────────────────────────────

InterruptKind = Literal["paper_gate", "live_gate", "live_pause_review"]


# ─── build_interrupt_payload ────────────────────────────────────────────


def build_interrupt_payload(
    state: dict[str, Any],
    kind: InterruptKind,
) -> dict[str, Any]:
    """Assemble the dict passed to ``interrupt(...)`` inside a gate node.

    Shape (stable contract for the dashboard renderer):

    ``{"kind": <kind>, "strategy_id": ..., "summary": {...}}``

    The ``summary`` sub-dict's keys depend on ``kind`` (see below). Within
    a kind, the shape is invariant across Stage 6 → Stage 11 even as
    upstream nodes populate more fields — None means "node that produces
    this hasn't run yet" (e.g. ``paper_monitor`` rationale is None until
    Stage 7 lands the paper subgraph).

    Parameters
    ----------
    state
        The current ``StrategyState`` (dict form — gate nodes receive
        the state mapping, not a Pydantic instance).
    kind
        Which gate is interrupting. Selects the rationale source per
        SPEC §4.1.

    Returns
    -------
    Payload dict for ``interrupt(...)``. Never mutates ``state``.

    Notes
    -----
    SPEC §4.1 layout (rationale prominent, metrics secondary) is the
    dashboard's contract. The keys below match the table in that
    section.

    For ``live_pause_review``: if
    ``state["artifacts"].get("kill_switch_event")`` is set, the payload
    takes the kill-switch path (no coordinator rationale, per BRD §5.6
    out-of-band kill switch). Otherwise the coordinator path is taken.
    """
    gate_decisions: dict[str, Any] = state.get("gate_decisions") or {}
    artifacts: dict[str, Any] = state.get("artifacts") or {}

    base: dict[str, Any] = {
        "kind": kind,
        "strategy_id": state.get("strategy_id"),
        "summary": {},
    }
    summary: dict[str, Any] = base["summary"]

    if kind == "paper_gate":
        # SPEC §4.1: risk_analyst (Opus 4.7) rationale is the primary
        # surface. Metrics (backtest IS/OOS, robustness) are secondary.
        summary["risk_analyst"] = gate_decisions.get("risk_analyst")
        summary["metrics"] = {
            "backtest": gate_decisions.get("backtest"),
            "robustness": gate_decisions.get("robustness"),
        }
        return base

    if kind == "live_gate":
        # SPEC §4.1: paper_monitor (Haiku 4.5) rationale is the primary
        # surface. paper_monitor is produced by Stage 7 — None here.
        summary["paper_monitor"] = gate_decisions.get("paper_monitor")
        summary["metrics"] = {
            "paper": gate_decisions.get("paper"),
        }
        return base

    # kind == "live_pause_review"
    kill_switch_event = artifacts.get("kill_switch_event")
    if kill_switch_event is not None:
        # SPEC §4.1 kill-switch path: row from kill_switch_events table
        # replaces the rationale block. Coordinator did not vote.
        summary["path"] = "kill_switch"
        summary["kill_switch_event"] = kill_switch_event
        summary["coordinator"] = None
        summary["reviewer_votes"] = None
        summary["metrics"] = {
            "drawdown": artifacts.get("drawdown_trajectory"),
            "recent_trades": artifacts.get("recent_trades"),
        }
        return base

    # Coordinator path — produced by Stage 8.
    summary["path"] = "coordinator"
    summary["coordinator"] = gate_decisions.get("coordinator")
    summary["reviewer_votes"] = {
        "risk_check": gate_decisions.get("risk_check"),
        "performance_check": gate_decisions.get("performance_check"),
        "regime_check": gate_decisions.get("regime_check"),
    }
    summary["metrics"] = {
        "current_drawdown": artifacts.get("current_drawdown"),
        "daily_pnl": artifacts.get("daily_pnl"),
        "consecutive_losses": artifacts.get("consecutive_losses"),
        "regime_delta": artifacts.get("regime_delta"),
    }
    return base


# ─── autoresume_for_test ────────────────────────────────────────────────


async def autoresume_for_test(
    graph: Any,
    thread_id: str,
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Resume an interrupted thread; assert it actually IS interrupted first.

    Test helper. Not for production: the real resume path is the
    FastAPI ``POST /threads/{tid}/approve`` handler (Stage 6c).

    Why the pre-check matters: a thread that ran to END (no interrupt
    landed) silently accepts ``Command(resume=...)`` as a no-op
    continuation. Tests that skip the check pass for the wrong reason —
    they assert the *current* state, not the *resumed* state. This helper
    refuses to proceed unless ``aget_state`` shows at least one
    interrupted task.

    Parameters
    ----------
    graph
        Compiled LangGraph graph (any object exposing ``aget_state`` and
        ``astream`` — kept ``Any`` so this works for the per-strategy
        graph and small test graphs alike).
    thread_id
        The ``configurable.thread_id`` the graph was invoked with.
    decision
        Payload passed to ``Command(resume=...)``. For ``paper_gate`` /
        ``live_gate`` this is an :class:`ApprovalDecision`-shaped dict.

    Returns
    -------
    The final event emitted by ``astream`` after resume. Tests inspect
    ``state`` / ``stage`` from this.

    Raises
    ------
    AssertionError
        If ``aget_state`` shows no parked interrupt on the thread.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await graph.aget_state(config)

    interrupted = any(getattr(task, "interrupts", ()) for task in snapshot.tasks)
    if not interrupted:
        raise AssertionError(
            f"thread {thread_id!r} is not parked at an interrupt — "
            f"aget_state().tasks has no .interrupts. next={snapshot.next!r}, "
            f"refusing to resume (a Command(resume=...) on a non-interrupted "
            f"thread is a silent no-op and would mask the test bug)."
        )

    last_event: dict[str, Any] = {}
    async for ev in graph.astream(Command(resume=decision), config=config):
        last_event = cast(dict[str, Any], ev)
    return last_event
