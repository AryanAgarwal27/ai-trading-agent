"""Prometheus metrics (Stage 10e, BRD §14).

A SMALL, deliberate metric set — three metrics, each wired to a real event
that already happens in the codebase, not a speculative dashboard. They
register on the prometheus_client DEFAULT registry, which the FastAPI
``GET /metrics`` endpoint serializes (BRD §14: "/metrics on the
orchestrator (FastAPI)").

Instrumented sites (wired to real work, not stubs):

- ``ait_kill_switch_fires_total`` (Counter) — incremented in
  :func:`orchestrator.kill_subscription._handle_message` once per well-formed
  out-of-band kill-switch event the orchestrator consumes (BRD §11, §5.6, §1.1
  rule 7). One increment = one kill fire that reached the graph; malformed /
  skipped messages do not count. No label: kill ``reason`` strings are
  free-form (unbounded cardinality), so they stay out of the label set.

- ``ait_strategies_by_stage`` (Gauge, label ``stage``) — set in
  :func:`orchestrator.supervisor.aget_portfolio_snapshot`, the function whose
  job IS to count ``strategy_registry`` rows by lifecycle stage (BRD §5.7).
  Chosen over ``sync_registry_stage`` (the operator's suggested site) because
  ``aget_portfolio_snapshot`` is the actual count-by-stage site and runs
  immediately after ``sync_registry_stage`` in every supervisor run (and on
  every spawn capacity check), so the gauge reflects the reconciled registry.

- ``ait_supervisor_runs_total`` (Counter, label ``trigger``) — incremented at
  the top of :func:`orchestrator.supervisor.run_supervisor`, once per run,
  labelled ``cron`` / ``event`` / ``manual`` (BRD §5.1, §13 row 9).

Cardinality is bounded by construction: ``stage`` ∈ the 7 BRD §5.7 lifecycle
stages, ``trigger`` ∈ {cron, event, manual}.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge

KILL_SWITCH_FIRES = Counter(
    "ait_kill_switch_fires_total",
    "Out-of-band kill-switch events consumed by the orchestrator (BRD §11, §5.6).",
)

STRATEGIES_BY_STAGE = Gauge(
    "ait_strategies_by_stage",
    "Count of strategy threads by lifecycle stage in strategy_registry (BRD §5.7).",
    ["stage"],
)

SUPERVISOR_RUNS = Counter(
    "ait_supervisor_runs_total",
    "Supervisor runs by trigger (BRD §5.1, §13 row 9).",
    ["trigger"],
)

# The 7 lifecycle stages (BRD §5.7). Used to set the gauge to 0 for stages with
# no threads so a drained stage reports 0 rather than a stale last-known value.
_STAGES: tuple[str, ...] = (
    "research",
    "validation",
    "paper_gate",
    "paper",
    "live_gate",
    "live",
    "archived",
)


def set_strategies_by_stage(by_stage: dict[str, int]) -> None:
    """Set the per-stage gauge from a ``{stage: count}`` map.

    Every known stage is set explicitly (to 0 when absent from ``by_stage``) so
    a stage that drains to empty reports ``0`` rather than its stale value. Any
    unexpected stage key is also surfaced rather than silently dropped.
    """
    for stage in _STAGES:
        STRATEGIES_BY_STAGE.labels(stage=stage).set(by_stage.get(stage, 0))
    for stage, count in by_stage.items():
        if stage not in _STAGES:
            STRATEGIES_BY_STAGE.labels(stage=stage).set(count)
