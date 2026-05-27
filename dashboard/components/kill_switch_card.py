"""Kill-switch panel — distinct red rendering for live_pause_review
when the path discriminator is ``kill_switch``.

The kill switch fires OUT-OF-BAND of the LangGraph thread (BRD §5.6):
APScheduler polls Freqtrade every 5 minutes and calls ``/api/v1/stop``
directly when global drawdown ≥ 12% or consecutive losses ≥ 10. There
is NO coordinator vote preceding it — by the time the operator sees
this panel, the strategy is already stopped on the exchange side.

SPEC §4.1 requires the dashboard to visually flag this gate as
kill-switch-originated (distinct colour / icon) so the operator does
not confuse it with a coordinator-driven pause. We use Streamlit's
``st.error`` for the red banner + a bordered details container.

Stage 6g scope: render-only. The resume-after-fix / archive action
UI lands in Stage 8 with the live subgraph; for now the operator
acknowledges via the Back-to-threads button and the thread remains
parked at its interrupt until Stage 8 lands.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_kill_switch_card(thread: dict[str, Any], payload: dict[str, Any]) -> None:
    """Render the kill-switch variant of live_pause_review.

    Expects ``payload`` to be the
    :func:`orchestrator.gates.hitl.build_interrupt_payload` output for
    ``kind="live_pause_review"`` with
    ``summary.path == "kill_switch"`` — i.e. ``summary.kill_switch_event``
    populated and ``summary.coordinator is None``.
    """
    summary = payload.get("summary") or {}
    event: dict[str, Any] = summary.get("kill_switch_event") or {}
    metrics: dict[str, Any] = summary.get("metrics") or {}

    # ── Red banner — the SPEC §4.1 "distinct colour" requirement.
    # st.error gives Streamlit's red alert styling and AppTest exposes
    # it as ``at.error[...]`` so the test can assert on the variant
    # without parsing CSS.
    st.error(
        "🛑 **KILL SWITCH FIRED** — this thread was paused out-of-band by the "
        "APScheduler kill-switch job (BRD §5.6), NOT by a coordinator vote. "
        "The Freqtrade instance has already been stopped on the exchange side."
    )

    # Top bar — match the other cards' shape but with the kill-switch
    # subheader so the operator sees this is a different surface.
    bar_left, bar_right = st.columns([5, 1])
    with bar_left:
        st.subheader(f"live_pause_review · `{thread['strategy_id']}` · kill-switch path")
        st.caption(f"thread_id: `{thread['thread_id']}` · stage: `{thread.get('stage', '—')}`")
    with bar_right:
        if st.button("← Threads", key=f"back_kill_{thread['thread_id']}", use_container_width=True):
            st.session_state.pop("selected_thread_id", None)
            st.rerun()

    st.divider()

    # ── Bordered details container — the SPEC §4.1 "structured event
    # row replaces the rationale block" requirement.
    with st.container(border=True):
        st.markdown("### Event")
        col_reason, col_fired = st.columns([3, 2])
        with col_reason:
            st.markdown(f"**Reason:** `{event.get('reason', '_unknown_')}`")
            st.markdown(f"**Action taken:** `{event.get('action_taken', '_unknown_')}`")
        with col_fired:
            st.markdown(f"**Fired at:** `{event.get('fired_at', '_unknown_')}`")

        st.markdown("### Metrics snapshot at fire time")
        event_metrics = event.get("metrics") or {}
        if event_metrics:
            st.json(event_metrics)
        else:
            st.caption("—  (no metrics snapshot in payload)")

    # ── Secondary surface: drawdown trajectory + recent trades.
    with st.expander("Drawdown trajectory + recent trades around fire time", expanded=False):
        col_dd, col_trades = st.columns(2)
        with col_dd:
            st.markdown("**Drawdown trajectory**")
            dd = metrics.get("drawdown")
            if dd:
                st.json(dd)
            else:
                st.caption("—  (no drawdown trajectory in payload)")
        with col_trades:
            st.markdown("**Recent trades around fired_at**")
            trades = metrics.get("recent_trades")
            if trades:
                st.json(trades)
            else:
                st.caption("—  (no recent trades in payload)")

    # No approve/reject form here — kill-switch resolution
    # (resume-after-fix vs archive) is Stage 8 work. For 6g, operator
    # acknowledges by clicking Back; the thread remains parked.
    st.caption(
        "Stage 8 will add `Resume` / `Archive` actions for kill-switch paused "
        "threads. For now, this view is acknowledge-only."
    )
