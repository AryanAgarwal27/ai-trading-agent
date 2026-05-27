"""Stage 6f Streamlit dashboard — threads list + paper_gate card.

SPEC §4.1 layout for the paper_gate card:

    [strategy header + back button]

    ### Rationale          (markdown-rendered, pinned, prominent)
    {risk_analyst.rationale}

    verdict · confidence · primary_concern  (chip row)

    ▾ Metrics             (collapsible — backtest + robustness)

    Notes: [____________]
    [Approve]  [Reject]

Data model: this dashboard polls ONE endpoint — ``GET /threads`` —
which embeds the interrupt payload (Stage 6f endpoint extension). The
operator's approve/reject calls ``POST /threads/{tid}/approve`` with
the ``X-Operator-Token`` header (SPEC §6 — token rotation = .env edit
+ restart of both FastAPI and Streamlit).

Autorefresh is conditional: ENABLED on the list view (so a new HITL
event appears within ``POLL_MS`` without a manual refresh) and
DISABLED on the card view (so the operator's in-progress ``notes``
text is not reset every poll).

Launch::

    .venv/Scripts/streamlit run dashboard/app.py --server.address 127.0.0.1
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

load_dotenv()

API_BASE_URL: str = os.environ.get("DASHBOARD_API_BASE_URL", "http://127.0.0.1:8000")
OPERATOR_TOKEN: str = os.environ.get("OPERATOR_TOKEN", "")
POLL_MS: int = 5000


# ─── HTTP client wrappers ──────────────────────────────────────────────


def _get_threads() -> list[dict[str, Any]]:
    """Poll ``GET /threads``. On failure, render an inline error and
    return an empty list — the dashboard stays usable, just empty."""
    try:
        with httpx.Client(base_url=API_BASE_URL, timeout=5.0) as client:
            r = client.get("/threads")
            r.raise_for_status()
            return list(r.json())
    except httpx.HTTPError as exc:
        st.error(f"GET /threads failed at {API_BASE_URL}: {exc}")
        return []


def _approve(thread_id: str, *, approved: bool, notes: str) -> dict[str, Any] | None:
    """Call ``POST /threads/{tid}/approve``. Returns the JSON body on
    success, ``None`` on any failure (with an inline error rendered)."""
    if not OPERATOR_TOKEN:
        st.error(
            "OPERATOR_TOKEN is empty. Set it in .env (matching the FastAPI "
            "server's value) and restart Streamlit."
        )
        return None
    try:
        with httpx.Client(base_url=API_BASE_URL, timeout=15.0) as client:
            r = client.post(
                f"/threads/{thread_id}/approve",
                json={"approved": approved, "notes": notes},
                headers={"X-Operator-Token": OPERATOR_TOKEN},
            )
            if r.status_code != 200:
                st.error(f"POST /approve → {r.status_code}: {r.text}")
                return None
            return dict(r.json())
    except httpx.HTTPError as exc:
        st.error(f"POST /approve failed: {exc}")
        return None


# ─── Threads list ──────────────────────────────────────────────────────


def render_threads_list() -> None:
    """Render the strategies-in-registry table.

    Threads with a pending HITL interrupt sort to the top — that is the
    operator's primary task. Idle / archived threads follow.
    """
    st.subheader("Strategy threads")

    threads = _get_threads()
    if not threads:
        st.info(
            "No strategies in the registry yet. Seed one via the orchestrator "
            "or run `scripts/midstage_seed.py` (deleted; see 6e notes)."
        )
        return

    threads_sorted = sorted(
        threads,
        key=lambda t: (not t.get("has_pending_interrupt"), t.get("strategy_id") or ""),
    )

    # Header row.
    header = st.columns([3, 2, 3, 2, 1])
    header[0].markdown("**strategy_id**")
    header[1].markdown("**stage**")
    header[2].markdown("**last updated**")
    header[3].markdown("**HITL**")
    header[4].markdown("")
    st.divider()

    for t in threads_sorted:
        cols = st.columns([3, 2, 3, 2, 1])
        cols[0].markdown(f"`{t['strategy_id']}`")
        cols[1].markdown(f"`{t.get('stage', '—')}`")
        cols[2].caption(t.get("last_updated") or "—")
        if t.get("has_pending_interrupt"):
            kind = (t.get("pending_interrupt_payload") or {}).get("kind", "?")
            cols[3].markdown(f":orange[**pending** · `{kind}`]")
            if cols[4].button("Review", key=f"review_{t['thread_id']}"):
                st.session_state["selected_thread_id"] = t["thread_id"]
                st.rerun()
        else:
            cols[3].caption("—")
            cols[4].caption("")


# ─── paper_gate card (SPEC §4.1) ───────────────────────────────────────


def render_paper_gate_card(thread: dict[str, Any]) -> None:
    """Render the paper_gate HITL card per SPEC §4.1.

    The thread dict carries the embedded ``pending_interrupt_payload``
    (Stage 6f endpoint extension), so this function does no extra HTTP
    calls — the parent loop already fetched the latest state.
    """
    payload = thread.get("pending_interrupt_payload") or {}
    summary = payload.get("summary") or {}
    risk = summary.get("risk_analyst") or {}
    metrics = summary.get("metrics") or {}

    # Top bar.
    bar_left, bar_right = st.columns([5, 1])
    with bar_left:
        st.subheader(f"paper_gate · `{thread['strategy_id']}`")
        st.caption(f"thread_id: `{thread['thread_id']}` · stage: `{thread.get('stage', '—')}`")
    with bar_right:
        if st.button("← Threads", key="back_to_list", use_container_width=True):
            st.session_state.pop("selected_thread_id", None)
            st.rerun()

    st.divider()

    # ── Rationale block (SPEC §4.1: large, full-width, markdown, pinned).
    st.markdown("### Rationale")
    rationale = (risk.get("rationale") or "").strip()
    if rationale:
        with st.container(border=True):
            st.markdown(rationale)
    else:
        st.warning(
            "No `risk_analyst.rationale` in the interrupt payload. "
            "Approve/reject without rationale is a bug upstream — flag it."
        )

    # ── Vote / confidence chips (below the rationale, per SPEC §4.1).
    chip_bits: list[str] = ["**risk_analyst**"]
    if "decision" in risk:
        chip_bits.append(f"verdict `{risk['decision']}`")
    confidence = risk.get("confidence")
    if isinstance(confidence, int | float):
        chip_bits.append(f"confidence `{confidence:.2f}`")
    if risk.get("primary_concern"):
        chip_bits.append(f"primary concern: _{risk['primary_concern']}_")
    st.markdown(" · ".join(chip_bits))

    # ── Metrics (collapsible, secondary surface).
    with st.expander("Metrics (backtest + robustness)", expanded=False):
        col_bt, col_rb = st.columns(2)
        with col_bt:
            st.markdown("**Backtest**")
            backtest = metrics.get("backtest")
            if backtest:
                st.json(backtest)
            else:
                st.caption("—  (no backtest summary in payload)")
        with col_rb:
            st.markdown("**Robustness**")
            robustness = metrics.get("robustness")
            if robustness:
                st.json(robustness)
            else:
                st.caption("—  (no robustness summary in payload)")

    # ── Approve / Reject form.
    st.divider()
    notes = st.text_area(
        "Notes (recorded in `gate_audits.payload`)",
        key=f"notes_{thread['thread_id']}",
        height=90,
        placeholder="Optional. Why are you approving / rejecting?",
    )

    btn_approve, btn_reject, _spacer = st.columns([1, 1, 3])
    with btn_approve:
        if st.button("✅ Approve", key=f"approve_{thread['thread_id']}", type="primary"):
            result = _approve(thread["thread_id"], approved=True, notes=notes)
            if result is not None:
                st.success(
                    f"Approved. next_stage={result.get('next_stage')!r} · "
                    f"audit_id={result.get('audit_id')}"
                )
                st.session_state.pop("selected_thread_id", None)
                st.rerun()
    with btn_reject:
        if st.button("❌ Reject", key=f"reject_{thread['thread_id']}"):
            result = _approve(thread["thread_id"], approved=False, notes=notes)
            if result is not None:
                st.info(
                    f"Rejected. next_stage={result.get('next_stage')!r} · "
                    f"audit_id={result.get('audit_id')}"
                )
                st.session_state.pop("selected_thread_id", None)
                st.rerun()


# ─── Placeholder for unsupported gate kinds (Stage 7 / 8) ───────────────


def render_unsupported_card(thread: dict[str, Any]) -> None:
    payload = thread.get("pending_interrupt_payload") or {}
    kind = payload.get("kind", "unknown")
    st.warning(
        f"Card for gate kind `{kind}` is not implemented yet. "
        f"Stage 7 lands `live_gate`; Stage 8 lands `live_pause_review`. "
        f"Raw payload below."
    )
    st.json(payload)
    if st.button("← Threads", key="back_unsupported"):
        st.session_state.pop("selected_thread_id", None)
        st.rerun()


# ─── Entry point ───────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="ai-trading-agent", layout="wide")
    st.title("ai-trading-agent — operator dashboard")

    selected_tid: str | None = st.session_state.get("selected_thread_id")

    if selected_tid is None:
        # List view — autorefresh ON so new HITL events surface within
        # POLL_MS without operator action.
        st_autorefresh(interval=POLL_MS, key="poll_threads_list")
        render_threads_list()
        return

    # Card view — autorefresh OFF (would reset the notes textarea on
    # every poll). Re-fetch /threads once per render so the selected
    # thread's payload is fresh, and to detect "thread already
    # advanced" (someone else approved it).
    threads = _get_threads()
    current = next(
        (t for t in threads if t.get("thread_id") == selected_tid),
        None,
    )
    if current is None or not current.get("has_pending_interrupt"):
        st.info(
            "This thread is no longer parked at an interrupt — it may have been "
            "advanced by another operator tab or via the API directly. Returning "
            "to the threads list."
        )
        st.session_state.pop("selected_thread_id", None)
        if st.button("← Threads"):
            st.rerun()
        return

    kind = (current.get("pending_interrupt_payload") or {}).get("kind", "")
    if kind == "paper_gate":
        render_paper_gate_card(current)
    else:
        render_unsupported_card(current)


main()
