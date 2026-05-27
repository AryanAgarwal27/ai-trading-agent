"""Stage 6f smoke tests for the Streamlit dashboard.

Two layers:

1. **Import smoke** — ``import dashboard.app`` runs the top-level code
   (Streamlit's bare-mode warning is captured by the harness). Catches
   syntax/typo regressions cheaply.

2. **AppTest render** — uses ``streamlit.testing.v1.AppTest`` with a
   mocked ``httpx.Client`` to confirm the threads-list and paper_gate
   card views render the SPEC §4.1 elements without exercising the
   real FastAPI server.

Streamlit's testing API does not let us reach into a card's text-area
post-rerun to drive the full approve loop deterministically — the
real approve path is covered by 6d's
``test_approve_with_valid_token_advances_thread_and_writes_audit_and_publishes``.
Here we verify the dashboard glue: data → layout.
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Any
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.integration


# ─── Fake httpx ─────────────────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, data: Any, status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code
        self.text = ""

    def json(self) -> Any:
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                f"{self.status_code}", request=None, response=None  # type: ignore[arg-type]
            )


class _FakeHttpxClient:
    """Context-manager stand-in for ``httpx.Client``.

    The ``threads_response`` class attribute is set per-test to the
    data the dashboard's ``_get_threads`` call will receive.
    """

    threads_response: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> _FakeHttpxClient:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def get(self, path: str) -> _FakeResponse:
        if path == "/threads":
            return _FakeResponse(type(self).threads_response)
        return _FakeResponse(None, status_code=404)

    def post(self, path: str, **kwargs: Any) -> _FakeResponse:  # noqa: ARG002
        return _FakeResponse({"resumed": True, "next_stage": "paper", "audit_id": 1})


# ─── 1. Import smoke ────────────────────────────────────────────────────


def test_dashboard_module_imports_cleanly() -> None:
    """Catch syntax errors / missing imports / typos cheaply.

    Importing the file runs Streamlit in bare mode and emits a
    ScriptRunContext warning — that's expected and harmless here.
    """
    with warnings.catch_warnings(), contextlib.suppress(Exception):
        warnings.simplefilter("ignore")
        # If the module already imported in this pytest session, the
        # cached import is fine — we just want to know it doesn't blow up.
        import dashboard.app  # noqa: F401


# ─── 2. AppTest — threads list empty state ──────────────────────────────


def test_dashboard_threads_list_renders_empty_state() -> None:
    """No threads in the registry → the "No strategies in the registry
    yet" info message renders. Confirms the list view path works."""
    from streamlit.testing.v1 import AppTest

    _FakeHttpxClient.threads_response = []
    with patch("httpx.Client", _FakeHttpxClient):
        at = AppTest.from_file("dashboard/app.py", default_timeout=10)
        at.run()

    assert not at.exception, f"app raised: {at.exception}"
    info_texts = [el.value for el in at.info]
    assert any("No strategies in the registry yet" in t for t in info_texts), (
        f"empty-state info message missing; saw info elements: {info_texts!r}"
    )


# ─── 3. AppTest — paper_gate card layout ────────────────────────────────


def test_dashboard_paper_gate_card_renders_spec_4_1_layout() -> None:
    """With one thread parked at paper_gate, after selecting it the
    card view renders the SPEC §4.1 layout: rationale heading, the
    rationale text, the verdict/confidence chip line, and the Metrics
    expander. Approve / Reject buttons are present.
    """
    from streamlit.testing.v1 import AppTest

    paper_gate_thread = {
        "strategy_id": "sid_dash_smoke",
        "thread_id": "thread_sid_dash_smoke",
        "stage": "paper_gate",
        "last_updated": "2026-05-27T15:00:00Z",
        "has_pending_interrupt": True,
        "pending_interrupt_payload": {
            "kind": "paper_gate",
            "strategy_id": "sid_dash_smoke",
            "summary": {
                "risk_analyst": {
                    "decision": "approve",
                    "primary_concern": "fee-stress sensitivity at 3x",
                    "rationale": (
                        "Sharpe holds above 1.2 OOS across all 6 folds; "
                        "fee-stress 3x degradation 52% (under 60% cap)."
                    ),
                    "confidence": 0.78,
                },
                "metrics": {
                    "backtest": {"sharpe_is": 1.91, "oos_ratio": 0.71, "max_dd": 0.14},
                    "robustness": {"score": 0.78, "passed": True},
                },
            },
        },
    }
    _FakeHttpxClient.threads_response = [paper_gate_thread]

    with patch("httpx.Client", _FakeHttpxClient):
        at = AppTest.from_file("dashboard/app.py", default_timeout=10)
        at.run()
        assert not at.exception, f"first render raised: {at.exception}"

        # Pre-selection: the list should render with a Review button.
        # AppTest exposes buttons keyed by their label.
        review_buttons = [b for b in at.button if b.label == "Review"]
        assert review_buttons, "expected a Review button on the threads list"
        review_buttons[0].click()
        at.run()
        assert not at.exception, f"after-click render raised: {at.exception}"

        # ── SPEC §4.1 elements:
        # 1. Rationale heading.
        markdowns = [el.value for el in at.markdown]
        assert any(m.startswith("### Rationale") for m in markdowns), (
            f"missing '### Rationale' header; saw markdown: {markdowns!r}"
        )
        # 2. The rationale text is rendered somewhere on the page.
        rationale_text = "fee-stress 3x degradation 52%"
        page_text = "\n".join(markdowns)
        assert rationale_text in page_text, (
            f"rationale text not found in markdown blocks: {markdowns!r}"
        )
        # 3. Chip row — verdict + confidence both surfaced.
        assert any(
            "verdict `approve`" in m and "confidence `0.78`" in m for m in markdowns
        ), f"chip row missing verdict/confidence; markdown was: {markdowns!r}"
        # 4. Metrics expander present.
        expander_labels = [exp.label for exp in at.expander]
        assert any(
            "Metrics" in label for label in expander_labels
        ), f"Metrics expander missing; expanders: {expander_labels!r}"

    # ── Approve + Reject buttons present.
    button_labels = [b.label for b in at.button]
    assert any("Approve" in lbl for lbl in button_labels), (
        f"Approve button missing; buttons: {button_labels!r}"
    )
    assert any("Reject" in lbl for lbl in button_labels), (
        f"Reject button missing; buttons: {button_labels!r}"
    )


# ─── 4. AppTest — live_gate card (Stage 6g scaffold) ───────────────────


def test_live_gate_card_renders() -> None:
    """live_gate payload routes to render_live_gate_card with the
    paper_monitor rationale as the primary surface (SPEC §4.1).

    Stage 6g scaffolds against synthetic payloads — Stage 7 lands the
    real paper_monitor agent that populates this shape."""
    from streamlit.testing.v1 import AppTest

    live_gate_thread = {
        "strategy_id": "sid_live_gate_smoke",
        "thread_id": "thread_sid_live_gate_smoke",
        "stage": "live_gate",
        "last_updated": "2026-05-27T16:00:00Z",
        "has_pending_interrupt": True,
        "pending_interrupt_payload": {
            "kind": "live_gate",
            "strategy_id": "sid_live_gate_smoke",
            "summary": {
                "paper_monitor": {
                    "decision": "advance",
                    "primary_concern": "Live Sharpe diverges slightly from paper",
                    "rationale": (
                        "Paper Sharpe 1.62 vs backtest 1.81 — KS p-value 0.21, "
                        "within tolerance. 30-day window completed with 87 trades."
                    ),
                    "confidence": 0.71,
                },
                "metrics": {
                    "paper": {
                        "sharpe": 1.62,
                        "sharpe_deviation_pct": 0.105,
                        "ks_pvalue": 0.21,
                        "n_trades": 87,
                    },
                },
            },
        },
    }
    _FakeHttpxClient.threads_response = [live_gate_thread]

    with patch("httpx.Client", _FakeHttpxClient):
        at = AppTest.from_file("dashboard/app.py", default_timeout=10)
        at.run()
        assert not at.exception, f"first render raised: {at.exception}"

        review_buttons = [b for b in at.button if b.label == "Review"]
        assert review_buttons, "expected a Review button on the threads list"
        review_buttons[0].click()
        at.run()
        assert not at.exception, f"after-click render raised: {at.exception}"

        markdowns = [el.value for el in at.markdown]

        # Rationale heading + content present.
        assert any(m.startswith("### Rationale") for m in markdowns), (
            f"missing '### Rationale' header; markdown: {markdowns!r}"
        )
        page_text = "\n".join(markdowns)
        assert "KS p-value 0.21" in page_text, (
            f"paper_monitor rationale text missing; markdown: {markdowns!r}"
        )
        # Chip row identifies the source as paper_monitor (NOT
        # risk_analyst) — that's the live_gate-vs-paper_gate distinction.
        assert any(
            "paper_monitor" in m and "verdict `advance`" in m and "confidence `0.71`" in m
            for m in markdowns
        ), f"paper_monitor chip row missing; markdown: {markdowns!r}"
        # No risk_analyst chip — would be a wrong-renderer bug.
        assert not any(
            "**risk_analyst**" in m for m in markdowns
        ), "live_gate card should NOT render a risk_analyst chip"

    # Approve/Reject buttons present (live_gate is an actionable gate).
    button_labels = [b.label for b in at.button]
    assert any("Approve" in lbl for lbl in button_labels)
    assert any("Reject" in lbl for lbl in button_labels)


# ─── 5. AppTest — live_pause_review coordinator path ───────────────────


def test_live_pause_review_coordinator_path_renders() -> None:
    """live_pause_review payload with ``summary.path == "coordinator"``
    routes to the coordinator branch of render_live_pause_review_card.

    Renders the coordinator rationale + a row of three reviewer-vote
    chips (risk_check, performance_check, regime_check). The kill-
    switch path's red banner must be ABSENT."""
    from streamlit.testing.v1 import AppTest

    coord_thread = {
        "strategy_id": "sid_pause_coord_smoke",
        "thread_id": "thread_sid_pause_coord_smoke",
        "stage": "live",
        "last_updated": "2026-05-27T17:00:00Z",
        "has_pending_interrupt": True,
        "pending_interrupt_payload": {
            "kind": "live_pause_review",
            "strategy_id": "sid_pause_coord_smoke",
            "summary": {
                "path": "coordinator",
                "coordinator": {
                    "verdict": "pause",
                    "rationale": (
                        "Performance drift exceeds threshold: live Sharpe -0.4 "
                        "after 8 days while paper baselined at 1.6. "
                        "Recommend operator review."
                    ),
                    "confidence": 0.82,
                },
                "reviewer_votes": {
                    "risk_check": {"verdict": "continue", "confidence": 0.6},
                    "performance_check": {"verdict": "pause", "confidence": 0.9},
                    "regime_check": {"verdict": "continue", "confidence": 0.55},
                },
                "metrics": {
                    "current_drawdown": 0.08,
                    "daily_pnl": -0.024,
                    "consecutive_losses": 3,
                    "regime_delta": "low_vol_flat → high_vol_down",
                },
            },
        },
    }
    _FakeHttpxClient.threads_response = [coord_thread]

    with patch("httpx.Client", _FakeHttpxClient):
        at = AppTest.from_file("dashboard/app.py", default_timeout=10)
        at.run()
        assert not at.exception

        [b for b in at.button if b.label == "Review"][0].click()
        at.run()
        assert not at.exception, f"after-click render raised: {at.exception}"

        markdowns = [el.value for el in at.markdown]
        page_text = "\n".join(markdowns)

        # Coordinator rationale + chip.
        assert any(m.startswith("### Rationale") for m in markdowns)
        assert "Performance drift exceeds threshold" in page_text
        assert any(
            "**coordinator**" in m and "verdict `pause`" in m for m in markdowns
        ), f"coordinator chip missing; markdown: {markdowns!r}"

        # Reviewer-vote sub-heading + each reviewer's chip.
        assert any("#### Reviewer votes" in m for m in markdowns), (
            f"reviewer votes subheading missing; markdown: {markdowns!r}"
        )
        assert any("risk_check" in m and "continue" in m for m in markdowns)
        assert any("performance_check" in m and "pause" in m for m in markdowns)
        assert any("regime_check" in m and "continue" in m for m in markdowns)

        # Kill-switch banner must NOT appear on the coordinator path.
        error_texts = [el.value for el in at.error]
        assert not any(
            "KILL SWITCH FIRED" in t for t in error_texts
        ), f"coordinator path leaked a kill-switch banner; errors: {error_texts!r}"


# ─── 6. AppTest — live_pause_review kill-switch path ───────────────────


def test_live_pause_review_kill_switch_path_renders() -> None:
    """live_pause_review payload with ``summary.path == "kill_switch"``
    routes to render_kill_switch_card.

    Asserts:
      - red "KILL SWITCH FIRED" banner is present (the SPEC §4.1
        distinct-colour requirement, exposed via at.error in AppTest);
      - kill_switch_event fields surface (reason, action_taken, fired_at);
      - coordinator rationale section is ABSENT (the discriminator
        worked — coordinator-vote UI must not bleed into the kill-switch
        path).
    """
    from streamlit.testing.v1 import AppTest

    kill_switch_thread = {
        "strategy_id": "sid_kill_smoke",
        "thread_id": "thread_sid_kill_smoke",
        "stage": "live",
        "last_updated": "2026-05-27T17:30:00Z",
        "has_pending_interrupt": True,
        "pending_interrupt_payload": {
            "kind": "live_pause_review",
            "strategy_id": "sid_kill_smoke",
            "summary": {
                "path": "kill_switch",
                "kill_switch_event": {
                    "reason": "drawdown_12pct_exceeded",
                    "metrics": {"drawdown": 0.131, "consecutive_losses": 4},
                    "action_taken": "POST /api/v1/stop",
                    "fired_at": "2026-05-27T14:32:11Z",
                },
                "coordinator": None,
                "reviewer_votes": None,
                "metrics": {
                    "drawdown": [0.04, 0.07, 0.09, 0.11, 0.131],
                    "recent_trades": [
                        {"pair": "BTC/USDT", "pnl": -42.10},
                        {"pair": "ETH/USDT", "pnl": -18.50},
                    ],
                },
            },
        },
    }
    _FakeHttpxClient.threads_response = [kill_switch_thread]

    with patch("httpx.Client", _FakeHttpxClient):
        at = AppTest.from_file("dashboard/app.py", default_timeout=10)
        at.run()
        assert not at.exception

        [b for b in at.button if b.label == "Review"][0].click()
        at.run()
        assert not at.exception, f"after-click render raised: {at.exception}"

        # ── Red banner present (SPEC §4.1 distinct-colour requirement).
        error_texts = [el.value for el in at.error]
        assert any(
            "KILL SWITCH FIRED" in t for t in error_texts
        ), f"red kill-switch banner missing; errors: {error_texts!r}"

        # ── Event fields surface in the bordered details container.
        markdowns = [el.value for el in at.markdown]
        page_text = "\n".join(markdowns)
        assert "drawdown_12pct_exceeded" in page_text, (
            f"kill_switch_event.reason missing from rendered markdown: {markdowns!r}"
        )
        assert "POST /api/v1/stop" in page_text, (
            f"kill_switch_event.action_taken missing: {markdowns!r}"
        )
        assert "2026-05-27T14:32:11Z" in page_text, (
            f"kill_switch_event.fired_at missing: {markdowns!r}"
        )

        # ── No coordinator chip / reviewer-votes block (path
        # discriminator correctly routed past the coordinator branch).
        assert not any(
            "**coordinator**" in m for m in markdowns
        ), "coordinator chip leaked into kill-switch path"
        assert not any(
            "#### Reviewer votes" in m for m in markdowns
        ), "reviewer-votes section leaked into kill-switch path"

    # ── No Approve / Reject buttons — kill-switch is acknowledge-only
    # in Stage 6g. (Stage 8 adds Resume / Archive actions.)
    button_labels = [b.label for b in at.button]
    assert not any(
        "Approve" in lbl for lbl in button_labels
    ), f"kill-switch path should NOT render Approve; buttons: {button_labels!r}"
    assert not any(
        "Reject" in lbl for lbl in button_labels
    ), f"kill-switch path should NOT render Reject; buttons: {button_labels!r}"
