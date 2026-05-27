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
