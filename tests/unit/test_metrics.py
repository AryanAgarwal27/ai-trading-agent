"""Unit tests for the Prometheus /metrics endpoint (Stage 10e, BRD §14).

No DB / no lifespan: ``GET /metrics`` only serializes the process-global
prometheus_client default registry, so it serves without ``app.state``.
"""

from __future__ import annotations

from httpx import ASGITransport, AsyncClient

from orchestrator.main import app
from orchestrator.observability import metrics


async def _scrape() -> tuple[int, str, str]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics")
    return resp.status_code, resp.headers.get("content-type", ""), resp.text


def _metric_value(text: str, needle: str) -> float:
    """Return the value of the exposition line that is exactly ``needle <value>``."""
    for line in text.splitlines():
        if line.startswith(needle + " "):
            return float(line.rsplit(" ", 1)[1])
    raise AssertionError(f"{needle!r} not found in metrics output")


async def test_metrics_endpoint_returns_prometheus_text() -> None:
    status_code, content_type, text = await _scrape()
    assert status_code == 200
    # Prometheus exposition content type (text/plain; version=0.0.4).
    assert "text/plain" in content_type
    assert "version=" in content_type
    # The three deliberate metrics are present (HELP/TYPE lines register at import).
    assert "ait_kill_switch_fires_total" in text
    assert "ait_strategies_by_stage" in text
    assert "ait_supervisor_runs_total" in text


async def test_supervisor_run_counter_reflects_a_change() -> None:
    needle = 'ait_supervisor_runs_total{trigger="manual"}'

    metrics.SUPERVISOR_RUNS.labels(trigger="manual").inc()
    _, _, text1 = await _scrape()
    before = _metric_value(text1, needle)

    metrics.SUPERVISOR_RUNS.labels(trigger="manual").inc()
    _, _, text2 = await _scrape()
    after = _metric_value(text2, needle)

    assert after == before + 1.0


async def test_strategies_by_stage_gauge_reflects_set() -> None:
    metrics.set_strategies_by_stage({"research": 3, "live": 1})
    _, _, text = await _scrape()

    assert _metric_value(text, 'ait_strategies_by_stage{stage="research"}') == 3.0
    assert _metric_value(text, 'ait_strategies_by_stage{stage="live"}') == 1.0
    # A stage absent from the map is set to 0 (not left stale).
    assert _metric_value(text, 'ait_strategies_by_stage{stage="paper"}') == 0.0
