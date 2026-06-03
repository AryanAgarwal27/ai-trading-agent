"""Stage 1b smoke test — FastAPI lifespan + Postgres saver/store bootstrap.

Asserts:
1. Driving the app's lifespan context manager opens AsyncPostgresSaver
   and AsyncPostgresStore against the live Postgres on 127.0.0.1:5433,
   stashing them on app.state.
2. With the lifespan active, GET /health returns 200 + {"ok": true}.

Uses httpx.AsyncClient with ASGITransport per operator request. ASGI
transport itself does not fire lifespan events, so the test drives the
FastAPI router's lifespan_context manually.

NOTE (marker): this is an INTEGRATION test — driving the lifespan opens
real AsyncPostgresSaver/Store connections (``from_conn_string().setup()``)
against a live Postgres and starts the scheduler + kill subscription;
nothing is mocked. It lives under ``tests/unit/`` because BRD §13 Stage 1
named that path, but it is ``@pytest.mark.integration`` so the CI unit run
(``pytest -m "not integration" tests/unit``) deselects it — it requires
infra CI's unit job does not provide. Same precedent as the integration
test in ``tests/unit/test_events.py``.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from orchestrator.main import app


@pytest.mark.integration
@pytest.mark.asyncio
async def test_lifespan_initializes_saver_and_store_and_health_returns_ok() -> None:
    async with app.router.lifespan_context(app):
        assert app.state.saver is not None, "AsyncPostgresSaver was not stashed on app.state"
        assert app.state.store is not None, "AsyncPostgresStore was not stashed on app.state"

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"ok": True}
