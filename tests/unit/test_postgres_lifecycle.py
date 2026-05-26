"""Stage 1b smoke test — FastAPI lifespan + Postgres saver/store bootstrap.

Asserts:
1. Driving the app's lifespan context manager opens AsyncPostgresSaver
   and AsyncPostgresStore against the live Postgres on 127.0.0.1:5433,
   stashing them on app.state.
2. With the lifespan active, GET /health returns 200 + {"ok": true}.

Uses httpx.AsyncClient with ASGITransport per operator request. ASGI
transport itself does not fire lifespan events, so the test drives the
FastAPI router's lifespan_context manually.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from orchestrator.main import app


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
