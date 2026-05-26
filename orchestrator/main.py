"""FastAPI app for the ai-trading-agent orchestrator.

Stage 1b scope: bring up the FastAPI lifespan, open the LangGraph
`AsyncPostgresSaver` and `AsyncPostgresStore` as async context managers,
call `setup()` on both (BRD §6.5), stash them on `app.state`, and expose
a single `GET /health` endpoint.

Connection URIs are read from the environment:
- `LANGGRAPH_CHECKPOINT_URI` → AsyncPostgresSaver
- `LANGGRAPH_STORE_URI`      → AsyncPostgresStore

These match the values in `.env.example` and the Stage 0 docker-compose
Postgres cluster. `.env` is loaded via python-dotenv on import so local
dev and pytest both see the right values without extra wiring.
"""

from __future__ import annotations

import os
from contextlib import AsyncExitStack, asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

load_dotenv()


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in."
        )
    return value


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    checkpoint_uri = _require_env("LANGGRAPH_CHECKPOINT_URI")
    store_uri = _require_env("LANGGRAPH_STORE_URI")

    async with AsyncExitStack() as stack:
        saver = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(checkpoint_uri)
        )
        store = await stack.enter_async_context(
            AsyncPostgresStore.from_conn_string(store_uri)
        )
        await saver.setup()
        await store.setup()

        app.state.saver = saver
        app.state.store = store

        yield


app = FastAPI(title="ai-trading-agent orchestrator", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}
