-- Stage 0 init: create the three logical DBs from BRD §5.8 and enable
-- pgvector on the langgraph_store DB (BRD §5.8 #2 + §5.9 semantic-search namespace).
--
-- This script runs once, on first boot of the postgres container, via
-- /docker-entrypoint-initdb.d/.

-- App DB (owned by us — Alembic migrations land in Stage 1)
CREATE DATABASE app;

-- LangGraph PostgresSaver checkpoints DB (managed by checkpointer.setup() in Stage 1)
CREATE DATABASE langgraph_checkpoints;

-- LangGraph long-term Store DB (managed by store.setup() in Stage 1; pgvector for semantic search)
CREATE DATABASE langgraph_store;

\connect langgraph_store
CREATE EXTENSION IF NOT EXISTS vector;

\connect app
-- pgvector also enabled on app DB in case telemetry / regime_log ever wants embeddings.
CREATE EXTENSION IF NOT EXISTS vector;
