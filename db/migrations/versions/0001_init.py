"""init app schema (BRD §5.8)

Creates the five `app` DB tables verbatim from BRD §5.8 with their CHECK
constraints and indexes:

  - strategy_registry
  - gate_audits
  - telemetry
  - kill_switch_events
  - regime_log

Revision ID: 0001
Revises:
Create Date: 2026-05-26
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE strategy_registry (
            strategy_id TEXT PRIMARY KEY,
            thread_id   TEXT NOT NULL UNIQUE,
            name        TEXT NOT NULL,
            template    TEXT NOT NULL,
            stage       TEXT NOT NULL,
            pairs       JSONB NOT NULL,
            timeframe   TEXT NOT NULL,
            freqtrade_userdir   TEXT,
            freqtrade_api_url   TEXT,
            freqtrade_pid       INT,
            started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_updated TIMESTAMPTZ NOT NULL DEFAULT now(),
            failure_reason TEXT
        );
        """
    )
    op.execute("CREATE INDEX ON strategy_registry(stage);")

    op.execute(
        """
        CREATE TABLE gate_audits (
            id BIGSERIAL PRIMARY KEY,
            strategy_id TEXT REFERENCES strategy_registry(strategy_id),
            gate        TEXT NOT NULL CHECK (gate IN ('backtest','paper','live','live_pause')),
            decision    TEXT NOT NULL CHECK (decision IN ('auto_pass','auto_fail','human_approve','human_reject','human_revise')),
            actor       TEXT NOT NULL,
            payload     JSONB NOT NULL,
            at          TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )
    op.execute("CREATE INDEX ON gate_audits(strategy_id, at DESC);")

    op.execute(
        """
        CREATE TABLE telemetry (
            id BIGSERIAL PRIMARY KEY,
            strategy_id TEXT REFERENCES strategy_registry(strategy_id),
            snapshot_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            stage       TEXT,
            metrics     JSONB NOT NULL,
            source      TEXT NOT NULL
        );
        """
    )
    op.execute("CREATE INDEX ON telemetry(strategy_id, snapshot_at DESC);")

    op.execute(
        """
        CREATE TABLE kill_switch_events (
            id BIGSERIAL PRIMARY KEY,
            strategy_id TEXT REFERENCES strategy_registry(strategy_id),
            fired_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            reason      TEXT NOT NULL,
            metrics     JSONB NOT NULL,
            action_taken TEXT NOT NULL
        );
        """
    )

    op.execute(
        """
        CREATE TABLE regime_log (
            at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            regime      TEXT NOT NULL,
            features    JSONB NOT NULL,
            detector    TEXT NOT NULL,
            PRIMARY KEY (at, detector)
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS regime_log;")
    op.execute("DROP TABLE IF EXISTS kill_switch_events;")
    op.execute("DROP TABLE IF EXISTS telemetry;")
    op.execute("DROP TABLE IF EXISTS gate_audits;")
    op.execute("DROP TABLE IF EXISTS strategy_registry;")
