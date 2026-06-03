"""Add strategy_registry.live_started_at — kill-switch idempotency anchor.

Stage 8f (Fork 5d): the out-of-band kill switch must not re-fire ``/stop`` on
every 5-minute poll after the first fire. The guard is "skip if a stop-row was
written AFTER the current live run began" (``kse.fired_at > sr.live_started_at``).

``live_started_at`` is the durable, app-DB anchor the scheduler reads — it
cannot read the graph's ``artifacts.live_started_at`` (that lives in the
LangGraph checkpoint, and reading it would couple the out-of-band switch to the
graph, violating BRD §1.1 rule 7), and ``strategy_registry.started_at`` is the
PAPER-era first-insert time (preserved on ON CONFLICT), not the live-run start.
``_write_live_registry`` (live_spawn) bumps this on every live spawn.

Nullable: paper-stage rows and pre-8f rows have it NULL. The guard's
``fired_at > NULL`` evaluates to NULL → the row is excluded → the guard reports
"not already fired" → the kill fires. That is the SAFE direction (fire rather
than suspend). See DEFERRED.md D-5 for the latent pause-resume edge.

Additive + nullable: no pre-flight check needed (unlike 0002's CHECK swap on
audit-trail rows). Downgrade silently drops the column.

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-03
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE strategy_registry ADD COLUMN live_started_at TIMESTAMPTZ;")


def downgrade() -> None:
    op.execute("ALTER TABLE strategy_registry DROP COLUMN live_started_at;")
