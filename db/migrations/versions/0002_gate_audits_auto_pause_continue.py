"""Extend gate_audits.decision CHECK with auto_continue + auto_pause.

Stage 8d (BRD §5.6): the live coordinator records an Opus-escalation row in
``gate_audits`` whose ``decision`` is the final verdict — continue / pause /
fail. The 0001 CHECK only allowed ``auto_pass`` / ``auto_fail`` / ``human_*``,
so a coordinator escalation resolving to *continue* (auto_continue) or *pause*
(auto_pause) could not be recorded faithfully. This migration adds the two
auto-* values.

Upgrade is purely additive — it extends the CHECK to allow the new values.
Downgrade is non-trivial post-deployment: it pre-flight-checks for rows using
the new decision values and refuses to proceed if any exist (silent deletion of
audit-trail rows is wrong per BRD §5.8). Day-zero downgrade is harmless;
post-coordinator-escalation downgrade requires the operator to export/delete
those rows first.

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-03
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The inline CHECK in 0001 is auto-named ``<table>_<column>_check`` by Postgres.
_CONSTRAINT = "gate_audits_decision_check"
_DECISIONS_0001 = "'auto_pass','auto_fail','human_approve','human_reject','human_revise'"
_DECISIONS_0002 = (
    "'auto_pass','auto_fail','auto_continue','auto_pause',"
    "'human_approve','human_reject','human_revise'"
)


def upgrade() -> None:
    op.execute(f"ALTER TABLE gate_audits DROP CONSTRAINT IF EXISTS {_CONSTRAINT};")
    op.execute(
        f"ALTER TABLE gate_audits ADD CONSTRAINT {_CONSTRAINT} "
        f"CHECK (decision IN ({_DECISIONS_0002}));"
    )


def downgrade() -> None:
    # Pre-flight: refuse to downgrade if any row uses the new decision values.
    # gate_audits is an audit trail (BRD §5.8); silent deletion is wrong.
    bind = op.get_bind()
    result = bind.execute(
        sa.text(
            "SELECT COUNT(*) FROM gate_audits " "WHERE decision IN ('auto_continue', 'auto_pause')"
        )
    )
    count = result.scalar()
    if count and count > 0:
        raise RuntimeError(
            f"Cannot downgrade 0002: {count} gate_audits row(s) use "
            f"decision IN ('auto_continue', 'auto_pause'). These are "
            f"audit-trail records and cannot be silently deleted. "
            f"Manual intervention required (export rows + delete + retry)."
        )
    op.execute(f"ALTER TABLE gate_audits DROP CONSTRAINT IF EXISTS {_CONSTRAINT};")
    op.execute(
        f"ALTER TABLE gate_audits ADD CONSTRAINT {_CONSTRAINT} "
        f"CHECK (decision IN ({_DECISIONS_0001}));"
    )
