"""Initial Refaat relational schema."""

from __future__ import annotations

from alembic import op

from refaat.infrastructure.database.schema import metadata

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

# migration-safety-ack: the initial revision emits the full schema via create_all, which builds only
# new tables on a fresh database and is a no-op elsewhere, so there is no rolling-upgrade delta.


def upgrade() -> None:
    metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    metadata.drop_all(bind=op.get_bind())
