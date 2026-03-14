"""add spread column to candles

Revision ID: 555d21c74d3c
Revises: 87b0aa696067
Create Date: 2026-03-14 07:02:51.722830

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "555d21c74d3c"
down_revision: Union[str, Sequence[str], None] = "87b0aa696067"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("candles", sa.Column("spread", sa.Numeric(10, 5), nullable=True))


def downgrade() -> None:
    op.drop_column("candles", "spread")
