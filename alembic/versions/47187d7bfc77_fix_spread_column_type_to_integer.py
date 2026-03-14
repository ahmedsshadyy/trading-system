"""fix spread column type to integer

Revision ID: 47187d7bfc77
Revises: 555d21c74d3c
Create Date: 2026-03-14 07:08:28.876559

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "47187d7bfc77"
down_revision: Union[str, Sequence[str], None] = "555d21c74d3c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "candles", "spread", type_=sa.Integer(), existing_type=sa.Numeric(10, 5)
    )


def downgrade() -> None:
    op.alter_column(
        "candles", "spread", type_=sa.Numeric(10, 5), existing_type=sa.Integer()
    )
