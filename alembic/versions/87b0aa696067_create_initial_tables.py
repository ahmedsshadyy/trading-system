"""create initial tables

Revision ID: 87b0aa696067
Revises:
Create Date: 2026-03-14 06:41:54.786685

"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "87b0aa696067"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "candles",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("instrument", sa.String(20), nullable=False),
        sa.Column("timeframe", sa.String(5), nullable=False),
        sa.Column("timestamp", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(12, 5), nullable=False),
        sa.Column("high", sa.Numeric(12, 5), nullable=False),
        sa.Column("low", sa.Numeric(12, 5), nullable=False),
        sa.Column("close", sa.Numeric(12, 5), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("instrument", "timeframe", "timestamp"),
    )

    op.create_table(
        "signals",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("instrument", sa.String(20), nullable=False),
        sa.Column("strategy_id", sa.Integer(), nullable=False),
        sa.Column("strategy_name", sa.String(100), nullable=False),
        sa.Column("timeframe_trigger", sa.String(5), nullable=False),
        sa.Column("feature_vector", sa.JSON(), nullable=False),
        sa.Column("raw_score", sa.Numeric(5, 4), nullable=True),
        sa.Column("label", sa.SmallInteger(), nullable=True),
        sa.Column("label_source", sa.String(20), nullable=True),
        sa.Column(
            "created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("version_tag", sa.String(50), nullable=False),
        sa.Column("strategies_included", sa.ARRAY(sa.Integer()), nullable=True),
        sa.Column("training_start", sa.Date(), nullable=False),
        sa.Column("training_end", sa.Date(), nullable=False),
        sa.Column("cv_precision", sa.Numeric(5, 4), nullable=True),
        sa.Column("cv_recall", sa.Numeric(5, 4), nullable=True),
        sa.Column("cv_f1", sa.Numeric(5, 4), nullable=True),
        sa.Column("holdout_precision", sa.Numeric(5, 4), nullable=True),
        sa.Column("brier_score", sa.Numeric(5, 4), nullable=True),
        sa.Column("decision_threshold", sa.Numeric(5, 4), nullable=True),
        sa.Column("profit_factor", sa.Numeric(6, 3), nullable=True),
        sa.Column("model_path", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("model_versions")
    op.drop_table("signals")
    op.drop_table("candles")
