"""initial schema

Revision ID: 0001_initial_schema
Revises: 
Create Date: 2026-01-12

"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("conversation_id", sa.String(length=64), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_conversations_conversation_id",
        "conversations",
        ["conversation_id"],
        unique=True,
    )

    op.create_table(
        "utterances",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("speaker", sa.String(length=32), nullable=False),
        sa.Column("language", sa.String(length=16), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("attributes", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_utterances_conversation_id",
        "utterances",
        ["conversation_id"],
        unique=False,
    )

    op.create_table(
        "structured_facts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("source_utterance_id", sa.Integer(), nullable=True),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("field_name", sa.String(length=128), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("evidence", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_utterance_id"],
            ["utterances.id"],
            ondelete="SET NULL",
        ),
    )
    op.create_index(
        "ix_structured_facts_conversation_id",
        "structured_facts",
        ["conversation_id"],
        unique=False,
    )
    op.create_index(
        "ix_structured_facts_source_utterance_id",
        "structured_facts",
        ["source_utterance_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_structured_facts_source_utterance_id", table_name="structured_facts")
    op.drop_index("ix_structured_facts_conversation_id", table_name="structured_facts")
    op.drop_table("structured_facts")

    op.drop_index("ix_utterances_conversation_id", table_name="utterances")
    op.drop_table("utterances")

    op.drop_index("ix_conversations_conversation_id", table_name="conversations")
    op.drop_table("conversations")
