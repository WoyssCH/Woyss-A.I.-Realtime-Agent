"""SQLAlchemy models for conversation capture."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base


class Conversation(Base):
    """Conversation session metadata."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    started_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))

    utterances: Mapped[list[Utterance]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    structured_facts: Mapped[list[StructuredFact]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class Utterance(Base):
    """Individual user or assistant utterances."""

    __tablename__ = "utterances"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    speaker: Mapped[str] = mapped_column(String(32))
    language: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(Text())
    confidence: Mapped[float] = mapped_column()
    started_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))
    attributes: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    conversation: Mapped[Conversation] = relationship(back_populates="utterances")
    structured_facts: Mapped[list[StructuredFact]] = relationship(
        back_populates="source_utterance",
        lazy="selectin",
    )


class StructuredFact(Base):
    """Structured information extracted from a conversation."""

    __tablename__ = "structured_facts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    source_utterance_id: Mapped[int | None] = mapped_column(
        ForeignKey("utterances.id", ondelete="SET NULL"), index=True
    )
    category: Mapped[str] = mapped_column(String(64))
    field_name: Mapped[str] = mapped_column(String(128))
    value: Mapped[str] = mapped_column(Text())
    confidence: Mapped[float] = mapped_column()
    evidence: Mapped[str] = mapped_column(Text())
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))

    conversation: Mapped[Conversation] = relationship(back_populates="structured_facts")
    source_utterance: Mapped[Utterance] = relationship(back_populates="structured_facts")
