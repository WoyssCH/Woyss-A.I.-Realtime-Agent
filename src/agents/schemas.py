"""Pydantic schemas for agent exchange."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, validator

Speaker = Literal["patient", "assistant", "staff"]


class UtteranceInput(BaseModel):
    """Captured utterance details prior to persistence."""

    conversation_id: str
    speaker: Speaker
    language: str
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @validator("text")
    def text_not_empty(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("Text may not be empty.")
        return text


class StructuredFactPayload(BaseModel):
    """Structured fact ready for persistence."""

    source_utterance_id: int | None = None
    category: str
    field_name: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str


class ActionDirective(BaseModel):
    """Instruction to trigger external system action."""

    action: str
    payload: dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
