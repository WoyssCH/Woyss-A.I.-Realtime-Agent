"""API-facing Pydantic models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StartConversationResponse(BaseModel):
    conversation_id: str
    status: str = "started"


class AudioUploadResponse(BaseModel):
    transcript: str
    assistant_response: str
    language: str
    confidence: float
    assistant_audio_b64: str = Field(description="Base64-encoded audio data for playback.")
    assistant_audio_mime: str = Field(description="MIME type for the returned audio bytes.")
    structured_facts: list[dict]
    actions: list[dict]


class StructuredFactResponse(BaseModel):
    category: str
    field_name: str
    value: str
    confidence: float
    evidence: str
