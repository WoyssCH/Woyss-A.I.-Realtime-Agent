"""FastAPI routes exposing assistant capabilities."""

from __future__ import annotations

import logging
from functools import lru_cache

import base64
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from agents.assistant_agent import AssistantAgent
from api.schemas import AudioUploadResponse, StartConversationResponse, StructuredFactResponse
from db.repository import ConversationRepository

LOGGER = logging.getLogger(__name__)

router = APIRouter()


@lru_cache(maxsize=1)
def _agent_factory() -> AssistantAgent:
    return AssistantAgent()


def get_agent() -> AssistantAgent:
    return _agent_factory()


@router.post(
    "/conversations/{conversation_id}/start",
    response_model=StartConversationResponse,
)
async def start_conversation(
    conversation_id: str,
    agent: AssistantAgent = Depends(get_agent),
) -> StartConversationResponse:
    await agent.start_conversation(conversation_id)
    return StartConversationResponse(conversation_id=conversation_id)


@router.post(
    "/conversations/{conversation_id}/audio",
    response_model=AudioUploadResponse,
)
async def upload_audio(
    conversation_id: str,
    speaker: str,
    audio_file: UploadFile = File(...),
    agent: AssistantAgent = Depends(get_agent),
) -> AudioUploadResponse:
    if audio_file.content_type not in {"audio/wav", "audio/x-wav"}:
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    audio_bytes = await audio_file.read()
    try:
        result = await agent.handle_audio(conversation_id, speaker, audio_bytes)
    except Exception as exc:
        LOGGER.exception("Audio handling failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AudioUploadResponse(
        transcript=result["transcript"],
        assistant_response=result["assistant_response"],
        language=result["language"],
        confidence=result["confidence"],
        assistant_audio_b64=base64.b64encode(result["assistant_audio"]).decode("ascii"),
        assistant_audio_mime=result["assistant_audio_mime"],
        structured_facts=result["structured_facts"],
        actions=result["actions"],
    )


@router.get(
    "/conversations/{conversation_id}/facts",
    response_model=list[StructuredFactResponse],
)
async def get_structured_facts(
    conversation_id: str,
) -> list[StructuredFactResponse]:
    repo = ConversationRepository()
    facts = await repo.list_structured_facts(conversation_id)
    return [
        StructuredFactResponse(
            category=fact.category,
            field_name=fact.field_name,
            value=fact.value,
            confidence=fact.confidence,
            evidence=fact.evidence,
        )
        for fact in facts
    ]
