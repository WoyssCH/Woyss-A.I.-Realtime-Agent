"""Main orchestration class for the Swiss dental practice voice assistant."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import asdict, dataclass, field

from agents.actions import ActionPlanner
from agents.errors import (
    DatabaseOperationError,
    LLMFailedError,
    NoSpeechDetectedError,
    TranscriptionFailedError,
    TTSFailedError,
)
from agents.extraction import InformationExtractor
from agents.schemas import ActionDirective, StructuredFactPayload, UtteranceInput
from agents.state_utils import (
    build_llm_history,
    content_with_speaker_prefix,
    normalize_speaker,
    preferred_language_from_utterances,
)
from config.settings import get_settings
from db.repository import ConversationRepository
from integrations.nextjs_bridge import NextJSBridge
from llm.factory import build_llm_client
from prompts.loader import load_prompt
from speech.transcriber import TranscriptionSegment, WhisperTranscriber, merge_segments
from speech.tts import BaseSynthesizer, build_synthesizer

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = load_prompt("assistant_system.txt")


@dataclass
class ConversationUtterance:
    """In-memory representation combining payload and database id."""

    payload: UtteranceInput
    db_id: int


@dataclass
class ConversationState:
    """Tracks conversational history for LLM context."""

    conversation_id: str
    history: list[dict[str, str]] = field(default_factory=list)
    utterances: list[ConversationUtterance] = field(default_factory=list)
    preferred_language: str | None = None


class AssistantAgent:
    """High-level orchestrator for the speech assistant pipeline."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._transcriber = WhisperTranscriber()
        self._tts: BaseSynthesizer = build_synthesizer()
        self._llm = build_llm_client()
        self._extractor = InformationExtractor(self._llm)
        self._action_planner = ActionPlanner(self._llm)
        self._nextjs_bridge = self._build_nextjs_bridge()
        self._repo = ConversationRepository()

    async def start_conversation(self, conversation_id: str) -> None:
        try:
            await self._repo.get_or_create_conversation(conversation_id)
        except Exception as exc:
            raise DatabaseOperationError() from exc

    async def handle_audio(
        self, conversation_id: str, speaker: str, audio_bytes: bytes
    ) -> dict:
        await self.start_conversation(conversation_id)
        state = await self._load_state(conversation_id)

        try:
            segments = await asyncio.to_thread(
                self._transcriber.transcribe, audio_bytes, state.preferred_language
            )
        except Exception as exc:
            raise TranscriptionFailedError() from exc

        if not segments:
            raise NoSpeechDetectedError()

        merged_text = merge_segments(segments)
        language = segments[0].language if segments else "de"
        confidence = self._aggregate_confidence(segments)

        speaker_norm = normalize_speaker(speaker)

        utterance_payload = UtteranceInput(
            conversation_id=conversation_id,
            speaker=speaker_norm,
            language=language,
            text=merged_text,
            confidence=confidence,
            attributes={"segments": [asdict(segment) for segment in segments]},
        )

        try:
            db_utterance = await self._repo.add_utterance(
                conversation_id,
                speaker=speaker_norm,
                language=language,
                text=merged_text,
                confidence=confidence,
                attributes=utterance_payload.attributes,
            )
        except Exception as exc:
            raise DatabaseOperationError() from exc

        state.utterances.append(ConversationUtterance(payload=utterance_payload, db_id=db_utterance.id))
        state.history.append(
            {
                "role": "user",
                "content": content_with_speaker_prefix(speaker_norm, merged_text),
            }
        )
        state.preferred_language = language

        try:
            assistant_text = await self._generate_response(state)
        except Exception as exc:
            raise LLMFailedError() from exc

        try:
            assistant_db = await self._repo.add_utterance(
                conversation_id,
                speaker="assistant",
                language=language,
                text=assistant_text,
                confidence=1.0,
                attributes={},
            )
        except Exception as exc:
            raise DatabaseOperationError() from exc

        assistant_payload = UtteranceInput(
            conversation_id=conversation_id,
            speaker="assistant",
            language=language,
            text=assistant_text,
            confidence=1.0,
            attributes={},
        )
        state.utterances.append(ConversationUtterance(payload=assistant_payload, db_id=assistant_db.id))
        state.history.append({"role": "assistant", "content": assistant_text})

        try:
            tts_audio = await self._tts.synthesize(assistant_text, language=language)
        except Exception as exc:
            raise TTSFailedError() from exc

        try:
            structured_facts = await self._extract_structured_data(state)
        except Exception as exc:
            LOGGER.exception("Structured extraction failed: %s", exc)
            structured_facts = []
        actions = await self._maybe_trigger_actions(state)

        return {
            "transcript": merged_text,
            "assistant_response": assistant_text,
            "assistant_audio": tts_audio,
            "assistant_audio_mime": "audio/mp3" if self._settings.tts_provider == "azure" else "audio/wav",
            "language": language,
            "confidence": confidence,
            "structured_facts": [fact.model_dump() for fact in structured_facts],
            "actions": [action.model_dump() for action in actions],
        }

    async def handle_text(
        self,
        conversation_id: str,
        *,
        speaker: str,
        text: str,
        language: str = "de",
        confidence: float = 1.0,
        attributes: dict | None = None,
    ) -> dict:
        await self.start_conversation(conversation_id)
        state = await self._load_state(conversation_id)

        speaker_norm = normalize_speaker(speaker)
        cleaned = text.strip()
        if not cleaned:
            raise NoSpeechDetectedError()

        utterance_payload = UtteranceInput(
            conversation_id=conversation_id,
            speaker=speaker_norm,
            language=language,
            text=cleaned,
            confidence=confidence,
            attributes=attributes or {},
        )

        try:
            db_utterance = await self._repo.add_utterance(
                conversation_id,
                speaker=speaker_norm,
                language=language,
                text=cleaned,
                confidence=confidence,
                attributes=utterance_payload.attributes,
            )
        except Exception as exc:
            raise DatabaseOperationError() from exc

        state.utterances.append(ConversationUtterance(payload=utterance_payload, db_id=db_utterance.id))
        state.history.append({"role": "user", "content": content_with_speaker_prefix(speaker_norm, cleaned)})
        state.preferred_language = language

        try:
            assistant_text = await self._generate_response(state)
        except Exception as exc:
            raise LLMFailedError() from exc

        try:
            assistant_db = await self._repo.add_utterance(
                conversation_id,
                speaker="assistant",
                language=language,
                text=assistant_text,
                confidence=1.0,
                attributes={},
            )
        except Exception as exc:
            raise DatabaseOperationError() from exc

        assistant_payload = UtteranceInput(
            conversation_id=conversation_id,
            speaker="assistant",
            language=language,
            text=assistant_text,
            confidence=1.0,
            attributes={},
        )
        state.utterances.append(ConversationUtterance(payload=assistant_payload, db_id=assistant_db.id))
        state.history.append({"role": "assistant", "content": assistant_text})

        try:
            structured_facts = await self._extract_structured_data(state)
        except Exception as exc:
            LOGGER.exception("Structured extraction failed: %s", exc)
            structured_facts = []

        actions = await self._maybe_trigger_actions(state)

        return {
            "transcript": cleaned,
            "assistant_response": assistant_text,
            "language": language,
            "confidence": confidence,
            "structured_facts": [fact.model_dump() for fact in structured_facts],
            "actions": [action.model_dump() for action in actions],
        }

    async def handle_text_stream(
        self,
        conversation_id: str,
        *,
        speaker: str,
        text: str,
        language: str = "de",
        confidence: float = 1.0,
        attributes: dict | None = None,
        temperature: float = 0.2,
    ):
        """Stream the assistant response text while persisting conversation state.

        This keeps the LLM backend swappable: any provider can implement
        `BaseLLMClient.stream_chat`; if not, it will fall back to a single chunk.

        Side effects:
        - Persists the user utterance immediately.
        - Streams assistant text via LLM.
        - Persists the assistant utterance at the end.
        - Triggers extraction/actions (best-effort) after assistant completion.
        """

        await self.start_conversation(conversation_id)
        state = await self._load_state(conversation_id)

        speaker_norm = normalize_speaker(speaker)
        cleaned = text.strip()
        if not cleaned:
            raise NoSpeechDetectedError()

        utterance_payload = UtteranceInput(
            conversation_id=conversation_id,
            speaker=speaker_norm,
            language=language,
            text=cleaned,
            confidence=confidence,
            attributes=attributes or {},
        )

        try:
            db_utterance = await self._repo.add_utterance(
                conversation_id,
                speaker=speaker_norm,
                language=language,
                text=cleaned,
                confidence=confidence,
                attributes=utterance_payload.attributes,
            )
        except Exception as exc:
            raise DatabaseOperationError() from exc

        state.utterances.append(ConversationUtterance(payload=utterance_payload, db_id=db_utterance.id))
        state.history.append({"role": "user", "content": content_with_speaker_prefix(speaker_norm, cleaned)})
        state.preferred_language = language

        assistant_parts: list[str] = []
        try:
            async for chunk in self._llm.stream_chat(state.history, temperature=temperature):
                if not chunk:
                    continue
                assistant_parts.append(chunk)
                yield chunk
        except Exception as exc:
            raise LLMFailedError() from exc

        assistant_text = "".join(assistant_parts).strip()
        if not assistant_text:
            assistant_text = "Entschuldigung, ich habe gerade keine Antwort."  # safe fallback

        try:
            assistant_db = await self._repo.add_utterance(
                conversation_id,
                speaker="assistant",
                language=language,
                text=assistant_text,
                confidence=1.0,
                attributes={},
            )
        except Exception as exc:
            raise DatabaseOperationError() from exc

        assistant_payload = UtteranceInput(
            conversation_id=conversation_id,
            speaker="assistant",
            language=language,
            text=assistant_text,
            confidence=1.0,
            attributes={},
        )
        state.utterances.append(ConversationUtterance(payload=assistant_payload, db_id=assistant_db.id))
        state.history.append({"role": "assistant", "content": assistant_text})

        try:
            _ = await self._extract_structured_data(state)
        except Exception as exc:
            LOGGER.exception("Structured extraction failed: %s", exc)

        try:
            _ = await self._maybe_trigger_actions(state)
        except Exception as exc:
            LOGGER.exception("Action planning failed: %s", exc)

    async def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        *,
        language_hint: str | None = None,
    ) -> tuple[str, str | None]:
        segments = await asyncio.to_thread(self._transcriber.transcribe, audio_bytes, language_hint)
        if not segments:
            raise NoSpeechDetectedError()
        merged_text = merge_segments(segments)
        language = segments[0].language if segments else language_hint
        return merged_text, language

    async def _load_state(self, conversation_id: str) -> ConversationState:
        try:
            recent = await self._repo.list_recent_utterances(conversation_id, limit=20)
        except Exception as exc:
            raise DatabaseOperationError() from exc

        utterance_inputs: list[UtteranceInput] = []
        utterance_entries: list[ConversationUtterance] = []
        for row in recent:
            speaker_norm = normalize_speaker(row.speaker)
            payload = UtteranceInput(
                conversation_id=conversation_id,
                speaker=speaker_norm,
                language=row.language,
                text=row.text,
                confidence=row.confidence,
                attributes=row.attributes or {},
            )
            utterance_inputs.append(payload)
            utterance_entries.append(ConversationUtterance(payload=payload, db_id=row.id))

        return ConversationState(
            conversation_id=conversation_id,
            history=build_llm_history(SYSTEM_PROMPT, utterance_inputs),
            utterances=utterance_entries,
            preferred_language=preferred_language_from_utterances(utterance_inputs),
        )

    async def _generate_response(self, state: ConversationState) -> str:
        response = await self._llm.chat(state.history, temperature=0.2)
        return response.strip()

    def _aggregate_confidence(self, segments: list[TranscriptionSegment]) -> float:
        if not segments:
            return 0.0
        confidences = [self._segment_confidence(segment.logprob) for segment in segments]
        return float(sum(confidences) / len(confidences))

    @staticmethod
    def _segment_confidence(log_prob: float) -> float:
        prob = math.exp(log_prob)
        return max(0.0, min(1.0, prob))

    async def _extract_structured_data(
        self, state: ConversationState
    ) -> list[StructuredFactPayload]:
        recent_utterances = [
            entry.payload for entry in state.utterances[-10:]
        ]  # limit context for extraction

        if not any(utt.speaker == "patient" for utt in recent_utterances):
            return []

        facts = await self._extractor.extract(state.conversation_id, recent_utterances)

        if not facts:
            return []

        existing = await self._repo.list_structured_facts(state.conversation_id)
        existing_keys = {(fact.field_name.lower(), fact.value.strip()) for fact in existing}
        deduped: list[StructuredFactPayload] = []
        for fact in facts:
            key = (fact.field_name.lower(), fact.value.strip())
            if key in existing_keys:
                continue
            deduped.append(fact)

        facts = deduped
        if not facts:
            return []

        # Heuristic: associate extracted facts with latest patient utterance
        latest_patient = next(
            (entry for entry in reversed(state.utterances) if entry.payload.speaker == "patient"),
            None,
        )

        fact_dicts = []
        for fact in facts:
            if latest_patient:
                fact.source_utterance_id = latest_patient.db_id
            payload = fact.model_copy()
            payload.source_utterance_id = fact.source_utterance_id
            fact_dicts.append(payload.model_dump())

        await self._repo.add_structured_facts(state.conversation_id, fact_dicts)
        return facts

    async def _maybe_trigger_actions(self, state: ConversationState) -> list[ActionDirective]:
        try:
            recent = [entry.payload for entry in state.utterances[-8:]]
            directives = await self._action_planner.plan(recent)
            filtered = [directive for directive in directives if directive.confidence >= 0.75]
            if self._nextjs_bridge:
                for directive in filtered:
                    await self._nextjs_bridge.dispatch(directive)
            return filtered
        except Exception as exc:
            LOGGER.exception("Action dispatch failed: %s", exc)
            return []

    def _build_nextjs_bridge(self) -> NextJSBridge | None:
        try:
            return NextJSBridge()
        except ValueError:
            return None
