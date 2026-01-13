"""LLM-powered structured information extraction with verification."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable

from agents.schemas import StructuredFactPayload, UtteranceInput
from agents.state_utils import canonical_speaker_label
from config.settings import get_settings
from llm.base import BaseLLMClient
from prompts.loader import load_prompt

LOGGER = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = load_prompt("extraction_system.txt")


VERIFICATION_PROMPT_TEMPLATE = """Kontext des Gespraechs:
{transcript}

Zu ueberpruefender Fakt:
Kategorie: {category}
Feld: {field}
Behaupteter Wert: {value}
Stimme dem Fakt nur zu, wenn er eindeutig aus dem Kontext hervorgeht.
Antwortformat JSON: {{"is_correct": bool, "confidence": float, "corrected_value": str|null}}.
"""




class InformationExtractor:
    """Extracts structured data from conversation transcripts with double-check."""

    def __init__(self, llm_client: BaseLLMClient) -> None:
        self._llm = llm_client
        settings = get_settings()
        self._confidence_threshold = settings.extraction_confidence_threshold

    async def extract(
        self,
        conversation_id: str,
        utterances: Iterable[UtteranceInput],
    ) -> list[StructuredFactPayload]:
        transcript_text = self._build_transcript(utterances)

        initial_facts = await self._initial_pass(transcript_text)
        verified_facts = []
        for fact in initial_facts:
            is_valid = await self._verify_fact(transcript_text, fact)
            if is_valid:
                verified_facts.append(fact)
        return verified_facts

    def _build_transcript(self, utterances: Iterable[UtteranceInput]) -> str:
        lines = []
        for utt in utterances:
            lines.append(f"{canonical_speaker_label(utt.speaker)} ({utt.language}): {utt.text}")
        return "\n".join(lines)

    async def _initial_pass(self, transcript_text: str) -> list[StructuredFactPayload]:
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Transkript:\n" + transcript_text,
            },
        ]

        raw_response = await self._llm.chat(messages, temperature=0.0)
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            LOGGER.error("Extractor returned invalid JSON: %s", raw_response)
            raise ValueError("Invalid extraction JSON") from exc

        facts_payload = payload.get("facts", [])
        structured_facts: list[StructuredFactPayload] = []
        for fact in facts_payload:
            if fact.get("reject") is True:
                continue
            try:
                structured_facts.append(
                    StructuredFactPayload(
                        category=fact["category"],
                        field_name=fact["field_name"],
                        value=fact["value"],
                        confidence=float(fact["confidence"]),
                        evidence=fact["evidence"],
                    )
                )
            except Exception as exc:  # catch schema issues
                LOGGER.warning("Skipping invalid fact payload %s: %s", fact, exc)
        return structured_facts

    async def _verify_fact(
        self, transcript_text: str, fact: StructuredFactPayload
    ) -> bool:
        prompt = VERIFICATION_PROMPT_TEMPLATE.format(
            transcript=transcript_text,
            category=fact.category,
            field=fact.field_name,
            value=fact.value,
        )
        response = await self._llm.complete(prompt, temperature=0.0)
        try:
            verdict = json.loads(response)
        except json.JSONDecodeError:
            LOGGER.warning("Verification response not JSON: %s", response)
            return False

        is_correct = verdict.get("is_correct") is True
        confidence = float(verdict.get("confidence", 0.0))
        if is_correct and confidence >= self._confidence_threshold:
            return True
        if is_correct and confidence < self._confidence_threshold:
            LOGGER.info(
                "Rejected fact due to low confidence %.2f: %s",
                confidence,
                fact.model_dump_json(),
            )
        if not is_correct and verdict.get("corrected_value"):
            LOGGER.info(
                "Extractor flagged correction for %s -> %s",
                fact.field_name,
                verdict.get("corrected_value"),
            )
        return False
