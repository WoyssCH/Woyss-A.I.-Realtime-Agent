from __future__ import annotations

import asyncio

import pytest

from agents.extraction import InformationExtractor
from agents.schemas import UtteranceInput
from llm.base import BaseLLMClient


class FakeLLM(BaseLLMClient):
    def __init__(
        self,
        *,
        chat_response: str,
        complete_response: str | None = None,
        fail_complete: bool = False,
    ) -> None:
        self._chat_response = chat_response
        self._complete_response = complete_response
        self._fail_complete = fail_complete
        self.complete_calls: list[str] = []

    async def complete(self, prompt: str, *, temperature: float = 0.1) -> str:
        if self._fail_complete:
            raise AssertionError("complete() should not have been called")
        self.complete_calls.append(prompt)
        return self._complete_response or ""

    async def chat(self, messages, *, temperature: float = 0.1) -> str:
        return self._chat_response


def _run(coro):
    return asyncio.run(coro)


def test_extractor_raises_on_invalid_json_from_llm_chat():
    llm = FakeLLM(chat_response="{not valid json")
    extractor = InformationExtractor(llm)

    utterances = [
        UtteranceInput(
            conversation_id="c1",
            speaker="patient",
            language="de",
            text="Ich heisse Max Muster.",
            confidence=0.9,
            attributes={},
        )
    ]

    with pytest.raises(ValueError, match="Invalid extraction JSON"):
        _run(extractor.extract("c1", utterances))


def test_extractor_rejects_fact_when_verification_is_not_json():
    llm = FakeLLM(
        chat_response=(
            '{"facts": [{"category": "patient", "field_name": "patient_name", "value": "Max Muster", '
            '"confidence": 0.9, "evidence": "Ich heisse Max Muster.", "reject": false}]}'
        ),
        complete_response="not json at all",
    )
    extractor = InformationExtractor(llm)

    utterances = [
        UtteranceInput(
            conversation_id="c1",
            speaker="patient",
            language="de",
            text="Ich heisse Max Muster.",
            confidence=0.9,
            attributes={},
        )
    ]

    facts = _run(extractor.extract("c1", utterances))
    assert facts == []
    assert len(llm.complete_calls) == 1


def test_extractor_rejects_fact_below_confidence_threshold():
    llm = FakeLLM(
        chat_response=(
            '{"facts": [{"category": "patient", "field_name": "patient_name", "value": "Max Muster", '
            '"confidence": 0.9, "evidence": "Ich heisse Max Muster.", "reject": false}]}'
        ),
        complete_response='{"is_correct": true, "confidence": 0.5, "corrected_value": null}',
    )
    extractor = InformationExtractor(llm)

    utterances = [
        UtteranceInput(
            conversation_id="c1",
            speaker="patient",
            language="de",
            text="Ich heisse Max Muster.",
            confidence=0.9,
            attributes={},
        )
    ]

    facts = _run(extractor.extract("c1", utterances))
    assert facts == []


def test_extractor_skips_rejected_facts_without_verification_call():
    llm = FakeLLM(
        chat_response=(
            '{"facts": [{"category": "patient", "field_name": "patient_name", "value": "Max Muster", '
            '"confidence": 0.9, "evidence": "Ich heisse Max Muster.", "reject": true}]}'
        ),
        fail_complete=True,
    )
    extractor = InformationExtractor(llm)

    utterances = [
        UtteranceInput(
            conversation_id="c1",
            speaker="patient",
            language="de",
            text="Ich heisse Max Muster.",
            confidence=0.9,
            attributes={},
        )
    ]

    facts = _run(extractor.extract("c1", utterances))
    assert facts == []
