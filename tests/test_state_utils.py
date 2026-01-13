from __future__ import annotations

from agents.schemas import UtteranceInput
from agents.state_utils import build_llm_history, normalize_speaker, preferred_language_from_utterances


def test_build_llm_history_maps_roles_and_includes_system_prompt():
    utterances = [
        UtteranceInput(
            conversation_id="c1",
            speaker="patient",
            language="de",
            text="Hallo",
            confidence=0.9,
            attributes={},
        ),
        UtteranceInput(
            conversation_id="c1",
            speaker="assistant",
            language="de",
            text="Guten Tag!",
            confidence=1.0,
            attributes={},
        ),
    ]

    history = build_llm_history("SYS", utterances)
    assert history[0] == {"role": "system", "content": "SYS"}
    assert history[1] == {"role": "user", "content": "PATIENT: Hallo"}
    assert history[2] == {"role": "assistant", "content": "Guten Tag!"}


def test_preferred_language_uses_last_known_language():
    utterances = [
        UtteranceInput(
            conversation_id="c1",
            speaker="patient",
            language="fr",
            text="Bonjour",
            confidence=0.9,
            attributes={},
        ),
        UtteranceInput(
            conversation_id="c1",
            speaker="assistant",
            language="de",
            text="Guten Tag!",
            confidence=1.0,
            attributes={},
        ),
    ]

    assert preferred_language_from_utterances(utterances) == "de"


def test_build_llm_history_canonicalizes_speaker_prefixes():
    speaker_dentist = normalize_speaker("Dr")
    speaker_staff = normalize_speaker("reception")

    utterances = [
        UtteranceInput(
            conversation_id="c1",
            speaker=speaker_dentist,
            language="de",
            text="Wir planen einen Kontrolltermin.",
            confidence=0.9,
            attributes={},
        ),
        UtteranceInput(
            conversation_id="c1",
            speaker=speaker_staff,
            language="de",
            text="Ich trage das ein.",
            confidence=0.9,
            attributes={},
        ),
    ]

    history = build_llm_history("SYS", utterances)
    assert history[1]["content"].startswith("DENTIST: ")
    assert history[2]["content"].startswith("STAFF: ")
