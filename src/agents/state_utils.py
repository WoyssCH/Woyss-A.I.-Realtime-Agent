from __future__ import annotations

from collections.abc import Iterable

from agents.schemas import Speaker, UtteranceInput


def role_for_speaker(speaker: str) -> str:
    speaker_norm = speaker.strip().lower()
    if speaker_norm == "assistant":
        return "assistant"
    return "user"


def canonical_speaker_label(speaker: str) -> str:
    speaker_norm = speaker.strip().lower()
    if not speaker_norm:
        return "USER"

    if speaker_norm in {"assistant", "lea"}:
        return "ASSISTANT"
    if speaker_norm in {"patient", "kunde", "klient", "client"}:
        return "PATIENT"
    if speaker_norm in {"staff", "team", "reception", "praxis", "assistant_staff"}:
        return "STAFF"
    if speaker_norm in {"dentist", "doctor", "dr", "zahnarzt", "aerztin", "arzt"}:
        return "DENTIST"

    return speaker.strip().upper()


def normalize_speaker(speaker: str) -> Speaker:
    """Map potentially messy speaker inputs to a small canonical set."""

    speaker_norm = (speaker or "").strip().lower()
    if speaker_norm in {"assistant", "lea"}:
        return "assistant"
    if speaker_norm in {"patient", "kunde", "klient", "client"}:
        return "patient"
    if speaker_norm in {"staff", "team", "reception", "praxis", "assistant_staff"}:
        return "staff"
    if speaker_norm in {"dentist", "doctor", "dr", "zahnarzt", "aerztin", "arzt"}:
        return "dentist"
    # Default: treat unknown speakers as patient-style user input.
    return "patient"


def content_with_speaker_prefix(speaker: str, text: str) -> str:
    speaker_norm = speaker.strip()
    if not speaker_norm:
        return text
    if speaker_norm.lower() == "assistant":
        return text
    return f"{canonical_speaker_label(speaker_norm)}: {text}"


def build_llm_history(system_prompt: str, utterances: Iterable[UtteranceInput]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for utt in utterances:
        history.append(
            {
                "role": role_for_speaker(utt.speaker),
                "content": content_with_speaker_prefix(utt.speaker, utt.text),
            }
        )
    return history


def preferred_language_from_utterances(utterances: list[UtteranceInput]) -> str | None:
    for utt in reversed(utterances):
        if utt.language:
            return utt.language
    return None
