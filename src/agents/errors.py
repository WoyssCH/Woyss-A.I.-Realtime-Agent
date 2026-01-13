"""Domain-specific exceptions for assistant operations.

These exceptions are safe to import from API layers without triggering heavy ML imports.
"""

from __future__ import annotations


class AssistantError(Exception):
    status_code: int = 500
    default_detail: str = "Assistant error"

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(detail or self.default_detail)
        self.detail = detail or self.default_detail


class NoSpeechDetectedError(AssistantError):
    status_code = 422
    default_detail = "Keine Spracheingabe erkannt."


class TranscriptionFailedError(AssistantError):
    status_code = 503
    default_detail = "Transkription fehlgeschlagen."


class LLMFailedError(AssistantError):
    status_code = 503
    default_detail = "LLM-Anfrage fehlgeschlagen."


class TTSFailedError(AssistantError):
    status_code = 503
    default_detail = "Sprachausgabe fehlgeschlagen."


class DatabaseOperationError(AssistantError):
    status_code = 503
    default_detail = "Datenbank-Operation fehlgeschlagen."