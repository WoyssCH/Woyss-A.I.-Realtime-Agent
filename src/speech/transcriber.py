"""High accuracy multilingual speech-to-text transcriber based on Whisper."""

from __future__ import annotations

import io
import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from langdetect import DetectorFactory, detect

from config.settings import get_settings

DetectorFactory.seed = 7  # deterministic language detection

LOGGER = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """Structured representation of a Whisper transcription segment."""

    start: float
    end: float
    text: str
    language: str
    logprob: float


class WhisperTranscriber:
    """Streaming-friendly transcription using faster-whisper."""

    def __init__(self) -> None:
        settings = get_settings()
        self._model = WhisperModel(
            model_size_or_path=settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )

    def transcribe(
        self, audio_bytes: bytes, language_hint: str | None = None
    ) -> list[TranscriptionSegment]:
        """Transcribe raw audio bytes into text segments."""

        with sf.SoundFile(io.BytesIO(audio_bytes), mode="r") as audio_file:
            audio_array = audio_file.read(dtype="float32")

        if isinstance(audio_array, np.ndarray) and audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)  # convert to mono
        elif not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)

        segments, info = self._model.transcribe(
            audio_array,
            beam_size=5,
            task="transcribe",
            language=language_hint,
            condition_on_previous_text=True,
            initial_prompt=self._initial_prompt(language_hint),
            temperature=0.0,
        )

        detected_language = info.language
        results: list[TranscriptionSegment] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue

            language = detected_language or self._safe_detect(text) or "unknown"
            results.append(
                TranscriptionSegment(
                    start=segment.start,
                    end=segment.end,
                    text=text,
                    language=language,
                    logprob=segment.avg_logprob,
                )
            )

        return results

    @staticmethod
    def _initial_prompt(language_hint: str | None) -> str | None:
        if not language_hint:
            return (
                "Dies ist ein Gespraech in einer Schweizer Zahnarztpraxis. "
                "Die sprechenden Personen verwenden Schweizerdeutsch, Hochdeutsch, "
                "Franzoesisch, Italienisch oder Raetoromanisch."
            )

        prompts = {
            "de": "Dies ist ein Gespraech in einer Schweizer Zahnarztpraxis in Schweizerdeutsch oder Hochdeutsch.",
            "fr": "Ceci est une conversation dans un cabinet dentaire suisse en francais.",
            "it": "Questa e una conversazione in uno studio dentistico svizzero in italiano.",
            "rm": "Quai ei in discurs en ina pratica da dentist svizra en rumantsch.",
        }
        return prompts.get(language_hint.split("-")[0], None)

    @staticmethod
    def _safe_detect(text: str) -> str | None:
        try:
            return detect(text)
        except Exception:  # langdetect throws generic exceptions
            LOGGER.debug("Language detection failed for text: %s", text)
        return None


def merge_segments(segments: Iterable[TranscriptionSegment]) -> str:
    """Merge segments into a single string."""

    return " ".join(segment.text for segment in segments).strip()
