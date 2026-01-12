"""Text-to-speech synthesis for female Swiss practice assistant voice."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from config.settings import get_settings

LOGGER = logging.getLogger(__name__)


class BaseSynthesizer(ABC):
    """Interface for all text-to-speech synthesizers."""

    @abstractmethod
    async def synthesize(
        self, text: str, language: str, voice: str | None = None
    ) -> bytes:
        """Synthesize speech for the given text and language."""


class AzureSynthesizer(BaseSynthesizer):
    """Wrapper around Azure Cognitive Services Speech SDK."""

    def __init__(self) -> None:
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "azure-cognitiveservices-speech is required for AzureSynthesizer."
            ) from exc

        settings = get_settings()
        if not settings.azure_speech_key or not settings.azure_speech_region:
            raise ValueError("Azure speech key and region must be configured.")

        speech_config = speechsdk.SpeechConfig(
            subscription=settings.azure_speech_key,
            region=settings.azure_speech_region,
        )

        speech_config.speech_synthesis_voice_name = settings.default_voice_female
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )

        self._speechsdk = speechsdk
        self._speech_config = speech_config

    async def synthesize(
        self, text: str, language: str, voice: str | None = None
    ) -> bytes:
        voice_name = voice or self._select_voice(language)
        self._speech_config.speech_synthesis_voice_name = voice_name
        synthesizer = self._speechsdk.SpeechSynthesizer(
            speech_config=self._speech_config,
            audio_config=None,  # allow retrieving audio data directly
        )
        ssml = self._build_ssml(text=text, voice_name=voice_name)
        result = synthesizer.speak_ssml_async(ssml).get()

        if result.reason == self._speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            raise RuntimeError(f"Azure TTS canceled: {cancellation.error_details}")

        return result.audio_data

    @staticmethod
    def _select_voice(language: str) -> str:
        mapping = {
            "de": "de-CH-LeniNeural",
            "fr": "fr-CH-ArianeNeural",
            "it": "it-CH-GiannaNeural",
            "rm": "de-CH-LeniNeural",  # fallback voice, Romansh not natively supported
            "en": "en-GB-LibbyNeural",
        }
        base_lang = language.split("-")[0]
        return mapping.get(base_lang, "de-CH-LeniNeural")

    @staticmethod
    def _build_ssml(text: str, voice_name: str) -> str:
        escaped_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return (
            "<speak version='1.0' xml:lang='de-CH'>"
            f"<voice name='{voice_name}'>{escaped_text}</voice>"
            "</speak>"
        )


class CoquiSynthesizer(BaseSynthesizer):
    """Offline TTS using Coqui TTS models."""

    def __init__(self) -> None:
        try:
            from TTS.api import TTS  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("TTS package required for CoquiSynthesizer.") from exc

        self._tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

    async def synthesize(
        self, text: str, language: str, voice: str | None = None
    ) -> bytes:
        wav_path = Path("data/tmp_tts.wav")
        wav_path.parent.mkdir(parents=True, exist_ok=True)

        lang = language.split("-")[0]
        # Use female speaker embedding provided by xtts model
        self._tts.tts_to_file(
            text=text,
            file_path=str(wav_path),
            speaker_wav=None,
            language=lang,
        )

        import io

        import soundfile as sf  # lazy import

        audio_array, sample_rate = sf.read(str(wav_path), dtype="float32")
        audio_array = np.asarray(audio_array, dtype=np.float32)
        wav_path.unlink(missing_ok=True)

        output = io.BytesIO()
        sf.write(output, audio_array, sample_rate, format="WAV")
        return output.getvalue()


def build_synthesizer() -> BaseSynthesizer:
    """Factory returning the configured synthesizer."""

    settings = get_settings()
    if settings.tts_provider == "azure":
        return AzureSynthesizer()
    if settings.tts_provider == "coqui":
        return CoquiSynthesizer()
    raise ValueError(f"Unsupported TTS provider: {settings.tts_provider}")
