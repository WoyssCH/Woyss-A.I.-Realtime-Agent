from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import numpy as np


def mulaw_decode(ulaw_bytes: bytes) -> np.ndarray:
    """Decode G.711 mu-law bytes to 16-bit PCM int16 numpy array."""

    data = np.frombuffer(ulaw_bytes, dtype=np.uint8)

    # Vectorized mu-law decode.
    mu = np.bitwise_not(data)
    sign = np.bitwise_and(mu, 0x80)
    exponent = np.right_shift(np.bitwise_and(mu, 0x70), 4)
    mantissa = np.bitwise_and(mu, 0x0F)

    magnitude = ((mantissa.astype(np.int32) << 1) + 33) << (exponent.astype(np.int32) + 2)
    pcm = magnitude.astype(np.int32) - 33
    pcm = np.where(sign != 0, -pcm, pcm)

    return pcm.astype(np.int16)


def pcm16_resample(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return pcm
    if pcm.size == 0:
        return pcm.astype(np.int16)

    x_old = np.arange(pcm.size, dtype=np.float32)
    x_new = np.linspace(0, pcm.size - 1, int(pcm.size * dst_rate / src_rate), dtype=np.float32)

    y_old = pcm.astype(np.float32)
    y_new = np.interp(x_new, x_old, y_old)

    return np.clip(y_new, -32768, 32767).astype(np.int16)


def pcm16_to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    import wave

    pcm_bytes = pcm.astype(np.int16).tobytes()
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buffer.getvalue()


@dataclass
class TwilioStreamSession:
    call_sid: str
    pcm_chunks: list[np.ndarray] = field(default_factory=list)
    last_event: str | None = None


class TwilioStreamStore:
    """In-memory store for Twilio Media Streams.

    Note: This is a single-process store. For multi-worker deployments, replace
    with Redis or another shared store.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: dict[str, TwilioStreamSession] = {}

    async def get_or_create(self, call_sid: str) -> TwilioStreamSession:
        async with self._lock:
            session = self._sessions.get(call_sid)
            if session is None:
                session = TwilioStreamSession(call_sid=call_sid)
                self._sessions[call_sid] = session
            return session

    async def append_ulaw_chunk(self, call_sid: str, payload_b64: str) -> None:
        raw = base64.b64decode(payload_b64)
        pcm8k = mulaw_decode(raw)
        async with self._lock:
            session = self._sessions.setdefault(call_sid, TwilioStreamSession(call_sid=call_sid))
            session.pcm_chunks.append(pcm8k)
            session.last_event = "media"

    async def pop_audio_wav(self, call_sid: str, *, dst_rate: int = 16000) -> bytes | None:
        async with self._lock:
            session = self._sessions.get(call_sid)
            if not session or not session.pcm_chunks:
                return None
            pcm8k = np.concatenate(session.pcm_chunks)
            session.pcm_chunks.clear()

        pcm16k = pcm16_resample(pcm8k, 8000, dst_rate)
        return pcm16_to_wav_bytes(pcm16k, dst_rate)

    async def handle_event(self, call_sid: str, message: dict[str, Any]) -> None:
        event = str(message.get("event") or "")
        if event == "media":
            media = message.get("media") or {}
            if media.get("track") and media.get("track") != "inbound":
                return
            payload = media.get("payload")
            if isinstance(payload, str) and payload:
                await self.append_ulaw_chunk(call_sid, payload)
        else:
            session = await self.get_or_create(call_sid)
            async with self._lock:
                session.last_event = event


GLOBAL_TWILIO_STREAM_STORE = TwilioStreamStore()


def parse_twilio_ws_message(text: str) -> dict[str, Any]:
    return json.loads(text)
