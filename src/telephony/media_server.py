from __future__ import annotations

import argparse
import asyncio
import logging
import os
import secrets
import socket
from dataclasses import dataclass

import numpy as np

from config.settings import get_settings
from integrations.twilio_streaming import pcm16_resample, pcm16_to_wav_bytes
from telephony.g711 import ulaw_decode, ulaw_encode
from telephony.rtp import build_rtp_packet, parse_rtp_packet

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class StreamState:
    remote_addr: tuple[str, int]
    in_ssrc: int
    out_ssrc: int
    out_sequence: int
    out_timestamp: int
    pcm8k: list[np.ndarray]


class RtpMediaServer:
    """Minimal UDP RTP media server (PCMU) for Asterisk ExternalMedia.

    Current scope:
    - Receives RTP/PCMU, decodes to PCM16 @ 8kHz and buffers it.
    - Provides a hook (`on_utterance`) where you can run ASR+LLM+TTS.
    - Sends RTP/PCMU back to the peer (Asterisk) using its source address.

    This is a scaffold: true barge-in and jitter handling are TODO.
    """

    def __init__(self, host: str, port: int, *, payload_type: int = 0) -> None:
        self._host = host
        self._port = port
        self._payload_type = payload_type
        self._sock: socket.socket | None = None
        self._streams: dict[int, StreamState] = {}

    async def run_forever(self) -> None:
        loop = asyncio.get_running_loop()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self._host, self._port))
        self._sock.setblocking(False)

        LOGGER.info("RTP media server listening on %s:%s", self._host, self._port)

        while True:
            data, addr = await loop.sock_recvfrom(self._sock, 2048)
            try:
                pkt = parse_rtp_packet(data)
            except ValueError:
                continue

            if pkt.payload_type != self._payload_type:
                continue

            stream = self._streams.get(pkt.ssrc)
            if stream is None:
                stream = StreamState(
                    remote_addr=(addr[0], addr[1]),
                    in_ssrc=pkt.ssrc,
                    out_ssrc=secrets.randbits(32),
                    out_sequence=secrets.randbits(16),
                    out_timestamp=secrets.randbits(32),
                    pcm8k=[],
                )
                self._streams[pkt.ssrc] = stream
                LOGGER.info("New RTP stream ssrc=%s from %s", pkt.ssrc, addr)

            stream.remote_addr = (addr[0], addr[1])
            pcm8k = ulaw_decode(pkt.payload)
            if pcm8k.size:
                stream.pcm8k.append(pcm8k)

            # Simple chunking heuristic for scaffolding: every ~1.2s at 8kHz.
            buffered = sum(x.size for x in stream.pcm8k)
            if buffered >= 9600:
                audio = np.concatenate(stream.pcm8k)
                stream.pcm8k.clear()
                asyncio.create_task(self.on_utterance(stream, audio))

    async def on_utterance(self, stream: StreamState, pcm8k: np.ndarray) -> None:
        """Handle a user utterance (PCM16 @ 8kHz). Override/extend in production."""

        # Resample to 16k for WhisperTranscriber (faster-whisper expects 16k when passing arrays).
        pcm16k = pcm16_resample(pcm8k, 8000, 16000)
        wav = pcm16_to_wav_bytes(pcm16k, 16000)

        # Import lazily to keep module import cheap.
        from agents.assistant_agent import AssistantAgent

        agent = AssistantAgent()

        try:
            transcript, _lang = await agent.transcribe_audio_bytes(wav, language_hint=None)
        except Exception:
            LOGGER.exception("ASR failed")
            return

        if not transcript.strip():
            return

        try:
            result = await agent.handle_text(
                conversation_id=f"pstn:{stream.in_ssrc}",
                speaker="patient",
                text=transcript,
                language="de",
                attributes={"telephony": {"ssrc": stream.in_ssrc}},
            )
            reply_text = str(result.get("assistant_response") or "")
        except Exception:
            LOGGER.exception("Agent pipeline failed")
            return

        if not reply_text:
            return

        # Generate local TTS audio (WAV bytes), convert to 8kHz PCMU RTP and send.
        try:
            tts_audio = await agent._tts.synthesize(reply_text, language="de")  # noqa: SLF001
        except Exception:
            LOGGER.exception("TTS failed")
            return

        await self._send_wav_as_rtp_ulaw(
            stream,
            tts_audio,
            dst_rate=8000,
            frame_ms=20,
        )

    async def _send_wav_as_rtp_ulaw(
        self,
        stream: StreamState,
        wav_bytes: bytes,
        *,
        dst_rate: int,
        frame_ms: int,
    ) -> None:
        if not self._sock:
            return

        # Decode WAV to float -> PCM16 via soundfile.
        import io

        import soundfile as sf

        with sf.SoundFile(io.BytesIO(wav_bytes), mode="r") as f:
            audio = f.read(dtype="float32")
            src_rate = int(f.samplerate)

        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        pcm = pcm16_resample(pcm, src_rate, dst_rate)

        frame_samples = int(dst_rate * frame_ms / 1000)
        if frame_samples <= 0:
            return

        payload_samples = frame_samples

        loop = asyncio.get_running_loop()
        for i in range(0, pcm.size, payload_samples):
            chunk = pcm[i : i + payload_samples]
            if chunk.size < payload_samples:
                # pad with silence
                pad = np.zeros(payload_samples - chunk.size, dtype=np.int16)
                chunk = np.concatenate([chunk, pad])

            ulaw = ulaw_encode(chunk)
            packet = build_rtp_packet(
                payload_type=self._payload_type,
                sequence=stream.out_sequence,
                timestamp=stream.out_timestamp,
                ssrc=stream.out_ssrc,
                marker=False,
                payload=ulaw,
            )
            stream.out_sequence = (stream.out_sequence + 1) & 0xFFFF
            stream.out_timestamp = (stream.out_timestamp + payload_samples) & 0xFFFFFFFF

            await loop.sock_sendto(self._sock, packet, stream.remote_addr)
            await asyncio.sleep(frame_ms / 1000)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTP media server for Asterisk ExternalMedia")
    parser.add_argument("--host", default=os.getenv("MEDIA_SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MEDIA_SERVER_PORT", "20000")))
    parser.add_argument("--pt", type=int, default=int(os.getenv("MEDIA_RTP_PAYLOAD_TYPE", "0")))
    return parser.parse_args()


async def _amain() -> None:
    args = _parse_args()
    server = RtpMediaServer(args.host, args.port, payload_type=args.pt)
    await server.run_forever()


def main() -> None:
    logging.basicConfig(level=get_settings().log_level)
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
