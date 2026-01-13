from __future__ import annotations

import argparse
import asyncio
import logging
import os
import secrets
import socket
from dataclasses import dataclass
from typing import Final

import numpy as np

from config.settings import get_settings
from integrations.twilio_streaming import pcm16_resample, pcm16_to_wav_bytes
from telephony.g711 import ulaw_decode, ulaw_encode
from telephony.rtp import build_rtp_packet, parse_rtp_packet
from telephony.vad import EnergyVAD, VADConfig

LOGGER = logging.getLogger(__name__)


FRAME_MS: Final[int] = 20
FRAME_SAMPLES_8K: Final[int] = 160


@dataclass(slots=True)
class StreamState:
    remote_addr: tuple[str, int]
    in_ssrc: int
    out_ssrc: int
    out_sequence: int
    out_timestamp: int
    vad: EnergyVAD
    speaking_task: asyncio.Task | None
    speech_task: asyncio.Task | None
    barge_in: asyncio.Event


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
        self._agent = None

    def _get_agent(self):
        if self._agent is None:
            from agents.assistant_agent import AssistantAgent

            self._agent = AssistantAgent()
        return self._agent

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
                    vad=EnergyVAD(VADConfig(frame_ms=FRAME_MS)),
                    speaking_task=None,
                    speech_task=None,
                    barge_in=asyncio.Event(),
                )
                self._streams[pkt.ssrc] = stream
                LOGGER.info("New RTP stream ssrc=%s from %s", pkt.ssrc, addr)

            stream.remote_addr = (addr[0], addr[1])
            pcm8k = ulaw_decode(pkt.payload)
            if not pcm8k.size:
                continue

            # Process in fixed 20ms frames for VAD + barge-in.
            for i in range(0, pcm8k.size, FRAME_SAMPLES_8K):
                frame = pcm8k[i : i + FRAME_SAMPLES_8K]
                if frame.size < FRAME_SAMPLES_8K:
                    break

                # Barge-in: if we detect speech while speaking, stop playback immediately.
                rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
                thr = stream.vad._threshold()  # noqa: SLF001
                if (
                    not stream.barge_in.is_set()
                    and stream.speaking_task
                    and not stream.speaking_task.done()
                    and rms >= thr
                ):
                    stream.barge_in.set()
                    stream.speaking_task.cancel()

                ended, utterance = stream.vad.push_frame(frame)
                if ended and utterance is not None and utterance.size:
                    # Cancel any ongoing response generation/speaking for this stream.
                    if stream.speech_task and not stream.speech_task.done():
                        stream.speech_task.cancel()
                    if stream.speaking_task and not stream.speaking_task.done():
                        stream.speaking_task.cancel()
                    stream.barge_in.clear()
                    stream.speech_task = asyncio.create_task(self.on_utterance(stream, utterance))

    async def on_utterance(self, stream: StreamState, pcm8k: np.ndarray) -> None:
        """Handle a user utterance (PCM16 @ 8kHz) with low-latency streaming response."""

        agent = self._get_agent()

        # Resample to 16k for Whisper.
        pcm16k = pcm16_resample(pcm8k, 8000, 16000)
        wav = pcm16_to_wav_bytes(pcm16k, 16000)

        try:
            transcript, _lang = await agent.transcribe_audio_bytes(wav, language_hint=None)
        except Exception:
            LOGGER.exception("ASR failed")
            return

        transcript = transcript.strip()
        if not transcript:
            return

        # Stream assistant text; convert to sentence-sized chunks and synthesize each chunk.
        chunk_queue: asyncio.Queue[str] = asyncio.Queue()

        async def producer() -> None:
            buf = ""
            async for tok in agent.handle_text_stream(
                conversation_id=f"pstn:{stream.in_ssrc}",
                speaker="patient",
                text=transcript,
                language="de",
                attributes={"telephony": {"ssrc": stream.in_ssrc}},
            ):
                buf += tok
                # Emit on sentence end or size.
                if any(p in buf for p in [".", "!", "?", "\n"]) or len(buf) >= 220:
                    text_chunk = buf.strip()
                    buf = ""
                    if text_chunk:
                        await chunk_queue.put(text_chunk)

            tail = buf.strip()
            if tail:
                await chunk_queue.put(tail)

            await chunk_queue.put("")  # sentinel

        async def consumer() -> None:
            while True:
                if stream.barge_in.is_set():
                    return

                chunk = await chunk_queue.get()
                if chunk == "":
                    return
                if not chunk:
                    continue

                try:
                    tts_audio = await agent._tts.synthesize(chunk, language="de")  # noqa: SLF001
                except Exception:
                    LOGGER.exception("TTS failed")
                    return

                await self._send_wav_as_rtp_ulaw(
                    stream,
                    tts_audio,
                    dst_rate=8000,
                    frame_ms=FRAME_MS,
                )

        prod_task = asyncio.create_task(producer())
        stream.speaking_task = asyncio.create_task(consumer())
        try:
            await asyncio.gather(prod_task, stream.speaking_task)
        except asyncio.CancelledError:
            prod_task.cancel()
            if stream.speaking_task:
                stream.speaking_task.cancel()
        except Exception:
            LOGGER.exception("Streaming response failed")

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
            if stream.barge_in.is_set():
                return
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
