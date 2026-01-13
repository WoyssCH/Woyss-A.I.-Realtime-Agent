from __future__ import annotations

import numpy as np


def ulaw_decode(ulaw_bytes: bytes) -> np.ndarray:
    """Decode G.711 mu-law bytes to PCM16 int16 array."""

    # Reuse the proven implementation from Twilio Media Streams utilities.
    from integrations.twilio_streaming import mulaw_decode

    return mulaw_decode(ulaw_bytes)


def ulaw_encode(pcm16: np.ndarray) -> bytes:
    """Encode PCM16 int16 array to G.711 mu-law bytes.

    This is a vectorized mu-law encoder suitable for realtime packetization.
    """

    if pcm16.size == 0:
        return b""

    x = pcm16.astype(np.int32)
    sign = (x < 0).astype(np.int32)
    x = np.abs(x)

    # Mu-law companding constants (G.711): bias=33, clip=32635.
    x = np.minimum(x, 32635)
    x = x + 33

    # Find exponent and mantissa.
    exponent = np.zeros_like(x)
    for exp in range(8):
        exponent = np.where(x >= (1 << (exp + 7)), exp, exponent)

    mantissa = (x >> (exponent + 3)) & 0x0F

    ulaw = np.bitwise_not((sign << 7) | (exponent << 4) | mantissa).astype(np.uint8)
    return ulaw.tobytes()
