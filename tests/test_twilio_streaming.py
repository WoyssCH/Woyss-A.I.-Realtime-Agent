from __future__ import annotations

import base64

import numpy as np

from integrations.twilio_streaming import mulaw_decode, pcm16_resample


def test_mulaw_decode_silence_is_near_zero():
    # Common mu-law 'silence' byte is 0xFF.
    pcm = mulaw_decode(b"\xFF" * 160)
    assert pcm.dtype == np.int16
    assert abs(int(pcm.mean())) < 200


def test_resample_changes_length():
    pcm = np.zeros(8000, dtype=np.int16)
    out = pcm16_resample(pcm, 8000, 16000)
    assert out.shape[0] == 16000


def test_mulaw_decode_accepts_base64_payload():
    payload = base64.b64encode(b"\xFF" * 10).decode("ascii")
    pcm = mulaw_decode(base64.b64decode(payload))
    assert pcm.shape[0] == 10
