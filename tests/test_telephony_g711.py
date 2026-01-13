from __future__ import annotations

import numpy as np

from telephony.g711 import ulaw_decode, ulaw_encode


def test_ulaw_encode_decode_shape_and_types() -> None:
    # 20ms of 8kHz samples
    pcm = (np.sin(np.linspace(0, 2 * np.pi, 160, endpoint=False)) * 12000).astype(np.int16)

    ulaw = ulaw_encode(pcm)
    assert isinstance(ulaw, bytes | bytearray)
    assert len(ulaw) == pcm.size

    decoded = ulaw_decode(ulaw)
    assert decoded.dtype == np.int16
    assert decoded.shape == pcm.shape


def test_ulaw_encode_decode_silence_is_stable() -> None:
    pcm = np.zeros(320, dtype=np.int16)
    ulaw = ulaw_encode(pcm)
    decoded = ulaw_decode(ulaw)

    # Mu-law isn't perfectly invertible, but silence should stay near zero.
    assert int(np.max(np.abs(decoded))) <= 200
