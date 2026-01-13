from __future__ import annotations

import numpy as np

from telephony.vad import EnergyVAD, VADConfig


def test_energy_vad_detects_utterance_end() -> None:
    cfg = VADConfig(
        start_frames=2,
        end_frames=3,
        preroll_frames=2,
        min_utterance_frames=2,
        min_rms=50.0,
        threshold_mult=2.0,
        noise_alpha=0.2,
    )
    vad = EnergyVAD(cfg)

    silence = np.zeros(160, dtype=np.int16)
    voice = (np.ones(160, dtype=np.int16) * 2000).astype(np.int16)

    # Prime noise floor with silence.
    for _ in range(5):
        ended, utt = vad.push_frame(silence)
        assert ended is False
        assert utt is None

    # Start speech.
    ended, utt = vad.push_frame(voice)
    assert ended is False
    assert utt is None

    ended, utt = vad.push_frame(voice)
    assert ended is False
    assert utt is None
    assert vad.in_speech is True

    # End speech with enough silence frames.
    for _ in range(2):
        ended, utt = vad.push_frame(silence)
        assert ended is False
        assert utt is None

    ended, utt = vad.push_frame(silence)
    assert ended is True
    assert utt is not None
    assert utt.dtype == np.int16
    assert utt.size > 0
