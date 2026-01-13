from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class VADConfig:
    frame_ms: int = 20
    start_frames: int = 3
    end_frames: int = 12
    preroll_frames: int = 8
    min_utterance_frames: int = 8
    min_rms: float = 250.0
    threshold_mult: float = 3.0
    noise_alpha: float = 0.05


class EnergyVAD:
    """A tiny energy-based VAD suitable for telephone audio.

    It adapts to a per-stream noise floor and emits utterance boundaries.
    """

    def __init__(self, cfg: VADConfig | None = None) -> None:
        self.cfg = cfg or VADConfig()
        self._noise_rms = 0.0
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self._preroll: list[np.ndarray] = []
        self._utterance: list[np.ndarray] = []

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def _rms(self, frame: np.ndarray) -> float:
        if frame.size == 0:
            return 0.0
        x = frame.astype(np.float32)
        return float(np.sqrt(np.mean(x * x)))

    def _threshold(self) -> float:
        base = max(self.cfg.min_rms, self._noise_rms * self.cfg.threshold_mult)
        return base

    def push_frame(self, frame: np.ndarray) -> tuple[bool, np.ndarray | None]:
        """Push a PCM16 frame.

        Returns:
            (ended, utterance_pcm) where ended indicates end-of-utterance.
        """

        rms = self._rms(frame)
        thr = self._threshold()
        is_voice = rms >= thr

        if not self._in_speech:
            # Update noise floor slowly using non-voice frames.
            if not is_voice:
                self._noise_rms = (1 - self.cfg.noise_alpha) * self._noise_rms + self.cfg.noise_alpha * rms

            self._preroll.append(frame)
            if len(self._preroll) > self.cfg.preroll_frames:
                self._preroll.pop(0)

            if is_voice:
                self._speech_run += 1
                self._silence_run = 0
            else:
                self._speech_run = 0

            if self._speech_run >= self.cfg.start_frames:
                self._in_speech = True
                self._utterance = list(self._preroll)
                self._preroll.clear()
                self._silence_run = 0

            return False, None

        # in speech
        self._utterance.append(frame)

        if is_voice:
            self._silence_run = 0
        else:
            self._silence_run += 1

        if self._silence_run >= self.cfg.end_frames:
            utterance = np.concatenate(self._utterance) if self._utterance else np.array([], dtype=np.int16)
            self._utterance.clear()
            self._in_speech = False
            self._speech_run = 0
            self._silence_run = 0

            if utterance.size < (self.cfg.min_utterance_frames * frame.size):
                return False, None

            return True, utterance

        return False, None
