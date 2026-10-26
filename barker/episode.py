"""Groups seconds of audio into episodes: a run of dog vocalising, bounded by quiet."""

import io
import wave
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .config import MAX_EPISODE, QUIET_GAP, SILENCE_DBFS, VOCAL_THRESHOLD
from .yamnet import FRAME, RATE

STEP = FRAME / RATE  # seconds per frame (0.975)
PREROLL = 2  # frames kept from before the first vocal second, so the clip doesn't start mid-whine
TAIL = 3  # quiet frames kept after the last vocal second


@dataclass
class Second:
    dbfs: float
    vocal: float  # best score among the vocal classes
    top: list  # [(label, score)]; empty for silence

    @property
    def is_vocal(self):
        return self.vocal >= VOCAL_THRESHOLD


@dataclass
class Episode:
    start: datetime
    seconds: list = field(default_factory=list)
    audio: list = field(default_factory=list)

    @property
    def vocal_seconds(self):
        return sum(s.is_vocal for s in self.seconds)

    @property
    def duration(self):
        return len(self.seconds) * STEP

    def timeline(self):
        """One line per second, as Claude reads it."""
        lines = []
        for i, s in enumerate(self.seconds):
            labels = ", ".join(f"{name} {score:.2f}" for name, score in s.top)
            labels = labels or ("silence" if s.dbfs < SILENCE_DBFS else "nothing recognisable")
            lines.append(f"+{i * STEP:5.1f}s  {s.dbfs:6.1f} dBFS  {'*' if s.is_vocal else ' '} {labels}")
        return "\n".join(lines)

    def wav(self):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            pcm = np.clip(np.concatenate(self.audio), -1, 1)
            w.writeframes((pcm * 32767).astype("<i2").tobytes())
        return buf.getvalue()


class Tracker:
    def __init__(self):
        self.recent = deque(maxlen=PREROLL)  # (time, frame, second) before an episode starts
        self.episode = None
        self.quiet = 0  # frames since the last vocal second

    def push(self, when, frame, second):
        """Feed one frame; returns an Episode when one finishes."""
        if self.episode is None:
            if not second.is_vocal:
                self.recent.append((when, frame, second))
                return None
            start = self.recent[0][0] if self.recent else when
            self.episode = Episode(start)
            for _, f, s in self.recent:
                self._add(f, s)
            self.recent.clear()
        self._add(frame, second)
        self.quiet = 0 if second.is_vocal else self.quiet + 1
        if self.quiet * STEP >= QUIET_GAP or self.episode.duration >= MAX_EPISODE:
            return self.flush()
        return None

    def flush(self):
        """End the current episode, if any, trimming the quiet beyond TAIL frames."""
        ep, self.episode = self.episode, None
        if ep is None:
            return None
        cut = len(ep.seconds) - max(0, self.quiet - TAIL)
        ep.seconds, ep.audio = ep.seconds[:cut], ep.audio[:cut]
        self.quiet = 0
        return ep

    def _add(self, frame, second):
        self.episode.seconds.append(second)
        self.episode.audio.append(frame)
