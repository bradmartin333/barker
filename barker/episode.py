"""Groups seconds of audio into episodes: a run of seconds that set off a trigger, bounded by quiet."""

import io
import wave
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .config import MAX_EPISODE, QUIET_GAP, SILENCE_DBFS, TRIGGERS
from .yamnet import FRAME, RATE

STEP = FRAME / RATE  # seconds per frame (0.975)
PREROLL = 2  # frames kept from before the first active second, so the clip doesn't start mid-whine
TAIL = 3  # quiet frames kept after the last active second


@dataclass
class Second:
    dbfs: float
    top: list  # [(label, score)]; empty for silence
    hits: tuple = ()  # names of the triggers this second set off

    def line(self, offset):
        """The second as one line of the timeline Claude reads: a column per trigger holds its mark if it fired."""
        marks = "".join(t.mark if t.name in self.hits else " " for t in TRIGGERS)
        labels = ", ".join(f"{name} {score:.2f}" for name, score in self.top)
        labels = labels or ("silence" if self.dbfs < SILENCE_DBFS else "nothing recognisable")
        return f"+{offset:5.1f}s  {self.dbfs:6.1f} dBFS  {marks} {labels}"


@dataclass
class Episode:
    start: datetime
    seconds: list = field(default_factory=list)
    audio: list = field(default_factory=list)

    @property
    def active_seconds(self):
        return sum(bool(s.hits) for s in self.seconds)

    @property
    def duration(self):
        return len(self.seconds) * STEP

    def tally(self):
        """Seconds per trigger, e.g. "dog noise 12s, banging 4s"."""
        counts = [(t.name, sum(t.name in s.hits for s in self.seconds)) for t in TRIGGERS]
        return ", ".join(f"{name} {n}s" for name, n in counts if n) or "nothing flagged"

    def timeline(self):
        return "\n".join(s.line(i * STEP) for i, s in enumerate(self.seconds))

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
        self.quiet = 0  # frames since the last active second

    def push(self, when, frame, second):
        """Feed one frame; returns an Episode when one finishes."""
        if self.episode is None:
            if not second.hits:
                self.recent.append((when, frame, second))
                return None
            start = self.recent[0][0] if self.recent else when
            self.episode = Episode(start)
            for _, f, s in self.recent:
                self._add(f, s)
            self.recent.clear()
        self._add(frame, second)
        self.quiet = 0 if second.hits else self.quiet + 1
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
