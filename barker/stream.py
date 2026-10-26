"""Audio in: YAMNet-sized frames of 16 kHz float32, from tailcam's PCM stream or a .wav file."""

import math
import os
import time
import urllib.request
import wave

import numpy as np
from scipy.signal import resample_poly

from .yamnet import FRAME, RATE

# Yielded in place of a frame while tailcam refuses the connection: the host is up but tailcam isn't running, which
# on the Pi means its door sensor has it stopped because the dog isn't in there. That's not a fault.
OFF = object()
# Seconds between tries while tailcam is off: soon enough to catch the dog going back in, and well inside the
# healthcheck's 60 s.
OFF_RETRY = 15


def _resampler(rate):
    """(source samples per frame, function taking them to FRAME samples at RATE)."""
    g = math.gcd(rate, RATE)
    up, down = RATE // g, rate // g
    n = round(FRAME * rate / RATE)

    def convert(x):
        y = resample_poly(x, up, down) if rate != RATE else x
        return np.pad(y, (0, max(0, FRAME - len(y))))[:FRAME].astype(np.float32)

    return n, convert


def _pcm(buf):
    return np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32768


def from_url(url, rate):
    """Frames from a raw mono s16le stream, forever. Reconnects when tailcam restarts or the mic goes away, and
    yields None after each drop so an episode in progress isn't left open across the gap. While the connection is
    refused, yields OFF every OFF_RETRY seconds instead."""
    n, convert = _resampler(rate)
    backoff = 5
    off = False
    while True:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                print(f"listening to {url}", flush=True)
                backoff = 5
                off = False
                while len(buf := resp.read(2 * n)) == 2 * n:
                    yield convert(_pcm(buf))
            print(f"{url} ended", flush=True)
        except OSError as e:
            # urlopen wraps connect errors in a URLError, whose reason is the original.
            if isinstance(getattr(e, "reason", e), ConnectionRefusedError):
                if not off:  # once, not every OFF_RETRY seconds for as long as the door is open
                    print(f"{url}: refused, so tailcam is off; trying every {OFF_RETRY}s", flush=True)
                off = True
                yield OFF
                time.sleep(OFF_RETRY)
                continue
            print(f"{url}: {e}", flush=True)
        off = False
        yield None
        time.sleep(backoff)
        backoff = min(backoff * 2, 60)


def from_wav(path):
    """Frames from a 16-bit PCM .wav file (any rate, mixed down to mono), then stops."""
    with wave.open(path) as w:
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: only 16-bit PCM .wav files are supported")
        x = _pcm(w.readframes(w.getnframes())).reshape(-1, w.getnchannels()).mean(axis=1)
        rate = w.getframerate()
    n, convert = _resampler(rate)
    for i in range(0, len(x), n):
        yield convert(x[i:i + n])


def frames(source, rate):
    return from_wav(source) if os.path.exists(source) else from_url(source, rate)
