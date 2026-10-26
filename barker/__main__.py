import json
import os
import queue
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime, timedelta

import numpy as np

from . import alert
from .config import CLIP_DIR, COOLDOWN, HEARTBEAT, MIN_VOCAL_SECONDS, SILENCE_DBFS, SOURCE, SOURCE_RATE
from .episode import STEP, Second, Tracker
from .judge import Judge
from .stream import OFF, frames
from .yamnet import YAMNet


def beat():
    """Touch the heartbeat file; the Docker healthcheck fails once it's a minute stale."""
    with open(HEARTBEAT, "w"):
        pass


def save_clip(ep, verdict):
    os.makedirs(CLIP_DIR, exist_ok=True)
    base = os.path.join(CLIP_DIR, f"{ep.start:%Y%m%d-%H%M%S}")
    with open(base + ".wav", "wb") as f:
        f.write(ep.wav())
    with open(base + ".txt", "w") as f:
        f.write(f"{json.dumps(asdict(verdict))}\n\n{ep.timeline()}\n")


def judge_episodes(episodes):
    """Worker thread: Claude can take several seconds, and the audio reader mustn't stall while it thinks."""
    judge = Judge()
    last_alert = float("-inf")
    while True:
        ep = episodes.get()
        verdict = judge(ep)
        print(f"{ep.start:%H:%M:%S} {ep.duration:.0f}s {ep.vocal_seconds} vocal -> "
              f"{'FUSSY' if verdict.fussy else 'fine'} ({verdict.confidence}, {verdict.by}): {verdict.summary}",
              flush=True)
        if CLIP_DIR:
            save_clip(ep, verdict)
        if verdict.fussy and alert.ENABLED:
            if time.monotonic() - last_alert < COOLDOWN:
                print("  alert skipped: still in cooldown", flush=True)
            else:
                try:
                    alert.send(ep, verdict)
                    last_alert = time.monotonic()
                    print("  alert sent", flush=True)
                except OSError as e:  # smtplib's errors are OSErrors too
                    print(f"  alert failed: {e}", flush=True)
        episodes.task_done()


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else SOURCE
    live = not os.path.exists(source)
    yamnet = YAMNet()
    tracker = Tracker()
    episodes = queue.Queue()
    threading.Thread(target=judge_episodes, args=(episodes,), daemon=True).start()
    if not alert.ENABLED:
        print("SMTP_ALERT_TO, SMTP_USERNAME or SMTP_PASSWORD unset: logging verdicts only", flush=True)

    def finished(ep):
        if ep is None:
            return
        if ep.vocal_seconds < MIN_VOCAL_SECONDS:
            print(f"{ep.start:%H:%M:%S} {ep.duration:.0f}s {ep.vocal_seconds} vocal -> too brief, not judged",
                  flush=True)
        else:
            episodes.put(ep)

    # A file is replayed as if it started now, so timestamps still read like a real night.
    t0 = datetime.now().astimezone()
    for i, frame in enumerate(frames(source, SOURCE_RATE)):
        if frame is None or frame is OFF:  # the stream dropped, or tailcam is off
            finished(tracker.flush())
            if frame is OFF:  # on purpose (see stream.OFF), so still healthy
                beat()
            continue
        beat()
        when = datetime.now().astimezone() - timedelta(seconds=STEP) if live else t0 + timedelta(seconds=i * STEP)
        dbfs = 20 * np.log10(np.sqrt(np.mean(frame ** 2)) + 1e-10)
        if dbfs < SILENCE_DBFS:
            second = Second(float(dbfs), 0.0, [])
        else:
            scores = yamnet.scores(frame)
            second = Second(float(dbfs), yamnet.vocal_score(scores), yamnet.top(scores))
        finished(tracker.push(when, frame, second))
    finished(tracker.flush())
    episodes.join()


if __name__ == "__main__":
    main()
