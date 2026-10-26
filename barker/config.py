import os
import tomllib
from dataclasses import dataclass


def _float(name, default):
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Trigger:
    """A kind of sound worth flagging; see triggers.toml."""
    name: str
    mark: str
    threshold: float
    classes: tuple
    note: str = ""


def load_triggers(path):
    """The [[trigger]] entries in a TOML file. Exits with the reason if one is malformed. Whether the classes are real
    YAMNet classes is checked when the model loads, since that's where the label list is."""
    with open(path, "rb") as f:
        entries = tomllib.load(f).get("trigger", [])
    triggers = []
    for e in entries:
        try:
            t = Trigger(e["name"], e["mark"], float(e["threshold"]), tuple(e["classes"]),
                        " ".join(e.get("note", "").split()))
        except (KeyError, TypeError, ValueError) as err:
            raise SystemExit(f"{path}: [[trigger]] {e.get('name', '')!r} is missing or has a bad {err}")
        if len(t.mark) != 1 or not t.classes or not 0 < t.threshold <= 1:
            raise SystemExit(f"{path}: trigger {t.name!r} needs a one-character mark, at least one class and a "
                             f"threshold above 0 and at most 1")
        triggers.append(t)
    if not triggers:
        raise SystemExit(f"{path}: no [[trigger]] entries, so nothing would ever be flagged")
    for field in ("name", "mark"):
        values = [getattr(t, field) for t in triggers]
        if len(set(values)) < len(values):
            raise SystemExit(f"{path}: two triggers share a {field}")
    return tuple(triggers)


# tailcam's /audio/<n>: raw mono s16le PCM, fetched directly rather than through traefik and the token cookie. The
# homelab sets this to tailcam on the Pi, over the tailnet. A local .wav path works too, for trying out a recording.
SOURCE = os.environ.get("BARKER_SOURCE", "http://tailcam:8080/audio/0")
SOURCE_RATE = int(os.environ.get("BARKER_SOURCE_RATE", "24000"))  # tailcam's AUDIO_RATE

# The kinds of sound worth flagging, each a set of YAMNet classes and a threshold.
TRIGGERS = load_triggers(os.environ.get("BARKER_TRIGGERS", os.path.join(os.path.dirname(__file__), "triggers.toml")))
# Below this level a second is treated as silence and not classified at all.
SILENCE_DBFS = _float("BARKER_SILENCE_DBFS", "-60")
# An episode starts at the first second that sets off a trigger (an active second) and ends after QUIET_GAP seconds
# without one, or at MAX_EPISODE.
QUIET_GAP = _float("BARKER_QUIET_GAP", "20")
MAX_EPISODE = _float("BARKER_MAX_EPISODE", "120")
# Episodes with fewer active seconds than this (a lone bark, a yip in a dream, a thump lying down) never reach Claude.
MIN_SECONDS = int(os.environ.get("BARKER_MIN_SECONDS", "3"))

# Reading a tag timeline is a small classification job, so Haiku: a judged episode costs a fraction of a cent.
MODEL = os.environ.get("BARKER_MODEL", "claude-haiku-5-5")
# Free text about the dog and the room, passed to Claude, e.g. "Juno, 3yo lab, crated overnight. Whining around 6am
# usually means she needs to go out."
CONTEXT = os.environ.get("BARKER_CONTEXT", "").strip()
# If Claude can't be reached, alert anyway when an episode has at least this many active seconds.
FALLBACK_SECONDS = int(os.environ.get("BARKER_FALLBACK_SECONDS", "20"))

# Shared Gmail login from the homelab's smtp/.env. With no SMTP_ALERT_TO, verdicts are only logged.
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_ALERT_TO = os.environ.get("SMTP_ALERT_TO", "")
# Minimum seconds between alert emails, so one long fuss doesn't send a dozen.
COOLDOWN = _float("BARKER_COOLDOWN", "900")

# If set, every judged episode is saved here (wav + timeline + verdict) for tuning the thresholds.
CLIP_DIR = os.environ.get("BARKER_CLIP_DIR", "")
# Touched for every second of audio, and while tailcam is off; the Docker healthcheck fails when it goes stale.
HEARTBEAT = os.environ.get("BARKER_HEARTBEAT", "/tmp/barker-heartbeat")
MODEL_PATH = os.environ.get("BARKER_YAMNET", os.path.join(os.path.dirname(__file__), "yamnet.tflite"))
