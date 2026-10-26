import os


def _float(name, default):
    return float(os.environ.get(name, default))


# tailcam's /audio/<n>: raw mono s16le PCM. On the homelab's proxy network tailcam answers on its container name,
# so this skips traefik and the token cookie. A local .wav path works too, for trying out a recording.
SOURCE = os.environ.get("BARKER_SOURCE", "http://tailcam:8080/audio/0")
SOURCE_RATE = int(os.environ.get("BARKER_SOURCE_RATE", "24000"))  # tailcam's AUDIO_RATE

# A second of audio counts as the dog vocalising when YAMNet scores any of VOCAL_CLASSES at least this high.
VOCAL_THRESHOLD = _float("BARKER_VOCAL_THRESHOLD", "0.3")
# Below this level a second is treated as silence and not classified at all.
SILENCE_DBFS = _float("BARKER_SILENCE_DBFS", "-60")
# An episode starts at the first vocal second and ends after QUIET_GAP seconds without one, or at MAX_EPISODE.
QUIET_GAP = _float("BARKER_QUIET_GAP", "20")
MAX_EPISODE = _float("BARKER_MAX_EPISODE", "120")
# Episodes with fewer vocal seconds than this (a lone bark, a yip in a dream) never reach Claude.
MIN_VOCAL_SECONDS = int(os.environ.get("BARKER_MIN_VOCAL_SECONDS", "3"))

MODEL = os.environ.get("BARKER_MODEL", "claude-opus-5-5")
# Free text about the dog and the room, passed to Claude, e.g. "Juno, 3yo lab, crated overnight. Whining around 6am
# usually means she needs to go out."
CONTEXT = os.environ.get("BARKER_CONTEXT", "").strip()
# If Claude can't be reached, alert anyway when an episode has at least this many vocal seconds.
FALLBACK_VOCAL_SECONDS = int(os.environ.get("BARKER_FALLBACK_VOCAL_SECONDS", "20"))

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
# Touched for every second of audio; the Docker healthcheck fails when it goes stale.
HEARTBEAT = os.environ.get("BARKER_HEARTBEAT", "/tmp/barker-heartbeat")
MODEL_PATH = os.environ.get("BARKER_YAMNET", os.path.join(os.path.dirname(__file__), "yamnet.tflite"))
