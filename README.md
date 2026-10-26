# barker

Listens to a dog through [tailcam](https://github.com/bradmartin333/tailcam)'s mic and emails when the dog is fussing: whining, crying, barking or banging at the crate door that keeps going. Normal room and dog noise, like shuffling to get comfy, a lone bark, dreaming or snoring, doesn't trigger it.

## How it works

Claude can't take audio as input, so barker turns the sound into text first:

1. **Listen.** barker stays subscribed to tailcam's `/audio/<n>` (raw s16le PCM). The mic runs on its own, so the camera stays idle and its LED stays off.
2. **Tag.** [YAMNet](https://www.kaggle.com/models/google/yamnet), Google's small AudioSet classifier, runs locally and tags each ~1 second (0.975 s) of audio, e.g. `Dog 0.74, Whimper (dog) 0.11`. Near-silent seconds are skipped.
3. **Group.** A second is active when it sets off a trigger from the [triggers file](#triggers), e.g. `Dog`, `Bark` or `Whimper` scoring at least 0.3. An episode starts there and ends after `BARKER_QUIET_GAP` seconds without one. Episodes with fewer than `BARKER_MIN_SECONDS` active seconds are dropped as too brief.
4. **Judge.** Claude gets the episode's second-by-second timeline, plus the last few episodes and its verdicts on them, and decides whether this is fussing or just noise.
5. **Alert.** A fussy verdict is emailed, with a one-line summary, the timeline and the clip attached as a `.wav`. After an email, further ones wait `BARKER_COOLDOWN` seconds.

If the Claude API can't be reached, an episode with at least `BARKER_FALLBACK_SECONDS` active seconds alerts anyway, so an outage doesn't mean silence.

The model is `claude-haiku-5-5` at low effort. Each call is a few thousand tokens at most, a fraction of a cent, and only episodes that pass step 3 make one. If Haiku misjudges your dog, set `BARKER_MODEL=claude-sonnet-5-5` or `claude-opus-5-5`.

## Run it

It runs in the homelab, and tailcam runs on a Pi (`pi-backup`) as a systemd service, so the homelab's compose file points `BARKER_SOURCE` at the Pi's port 8080 over the tailnet. That skips traefik and the token cookie. Point it at whichever camera has the mic you want: the IR camera is `/audio/2` there. The image is `bradmartin333/barker:latest`, which CI pushes to Docker Hub on every commit to `main`. The YAMNet model is baked into it.

```sh
docker run -d --name barker --restart unless-stopped --read-only --tmpfs /tmp \
  -e ANTHROPIC_API_KEY -e SMTP_USERNAME -e SMTP_PASSWORD -e SMTP_ALERT_TO \
  -e BARKER_SOURCE=http://100.69.33.61:8080/audio/2 -e TZ=America/New_York \
  -v ./triggers.toml:/etc/barker/triggers.toml:ro -e BARKER_TRIGGERS=/etc/barker/triggers.toml \
  bradmartin333/barker:latest
```

Leave out the `-v` and `BARKER_TRIGGERS` lines to use the [triggers file](#triggers) baked into the image. barker keeps nothing on disk: the alert's `.wav` is built in memory, and only `BARKER_CLIP_DIR` saves clips. `--read-only` holds it to that, and `/tmp` only holds the healthcheck's empty heartbeat file.

Set `TZ` so the timestamps in emails and in what Claude sees are local. Without `SMTP_ALERT_TO`, `SMTP_USERNAME` and `SMTP_PASSWORD`, barker only logs its verdicts (`docker logs barker`), which is a good way to watch what it would have sent while you tune the thresholds.

The healthcheck fails when no audio has arrived for a minute, e.g. when the Pi is unreachable or the camera has no mic. A refused connection is the exception: the Pi is up but tailcam isn't running, which there means the door sensor has stopped it because the dog isn't in there. barker stays healthy then and tries again every 15 seconds. The catch is that a tailcam crash-looping on the Pi also refuses connections, so barker can't tell that apart from an open door.

## Configuration

| Env var | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | | Claude API key |
| `BARKER_SOURCE` | `http://tailcam:8080/audio/0` | tailcam audio URL, or a `.wav` path |
| `BARKER_SOURCE_RATE` | `24000` | sample rate of the stream (tailcam's is 24 kHz) |
| `BARKER_TRIGGERS` | `barker/triggers.toml` | the [triggers file](#triggers) |
| `BARKER_SILENCE_DBFS` | `-60` | quieter seconds aren't classified |
| `BARKER_QUIET_GAP` | `20` | seconds without an active second that end an episode |
| `BARKER_MAX_EPISODE` | `120` | longer episodes are judged in pieces of this many seconds |
| `BARKER_MIN_SECONDS` | `3` | episodes with fewer active seconds aren't sent to Claude |
| `BARKER_MODEL` | `claude-haiku-5-5` | |
| `BARKER_CONTEXT` | | free text for Claude about the dog and the room, e.g. `Juno, 3yo lab, crated overnight. Whining around 6am usually means she needs to go out.` |
| `BARKER_FALLBACK_SECONDS` | `20` | active seconds that alert when Claude is unreachable |
| `BARKER_COOLDOWN` | `900` | minimum seconds between emails |
| `BARKER_CLIP_DIR` | | if set, every judged episode is saved here as `.wav` and `.txt` (timeline and verdict) |
| `SMTP_USERNAME`, `SMTP_PASSWORD` | | Gmail login (the homelab's `smtp/.env`) |
| `SMTP_ALERT_TO` | | where alerts go; separate several addresses with commas |
| `SMTP_HOST`, `SMTP_PORT` | `smtp.gmail.com`, `587` | STARTTLS |

## Triggers

What counts as worth flagging lives in a TOML file, not the code: [`barker/triggers.toml`](barker/triggers.toml) by default, or wherever `BARKER_TRIGGERS` points. The homelab mounts its own copy. Each `[[trigger]]` is one kind of sound:

```toml
[[trigger]]
name = "banging"     # how it's counted in logs, emails and what Claude reads
mark = "#"           # shown in the timeline on each second it fires
threshold = 0.25     # YAMNet score (0-1) at which a second counts
classes = ["Bang", "Knock", "Rattle", "Clatter"]
note = "Pawing at or rattling the crate door. Kept up for tens of seconds, the dog wants out."
```

A second is active when any of a trigger's classes scores at least its threshold. The class names must match YAMNet's 521 labels exactly (`unzip -p barker/yamnet.tflite yamnet_label_list.txt`); barker refuses to start on one it doesn't know. The note goes to Claude, so use it to say what the sound means for this dog, when it isn't fussing, and how readily to alert. For seconds of its own kind, a note takes precedence over Claude's default caution and its list of sounds that are normally fine (settling in, snoring, panting), but people, TV, music and traffic never alert whatever it says. The file is read at startup, so restart barker after editing it.

## Tuning

Before adding a trigger or trusting its threshold, see what YAMNet makes of the real sound. Record a minute of it from tailcam, e.g. while the dog paws at the crate door, then replay it with `--tags`, which prints every second's tags and the triggers it set off without judging anything, so it needs no API key:

```sh
ssh brad@pi-backup 'ffmpeg -loglevel error -f s16le -ar 24000 -ac 1 -i http://localhost:8080/audio/2 -t 60 -f wav -' > paw.wav
./run.sh --tags paw.wav
```

Pick the classes that score high while the sound is happening, and a threshold below their usual scores there but above what they score the rest of the time.

Then run it for a few nights with `BARKER_CLIP_DIR` set and email off, and listen to the saved clips next to their verdicts. Lower a trigger's threshold if its sound is missed entirely (it never shows up in the logs). Raise `BARKER_MIN_SECONDS` if too many trivial episodes reach Claude. Use `BARKER_CONTEXT` and the triggers' notes to tell Claude what's normal for this dog and how readily to alert; both take precedence over its default caution.

## Local dev

```sh
./run.sh path/to/recording.wav         # replay a 16-bit .wav through the whole pipeline, then exit
./run.sh --tags path/to/recording.wav  # print each second's tags and triggers, judging nothing
./run.sh                               # listen to BARKER_SOURCE
```

`run.sh` loads settings from a `.env` beside it. The first run downloads the YAMNet model (4 MB, checksummed) to `barker/yamnet.tflite`.
