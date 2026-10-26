# barker

Listens to a dog through [tailcam](https://github.com/bradmartin333/tailcam)'s mic and emails when the dog is fussing: whining, crying or barking that keeps going. Normal room and dog noise, like shuffling to get comfy, a lone bark, dreaming or snoring, doesn't trigger it.

## How it works

Claude can't take audio as input, so barker turns the sound into text first:

1. **Listen.** barker stays subscribed to tailcam's `/audio/<n>` (raw s16le PCM). The mic runs on its own, so the camera stays idle and its LED stays off.
2. **Tag.** [YAMNet](https://www.kaggle.com/models/google/yamnet), Google's small AudioSet classifier, runs locally and tags each ~1 second (0.975 s) of audio, e.g. `Dog 0.74, Whimper (dog) 0.11`. Near-silent seconds are skipped.
3. **Group.** A second counts as the dog vocalising when `Dog`, `Bark`, `Whimper`, `Howl` or a similar class scores at least `BARKER_VOCAL_THRESHOLD`. An episode starts there and ends after `BARKER_QUIET_GAP` seconds without one. Episodes with fewer than `BARKER_MIN_VOCAL_SECONDS` vocal seconds are dropped as too brief.
4. **Judge.** Claude gets the episode's second-by-second timeline, plus the last few episodes and its verdicts on them, and decides whether this is fussing or just noise.
5. **Alert.** A fussy verdict is emailed, with a one-line summary, the timeline and the clip attached as a `.wav`. After an email, further ones wait `BARKER_COOLDOWN` seconds.

If the Claude API can't be reached, an episode with at least `BARKER_FALLBACK_VOCAL_SECONDS` vocal seconds alerts anyway, so an outage doesn't mean silence.

The model is `claude-haiku-5-5` at low effort. Each call is a few thousand tokens at most, a fraction of a cent, and only episodes that pass step 3 make one. If Haiku misjudges your dog, set `BARKER_MODEL=claude-sonnet-5-5` or `claude-opus-5-5`.

## Run it

It runs in the homelab next to tailcam, on the same `proxy` network, so `BARKER_SOURCE` can reach `http://tailcam:8080/audio/0` directly without going through traefik. The image is `bradmartin333/barker:latest`, which CI pushes to Docker Hub on every commit to `main`. The YAMNet model is baked into it.

```sh
docker run -d --name barker --restart unless-stopped --network proxy \
  -e ANTHROPIC_API_KEY -e SMTP_USERNAME -e SMTP_PASSWORD -e SMTP_ALERT_TO \
  -e TZ=America/New_York \
  bradmartin333/barker:latest
```

Set `TZ` so the timestamps in emails and in what Claude sees are local. Without `SMTP_ALERT_TO`, `SMTP_USERNAME` and `SMTP_PASSWORD`, barker only logs its verdicts (`docker logs barker`), which is a good way to watch what it would have sent while you tune the thresholds.

The healthcheck fails when no audio has arrived for a minute, e.g. when tailcam is down or the camera has no mic.

## Configuration

| Env var | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | | Claude API key |
| `BARKER_SOURCE` | `http://tailcam:8080/audio/0` | tailcam audio URL, or a `.wav` path |
| `BARKER_SOURCE_RATE` | `24000` | sample rate of the stream (tailcam's is 24 kHz) |
| `BARKER_VOCAL_THRESHOLD` | `0.3` | YAMNet score (0–1) at which a second counts as the dog vocalising |
| `BARKER_SILENCE_DBFS` | `-60` | quieter seconds aren't classified |
| `BARKER_QUIET_GAP` | `20` | seconds without vocalising that end an episode |
| `BARKER_MAX_EPISODE` | `120` | longer episodes are judged in pieces of this many seconds |
| `BARKER_MIN_VOCAL_SECONDS` | `3` | shorter episodes aren't sent to Claude |
| `BARKER_MODEL` | `claude-haiku-5-5` | |
| `BARKER_CONTEXT` | | free text for Claude about the dog and the room, e.g. `Juno, 3yo lab, crated overnight. Whining around 6am usually means she needs to go out.` |
| `BARKER_FALLBACK_VOCAL_SECONDS` | `20` | vocal seconds that alert when Claude is unreachable |
| `BARKER_COOLDOWN` | `900` | minimum seconds between emails |
| `BARKER_CLIP_DIR` | | if set, every judged episode is saved here as `.wav` and `.txt` (timeline and verdict) |
| `SMTP_USERNAME`, `SMTP_PASSWORD` | | Gmail login (the homelab's `smtp/.env`) |
| `SMTP_ALERT_TO` | | where alerts go |
| `SMTP_HOST`, `SMTP_PORT` | `smtp.gmail.com`, `587` | STARTTLS |

## Tuning

Run it for a few nights with `BARKER_CLIP_DIR` set and email off, then listen to the saved clips next to their verdicts. Lower `BARKER_VOCAL_THRESHOLD` if quiet whining is missed entirely (it never shows up in the logs). Raise `BARKER_MIN_VOCAL_SECONDS` if too many trivial episodes reach Claude. Use `BARKER_CONTEXT` to tell Claude what's normal for this dog.

## Local dev

```sh
./run.sh path/to/recording.wav     # replay a 16-bit .wav through the whole pipeline, then exit
./run.sh                           # listen to BARKER_SOURCE
```

`run.sh` loads settings from a `.env` beside it. The first run downloads the YAMNet model (4 MB, checksummed) to `barker/yamnet.tflite`.
