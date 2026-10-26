"""Asks Claude whether an episode is the dog fussing or just normal noise."""

import json
from collections import deque
from dataclasses import dataclass

import anthropic

from .config import CONTEXT, FALLBACK_VOCAL_SECONDS, MODEL

SYSTEM = """\
You monitor a dog through the microphone of an infrared camera in the room where it rests and sleeps. You can't \
hear the audio. A local classifier (YAMNet, trained on Google's AudioSet) has tagged each ~1 second of it, and \
you decide from those tags whether the dog is fussing, so that its owner gets an alert.

Fussing means the dog is unsettled and wants something: sustained or repeated whining, whimpering, crying, \
barking, yipping or howling. It usually escalates or keeps coming back over tens of seconds.

These are normal and must not alert:
- settling in: shuffling, scratching, licking, a collar jingle, a thump as the dog lies down, one sigh or groan
- a single bark, yip or whimper on its own, including the short muffled noises of a dog dreaming
- snoring, heavy breathing, panting after moving
- people, TV, music or traffic, even when YAMNet gives them a dog-like tag at a low score

Reading the timeline: each line is the offset from the episode's start, the loudness in dBFS (0 is the loudest; -60 is near silence), a \
`*` when a dog-vocal class scored above the trigger threshold, and up to three top tags with scores from 0 to 1. \
The tags are noisy: a lone low-scoring "Whimper" among rustles is probably a squeak, while many seconds of \
"Whimper (dog)" or "Crying, sobbing" at 0.5 and above are whining. Recent episodes and your earlier verdicts are \
included because a dog that keeps starting up again is fussing even if each burst is short.

When unsure, prefer not to alert; the owner can always look at the camera. Write the summary as one plain \
sentence for the owner, e.g. "Whining on and off for about a minute, getting louder." or "Settling noises and \
one short bark."\
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "fussy": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "summary": {"type": "string"},
    },
    "required": ["fussy", "confidence", "summary"],
    "additionalProperties": False,
}


@dataclass
class Verdict:
    fussy: bool
    confidence: str
    summary: str
    by: str  # the model that decided, or "fallback"


class Judge:
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.history = deque(maxlen=6)  # (episode, verdict), oldest first

    def __call__(self, ep):
        try:
            verdict = self._ask(ep)
        except (anthropic.APIError, ValueError) as e:
            # The dog shouldn't go unheard because the API is down: fall back to counting vocal seconds.
            print(f"judge: {type(e).__name__}: {e}", flush=True)
            fussy = ep.vocal_seconds >= FALLBACK_VOCAL_SECONDS
            verdict = Verdict(fussy, "low", f"Claude was unavailable; {ep.vocal_seconds} seconds of dog noise "
                                            f"in {ep.duration:.0f} seconds.", "fallback")
        self.history.append((ep, verdict))
        return verdict

    def _ask(self, ep):
        parts = []
        if CONTEXT:
            parts.append(f"About this dog and room:\n{CONTEXT}")
        if self.history:
            lines = [f"- {e.start:%H:%M:%S}, {e.duration:.0f}s, {e.vocal_seconds} vocal: "
                     f"{'fussy' if v.fussy else 'not fussy'} ({v.summary})" for e, v in self.history]
            parts.append("Recent episodes (oldest first):\n" + "\n".join(lines))
        parts.append(f"New episode starting {ep.start:%A %H:%M:%S}, {ep.duration:.0f}s long, "
                     f"{ep.vocal_seconds} vocal seconds:\n{ep.timeline()}")

        response = self.client.beta.messages.create(
            model=MODEL,
            max_tokens=4000,
            system=SYSTEM,
            messages=[{"role": "user", "content": "\n\n".join(parts)}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            # On a safety decline the API retries on a fallback model it picks; unlikely here, but harmless.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason in ("refusal", "max_tokens"):
            raise ValueError(f"stop_reason {response.stop_reason} (request {response._request_id})")
        data = json.loads(next((b.text for b in response.content if b.type == "text"), ""))
        return Verdict(data["fussy"], data["confidence"], data["summary"], response.model)
