"""YAMNet, Google's AudioSet sound classifier: 0.975 s of 16 kHz audio in, 521 class scores out."""

import hashlib
import os
import sys
import urllib.request
import zipfile

import numpy as np

from .config import MODEL_PATH

RATE = 16000
FRAME = 15600  # samples per classification, fixed by the model
URL = ("https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/"
       "audio_classification/android/lite-model_yamnet_classification_tflite_1.tflite")
SHA256 = "10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de"

# Classes that mean the dog is making noise. The specific ones score low on this model: a puppy whining gets
# "Dog" at 0.5-0.75 but "Whimper (dog)" only around 0.1, so "Dog" itself has to count. Plain Whimper and Crying
# catch whines the dog-specific class misses. Telling settling noises from fussing is left to Claude.
VOCAL_CLASSES = ("Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)", "Whimper", "Crying, sobbing")
# Parents in AudioSet's hierarchy that score alongside their children and would crowd them out of the top labels.
GENERIC = {"Animal", "Domestic animals, pets", "Canidae, dogs, wolves", "Wild animals",
           "Livestock, farm animals, working animals"}


def ensure_model(path=MODEL_PATH):
    """Download the model if it's missing; the Docker build calls this so the image ships with it."""
    if not os.path.exists(path):
        print(f"downloading YAMNet to {path}", flush=True)
        data = urllib.request.urlopen(URL, timeout=60).read()
        if hashlib.sha256(data).hexdigest() != SHA256:
            sys.exit("YAMNet download failed its checksum")
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)
    return path


class YAMNet:
    def __init__(self, path=MODEL_PATH):
        from ai_edge_litert.interpreter import Interpreter

        path = ensure_model(path)
        self.interpreter = Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]["index"]
        self.output = self.interpreter.get_output_details()[0]["index"]
        # The class names ride along inside the .tflite, which is also a zip.
        with zipfile.ZipFile(path) as z:
            self.labels = z.read("yamnet_label_list.txt").decode().splitlines()
        self.vocal = [self.labels.index(c) for c in VOCAL_CLASSES]

    def scores(self, frame):
        """Class scores (0-1, independent, not a softmax) for FRAME float32 samples in [-1, 1]."""
        self.interpreter.set_tensor(self.input, frame.astype(np.float32))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output)[0]

    def top(self, scores, n=3, floor=0.1):
        """[(label, score)] for the n best specific classes scoring at least floor, plus the best dog-specific
        class (Bark, Whimper, ...) when it's at least 0.05, since that's what tells a whine from a bark."""
        best = [i for i in np.argsort(scores)[::-1] if self.labels[i] not in GENERIC][:n]
        out = [i for i in best if scores[i] >= floor]
        specific = max(self.vocal[1:], key=lambda i: scores[i])
        if specific not in out and scores[specific] >= 0.05:
            out.append(specific)
        return [(self.labels[i], float(scores[i])) for i in out]

    def vocal_score(self, scores):
        return float(scores[self.vocal].max())


if __name__ == "__main__":
    ensure_model()
