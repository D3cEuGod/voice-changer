"""Recognise the eval outputs with SenseVoice and score them against the true sentences.

Run with .run\\export-env\\Scripts\\python.exe from the project root:  <this> <tag> [<tag> ...]
Use the tag "inputs" to score the unconverted eval clips (the recogniser's own floor).
"""
import sys
from collections import defaultdict
from math import gcd
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path.cwd()
folder = next(x for x in (ROOT / ".run").glob("sherpa-onnx-sense-voice-*") if x.is_dir())
model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
    str(folder / "model.int8.onnx"), str(folder / "tokens.txt"), num_threads=4, language="zh", use_itn=False)
DIGITS = str.maketrans("0123456789", "零一二三四五六七八九")
truth = ["".join(ch for ch in line if ch.isalnum())
         for line in (ROOT / ".run/evalset/sentences.txt").read_text(encoding="utf-8").splitlines() if line.strip()]


def recognise(path):
    samples, rate = sf.read(str(path), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    g = gcd(rate, 16000)
    stream = model.create_stream()
    stream.accept_waveform(16000, resample_poly(samples, 16000 // g, rate // g).astype(np.float32))
    model.decode_stream(stream)
    return "".join(ch for ch in stream.result.text if ch.isalnum()).translate(DIGITS)


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


verbose = "-v" in sys.argv
for tag in [a for a in sys.argv[1:] if a != "-v"]:
    errors, chars = defaultdict(int), defaultdict(int)
    if tag == "inputs":
        files = [(f"{'female' if p.stem[0] == 'f' else 'male'}-input-{p.stem[1:]}", p)
                 for p in sorted((ROOT / ".run/evalset").glob("[fm][0-9].wav"))]
    else:
        files = [(p.stem, p) for p in sorted((ROOT / ".run/eval" / tag).glob("*-*-[0-9].wav"))]
    for stem, path in files:
        name, voice, index = stem.rsplit("-", 2)
        expected = truth[int(index) - 1]
        text = recognise(path)
        wrong = distance(expected, text)
        for key in (voice, name, f"{name}/{voice}", "ALL"):
            errors[key] += wrong
            chars[key] += len(expected)
        if verbose and wrong:
            print(f"   {stem}: {wrong}/{len(expected)}  {text}", flush=True)
    order = ["ALL"] + sorted(k for k in errors if k != "ALL" and "/" not in k) + sorted(k for k in errors if "/" in k)
    print(f"{tag} CER " + "  ".join(f"{k} {errors[k] / max(chars[k], 1) * 100:.1f}%" for k in order), flush=True)
