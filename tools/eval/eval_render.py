"""Render the eval set through the backend's live path in-process and report objective metrics.

Run with the backend venv from the project root:
    .venv\\Scripts\\python.exe <this> <tag> [voice ...]
Environment:
    EVAL_FRAME_MS=120                  test a shorter streaming packet (default 200)
    EVAL_SET=character.NAME=value;character.TABLE.voice=value;...
                                       set module-level tunables (values parsed as Python literals)
Outputs go to .run/eval/<tag>/. Nothing here talks to the running server on 8765.
"""
import ast
import importlib
import json
import os
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from backend import engine  # noqa: E402
from backend.live import SHORT_FRAME, SPEECH_LOOKAHEAD, LiveVoice  # noqa: E402

RATE = engine.RATE
RENDER_FRAME = round(RATE * int(os.environ.get("EVAL_FRAME_MS", "200")) / 1000)
EVALSET = ROOT / ".run/evalset"
GAP = int(0.7 * RATE)


def load(path):
    return engine.decode(Path(path).read_bytes()).astype(np.float32)


def build_clips():
    """One long clip per condition, plus the sample range of each sentence."""
    clips, ranges = {}, {}
    for prefix, name in (("f", "female"), ("m", "male")):
        parts, spans, position = [np.zeros(GAP, np.float32)], [], GAP
        for index in range(1, 9):
            sentence = load(EVALSET / f"{prefix}{index}.wav")
            spans.append((position, position + len(sentence)))
            parts += [sentence, np.zeros(GAP, np.float32)]
            position += len(sentence) + GAP
        clips[name], ranges[name] = np.concatenate(parts), spans
    # Fan hum, broadband noise and keyboard clicks at 10 dB SNR (same recipe as the earlier call test).
    male = clips["male"]
    rng = np.random.default_rng(231)
    t = np.arange(len(male)) / RATE
    noise = (.5 * np.sin(2 * np.pi * 120 * t) + .25 * np.sin(2 * np.pi * 240 * t)
             + np.convolve(rng.standard_normal(len(t)), np.ones(7) / 7, mode="same"))
    for start in rng.integers(0, len(male) - 450, len(male) // 4000):
        noise[start:start + 450] += 4 * np.exp(-np.arange(450) / 65) * rng.standard_normal(450)
    speech_rms = np.sqrt(np.mean(male[np.abs(male) > 0.01] ** 2))
    noise *= speech_rms / (np.sqrt(np.mean(noise ** 2)) * 10 ** (10 / 20))
    clips["noisy"], ranges["noisy"] = (male + noise).astype(np.float32), ranges["male"]
    return clips, ranges


_campplus = None


def speaker_embedding(samples, rate):
    global _campplus
    import kaldi_native_fbank as knf
    import onnxruntime as ort
    if _campplus is None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        _campplus = ort.InferenceSession(str(ROOT / ".run/campplus.onnx"), options, providers=["CPUExecutionProvider"])
    g = gcd(rate, 16000)
    audio = resample_poly(samples, 16000 // g, rate // g).astype(np.float32)
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.mel_opts.num_bins = 80
    bank = knf.OnlineFbank(opts)
    bank.accept_waveform(16000, (audio * 32768).tolist())
    bank.input_finished()
    feats = np.stack([bank.get_frame(i) for i in range(bank.num_frames_ready)])
    feats -= feats.mean(axis=0)
    vector = _campplus.run(None, {_campplus.get_inputs()[0].name: feats[None]})[0].flatten()
    return vector / np.linalg.norm(vector)


def render(voice, samples, strength=1.0):
    live = LiveVoice(voice, strength, SPEECH_LOOKAHEAD, mode=os.environ.get("EVAL_MODE", "character"))
    padded = np.pad(samples, (0, (-len(samples)) % RENDER_FRAME))
    parts, times = [], []
    for frame in padded.reshape(-1, RENDER_FRAME):
        started = time.perf_counter()
        parts.append(live.process(frame))
        times.append(time.perf_counter() - started)
    parts.append(live.flush())
    return np.concatenate(parts)[:len(samples)], np.asarray(times) * 1000


def apply_settings():
    for item in filter(None, os.environ.get("EVAL_SET", "").split(";")):
        target, value = item.split("=", 1)
        module, name, *key = target.split(".")
        owner = importlib.import_module("backend." + module)
        if key:  # character.TIMBRE_REFINE.anon=0 sets one entry of a per-voice table
            getattr(owner, name)[key[0]] = ast.literal_eval(value)
        else:
            setattr(owner, name, ast.literal_eval(value))
        print(f"set backend.{target} = {value}", flush=True)


def main():
    tag = sys.argv[1]
    apply_settings()
    voices = sys.argv[2:] or list(engine.VOICES)
    out = ROOT / ".run/eval" / tag
    out.mkdir(parents=True, exist_ok=True)
    refs = {v: speaker_embedding(*sf.read(str(ROOT / f"assets/references/{v}.wav"), dtype="float32")) for v in voices}
    clips, ranges = build_clips()
    render(voices[0], clips["female"][:2 * RATE])  # warm up
    rows = []
    for voice in voices:
        render(voice, clips["female"][:2 * RATE])  # warm up this voice's graphs
        for name, clip in clips.items():
            output, times = render(voice, clip)
            margin = int(0.25 * RATE)
            for index, (start, end) in enumerate(ranges[name], 1):
                sf.write(str(out / f"{name}-{voice}-{index}.wav"), output[max(0, start - margin):end + margin], RATE,
                         subtype="PCM_16")
            spoken = np.concatenate([output[a:b] for a, b in ranges[name]])
            pauses = np.concatenate([output[b + margin:b + GAP - margin] for a, b in ranges[name]])
            row = {
                "voice": voice, "input": name, "mode": os.environ.get("EVAL_MODE", "character"),
                "frame_samples": RENDER_FRAME,
                "ms_mean": round(float(times.mean()), 1), "ms_p95": round(float(np.percentile(times, 95)), 1),
                "ms_max": round(float(times.max()), 1),
                "peak": round(float(np.max(np.abs(output))), 3),
                "clipped_pct": round(float(np.mean(np.abs(output) >= 0.985) * 100), 3),
                "speech_rms_db": round(float(20 * np.log10(np.sqrt(np.mean(spoken ** 2)) + 1e-9)), 1),
                "pause_rms_db": round(float(20 * np.log10(np.sqrt(np.mean(pauses ** 2)) + 1e-9)), 1),
                "similarity": round(float(refs[voice] @ speaker_embedding(spoken, RATE)), 3),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    (out / "metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    by_voice = {v: round(float(np.mean([r["similarity"] for r in rows if r["voice"] == v])), 3) for v in voices}
    print(f"{tag} SUMMARY similarity {by_voice}  mean ms {np.mean([r['ms_mean'] for r in rows]):.1f}"
          f"  worst p95 ms {max(r['ms_p95'] for r in rows):.1f}", flush=True)


if __name__ == "__main__":
    main()
