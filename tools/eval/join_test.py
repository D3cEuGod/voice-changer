"""Packet-periodic artifact test on real speech.

The same clip is rendered 21 times, each time delayed by another 1/21 of a packet. Every moment of the
speech therefore lands once in every position inside a packet, so after folding the outputs modulo the
packet length the speech content averages out evenly and whatever remains is caused by the packet joins.

Run from the project root with the backend venv. argv: tag [voice ...]; EVAL_SET as in eval_render.py.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from backend import engine  # noqa: E402
from backend.live import SHORT_FRAME, SPEECH_LOOKAHEAD, LiveVoice  # noqa: E402

RATE = engine.RATE
STEPS = 21
WIDTH = SHORT_FRAME // STEPS  # 210 samples = 9.5 ms


def render(voice, samples):
    live = LiveVoice(voice, 1.0, SPEECH_LOOKAHEAD)
    padded = np.pad(samples, (0, (-len(samples)) % SHORT_FRAME))
    parts = [live.process(frame) for frame in padded.reshape(-1, SHORT_FRAME)]
    parts.append(live.flush())
    return np.concatenate(parts)[:len(padded)]


def folded(signal_sets):
    """Mean energy per packet position over all renders."""
    total = np.zeros(STEPS)
    for y in signal_sets:
        y = y[:len(y) // SHORT_FRAME * SHORT_FRAME]
        total += (y ** 2).reshape(-1, STEPS, WIDTH).mean(axis=2).sum(axis=0)
    return total / total.mean()


def main():
    tag = sys.argv[1]
    voices = sys.argv[2:] or list(engine.VOICES)
    sys.path.insert(0, str(Path(__file__).parent))
    from eval_render import apply_settings
    apply_settings()  # same EVAL_SET switches as eval_render.py
    clip = engine.decode((ROOT / ".run/evalset/m2.wav").read_bytes()).astype(np.float32)
    high_pass = butter(4, 5000, btype="highpass", fs=RATE, output="sos")
    inputs = [np.pad(clip, (SHORT_FRAME + k * WIDTH, SHORT_FRAME)) for k in range(STEPS)]
    control = folded(inputs)
    print(f"{tag} input control: level min/max {control.min():.2f}/{control.max():.2f}", flush=True)
    for voice in voices:
        outputs = [render(voice, x) for x in inputs]
        level = folded(outputs)
        high = folded([sosfilt(high_pass, y) for y in outputs])
        print(f"{tag} {voice}: level min/max {level.min():.2f}/{level.max():.2f}   high-band min/max {high.min():.2f}/{high.max():.2f}",
              flush=True)
        print("   level:", " ".join(f"{v:.2f}" for v in level), flush=True)
        print("   high :", " ".join(f"{v:.2f}" for v in high), flush=True)


if __name__ == "__main__":
    main()
