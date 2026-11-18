"""Compare an earlier gate against this checkout with synthetic soft speech.

No devices opened. Requires .run/evalset from make_evalset.ps1. Optional baseline
is read with git show; it runs locally with the same shared model sessions.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import types
from unittest.mock import patch

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend import character, live
from backend.engine import RATE


def render(model, voice, signal):
    frame = 2646
    with patch.object(live, 'CharacterModel', model):
        session = live.LiveVoice(voice, 1, live.SPEECH_LOOKAHEAD, pitch_offset=4)
    signal = np.pad(signal, (0, -len(signal) % frame))
    parts = [session.process(signal[i:i+frame]) for i in range(0, len(signal), frame)]
    parts.append(session.flush())
    return np.concatenate(parts)[:len(signal)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--voices', nargs='+', default=['tomori','taffy','anon'])
    parser.add_argument('--soft-factor', type=float, default=.3)
    parser.add_argument('--noise-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models = {'after': character.CharacterModel}
    if args.baseline_ref:
        source = subprocess.check_output(['git', 'show', f'{args.baseline_ref}:backend/character.py'], cwd=ROOT)
        old = types.ModuleType('backend._baseline_character')
        old.__package__ = 'backend'
        exec(compile(source, '<baseline_character>', 'exec'), old.__dict__)
        old.graph, old.voice_index = character.graph, character.voice_index
        models = {'before': old.CharacterModel, **models}

    x, rate = sf.read(ROOT/'.run/evalset/m2.wav', dtype='float32')
    x = resample_poly(x, RATE, rate).astype(np.float32)
    x *= .02 / np.sqrt(np.mean(x*x))
    onset = x.copy()
    onset[:round(.9*RATE)] *= args.soft_factor
    conditions = {'normal': x, 'soft-onset': onset, 'quiet': x*args.soft_factor}
    if args.noise_only:
        t = np.arange(5*RATE)/RATE
        fan = .006*(np.sin(2*np.pi*120*t)+.5*np.sin(2*np.pi*240*t))
        fan[:RATE//2] = 0
        keyboard = np.zeros(5*RATE)
        rng = np.random.default_rng(112)
        for start in range(RATE//2, len(keyboard)-800, RATE//3):
            keyboard[start:start+800] = .12*np.exp(-np.arange(800)/100)*rng.standard_normal(800)
        conditions = {'fan':fan.astype(np.float32), 'keyboard':keyboard.astype(np.float32)}
    rows = []
    for name, speech in conditions.items():
        signal = np.r_[np.zeros(RATE), speech, np.zeros(RATE)].astype(np.float32)
        rng = np.random.default_rng(42)
        signal += .0015*rng.standard_normal(len(signal)).astype(np.float32)
        sf.write(args.output/f'{name}-input.wav', signal, RATE)
        for version, model in models.items():
            for voice in args.voices:
                start = time.perf_counter()
                output = render(model, voice, signal)
                sf.write(args.output/f'{name}-{version}-{voice}.wav', output, RATE)
                head = output[RATE:round(1.9*RATE)]
                pauses = np.r_[output[:round(.7*RATE)], output[-round(.5*RATE):]]
                row = {'input':name, 'version':version, 'voice':voice,
                       'onset_rms':float(np.sqrt(np.mean(head*head))),
                       'pause_rms':float(np.sqrt(np.mean(pauses*pauses))),
                       'output_rms':float(np.sqrt(np.mean(output*output))),
                       'peak':float(np.max(np.abs(output))),
                       'clipped_pct':float(np.mean(np.abs(output)>.985)*100),
                       'elapsed':round(time.perf_counter()-start,2)}
                rows.append(row)
                print(json.dumps(row), flush=True)
    (args.output/'metrics.json').write_text(json.dumps(rows,indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
