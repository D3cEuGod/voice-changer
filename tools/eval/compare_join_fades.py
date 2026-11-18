"""Isolate packet crossfade changes by replaying identical neural renderings.

Only bundled synthetic fixtures are used. This opens no audio devices. The
short-join result is also compared to the 0.5.0 join implementation, with the
same +2 pitch and the same cached windows on both sides.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend import character, engine, live

FRAME = 2646


def render(module, model, voice, signal):
    with patch.object(module, 'CharacterModel', return_value=model):
        session = module.LiveVoice(voice, 1, live.SPEECH_LOOKAHEAD, pitch_offset=2)
    parts = [session.process(signal[i:i+FRAME]) for i in range(0, len(signal), FRAME)]
    parts.append(session.flush())
    return np.concatenate(parts)


def replay(module, profile, voice, signal, windows):
    iterator = iter(windows)
    model = SimpleNamespace(context_samples=engine.RATE, strength=1, local=profile,
                            render=lambda window, count: next(iterator))
    result = render(module, model, voice, signal)
    assert next(iterator, None) is None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--voices', nargs='+', default=['anon', 'tomori'])
    parser.add_argument('--samples', nargs='+', default=['m1', 'm2', 'm3', 'm4'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output(['git', 'show', '95e4b1b:backend/live.py'], cwd=ROOT)
    old = ModuleType('backend._legacy_join')
    old.__package__ = 'backend'
    exec(compile(source, '<0.5.0 live join>', 'exec'), old.__dict__)
    truths = (ROOT/'.run/evalset/sentences.txt').read_text(encoding='utf-8').splitlines()
    rows = []
    for voice in args.voices:
        for sample in args.samples:
            speech = engine.decode((ROOT/f'.run/evalset/{sample}.wav').read_bytes()).astype(np.float32)
            present = np.flatnonzero(np.abs(speech) > .0001)
            speech = speech[present[0]:present[-1]+1].copy()
            speech *= .02/np.sqrt(np.mean(speech**2))
            for condition in ('normal', 'weak-edges'):
                x = speech.copy()
                if condition == 'weak-edges':
                    x[:round(.65*engine.RATE)] *= .3
                    x[-round(.65*engine.RATE):] *= np.linspace(1, .15, round(.65*engine.RATE))
                signal = np.r_[np.zeros(engine.RATE), x, np.zeros(engine.RATE)].astype(np.float32)
                if condition == 'weak-edges':
                    signal += np.random.default_rng(53).standard_normal(len(signal)).astype(np.float32)*.0015
                signal = np.pad(signal, (0, -len(signal) % FRAME))
                sf.write(args.output/f'input-{sample}-{condition}.wav', signal, engine.RATE)
                model = character.CharacterModel(voice, 1, pitch_offset=2)
                original_render, windows = model.render, []
                def capture(window, count):
                    result = original_render(window, count)
                    windows.append(result)
                    return result
                model.render = capture
                render(live, model, voice, signal)
                short = replace(model.local, stream_crossfade_ms=0)
                short_result = replay(live, short, voice, signal, windows)
                legacy = replay(old, short, voice, signal, windows)
                error = float(np.max(np.abs(short_result-legacy)))
                np.testing.assert_allclose(short_result, legacy, atol=1e-7, rtol=0)
                for variant, profile in [('fade40', replace(short, stream_crossfade_ms=40)), ('short', short)]:
                    output = short_result if variant == 'short' else replay(live, profile, voice, signal, windows)
                    name = f'{voice}-{sample}-{condition}-{variant}.wav'
                    sf.write(args.output/name, output, engine.RATE)
                    row = dict(voice=voice, sample=sample, condition=condition, variant=variant, file=name,
                               truth=truths[int(sample[1:])-1], pitch_offset=2,
                               legacy_join_max_abs_error=error, seconds=len(output)/engine.RATE,
                               peak=float(np.max(np.abs(output))))
                    rows.append(row)
                print(f'{voice} {sample} {condition}: replay matches old short join (max {error:g})', flush=True)
    (args.output/'manifest.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
