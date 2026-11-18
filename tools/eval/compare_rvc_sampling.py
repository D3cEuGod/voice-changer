"""Compare zero and native-scale latent sampling with otherwise identical settings.

Uses existing synthetic speech and public example audio, never the user's mic.
The vocoder's own NSF excitation remains stochastic in both conditions.
"""
import argparse
from pathlib import Path
import json
import sys
from unittest.mock import patch
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend import character, engine, live
from backend.rvc_noise import RollingLatent
from tools.eval.compare_join_fades import render


class ZeroLatent:
    def advance(self, *args):
        return np.zeros((1, 192, 100), np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--voices', nargs='+', default=['anon', 'tomori', 'taffy', 'tannhauser'])
    parser.add_argument('--samples', nargs='+', default=['m2', 'm3', 'f2'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    truths = (ROOT/'.run/evalset/sentences.txt').read_text(encoding='utf-8').splitlines()
    rows = []
    for sample in args.samples:
        x = engine.decode((ROOT/f'.run/evalset/{sample}.wav').read_bytes()).astype(np.float32)
        active = np.flatnonzero(np.abs(x) > .0001)
        x = x[active[0]:active[-1]+1].copy()
        x *= .015/np.sqrt(np.mean(x*x))
        x[:round(.4*engine.RATE)] *= .3
        x[-round(.5*engine.RATE):] *= np.linspace(1, .15, round(.5*engine.RATE))
        signal = np.r_[np.zeros(engine.RATE), x, np.zeros(engine.RATE)].astype(np.float32)
        signal += np.random.default_rng(53).standard_normal(len(signal)).astype(np.float32)*.0015
        signal = np.pad(signal, (0, -len(signal) % 2646))
        for voice in args.voices:
            for variant in ('zero', 'native'):
                factory = ZeroLatent if variant == 'zero' else lambda: RollingLatent(seed=53)
                with patch.object(character, 'RollingLatent', side_effect=factory):
                    model = character.CharacterModel(voice, 1, pitch_offset=2)
                y = render(live, model, voice, signal)
                name = f'{voice}-{sample}-{variant}.wav'
                sf.write(args.output/name, y, engine.RATE)
                rows.append(dict(file=name, voice=voice, sample=sample, variant=variant,
                                 condition='weak-edges', truth=truths[int(sample[1:])-1],
                                 pitch_offset=2, index_rate=model.local.index_rate,
                                 seconds=len(y)/engine.RATE, peak=float(np.max(np.abs(y))),
                                 pause_rms=float(np.sqrt(np.mean(y[-engine.RATE//2:]**2)))))
            print(f'{voice} {sample}', flush=True)
    (args.output/'manifest.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
