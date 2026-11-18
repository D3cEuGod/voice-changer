"""Offline long-phrase / pause / soft-onset regression using synthetic speech.

Writes only test audio. No microphone, virtual cable or Discord devices opened.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, butter, sosfilt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.engine import RATE
from backend.live import LiveVoice, SPEECH_LOOKAHEAD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--voices', nargs='+', default=['tomori','taffy','anon','tannhauser'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    x, sr = sf.read(ROOT/'.run/evalset/m2.wav', dtype='float32')
    x = resample_poly(x, RATE, sr).astype(np.float32)
    x *= .02/np.sqrt(np.mean(x*x))
    soft = x.copy()
    soft[:round(.6*RATE)] *= .3
    # A quiet final syllable and a sustained hum continue into the pause.
    long = np.tile(x, 3)
    long[-round(.4*RATE):] *= np.linspace(1,.15,round(.4*RATE))
    rows = []
    for label, speech in [('long',long),('soft',soft)]:
        signal = np.r_[np.zeros(RATE),speech,np.zeros(3*RATE)].astype(np.float32)
        t = np.arange(len(signal))/RATE
        rng = np.random.default_rng(831)
        noise = .0012*np.sin(2*np.pi*120*t) + .0005*np.sin(2*np.pi*240*t)
        signal += (noise + .0005*rng.standard_normal(len(t))).astype(np.float32)
        signal = np.pad(signal,(0,-len(signal)%2646))
        end = RATE+len(speech)
        sf.write(args.output/f'{label}-input.wav',signal,RATE)
        for voice in args.voices:
            session = LiveVoice(voice,1,SPEECH_LOOKAHEAD,pitch_offset=4)
            parts,times=[],[]
            for pos in range(0,len(signal),2646):
                start=time.perf_counter()
                parts.append(session.process(signal[pos:pos+2646]))
                times.append(time.perf_counter()-start)
            parts.append(session.flush())
            output=np.concatenate(parts)
            sf.write(args.output/f'{label}-{voice}.wav',output,RATE)
            tail=output[end+round(.12*RATE):end+round(.8*RATE)]
            pause=output[end+RATE:end+3*RATE]
            high=sosfilt(butter(3,4500,btype='highpass',fs=RATE,output='sos'),tail)
            row={'input':label,'voice':voice,'tail_rms':float(np.sqrt(np.mean(tail**2))),
                 'pause_rms':float(np.sqrt(np.mean(pause**2))),
                 'tail_high_rms':float(np.sqrt(np.mean(high**2))),
                 'peak':float(np.max(np.abs(output))),
                 'p95_ms':float(np.percentile(times,95)*1000)}
            rows.append(row)
            print(json.dumps(row),flush=True)
    (args.output/'metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
