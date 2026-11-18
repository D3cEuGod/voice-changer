"""Offline role-specific RVC parameter trial. Uses only existing synthetic files."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly,butter,sosfilt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from backend import local_rvc,engine
from backend.live import LiveVoice,SPEECH_LOOKAHEAD


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--extended',action='store_true')
    parser.add_argument('--trim',action='store_true')
    parser.add_argument('--variants',nargs='+',default=['old-plus4','plus2','gentler','lower'])
    parser.add_argument('--voice',default='tannhauser')
    parser.add_argument('--samples',nargs='+')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    original=replace(local_rvc.profile(args.voice),index_rate=.75,protect=.33,
                     stream_lookahead_ms=0,stream_crossfade_ms=0,output_gate_full_rms=0)
    variants={
        'old-plus4':(original,4),
        'plus2':(original,2),
        'gentler':(replace(original,index_rate=.4,protect=.1),2),
        'balanced':(replace(original,index_rate=.5,protect=.15),2),
        'no-retrieval':(replace(original,index_rate=0,protect=.1),2),
        'no-retrieval-low':(replace(original,index_rate=0,protect=.1,target_hz=260),2),
        'context':(replace(original,stream_lookahead_ms=180),2),
        'context-balanced':(replace(original,index_rate=.5,protect=.15,stream_lookahead_ms=180),2),
        'lower':(replace(original,target_hz=330,index_rate=.4,protect=.1),2),
        'protected':(replace(original,index_rate=.4,protect=.1,
                             stream_lookahead_ms=160,stream_crossfade_ms=40),2),
        'tail':(replace(original,protect=.15,stream_crossfade_ms=40,output_gate_full_rms=.00045),2),
    }
    rows=[]
    samples=args.samples or (['m1','m2','m3','m4','f2','f3'] if args.extended else ['m2','m3'])
    for sample in samples:
        index=int(sample[1:])
        x,rate=sf.read(ROOT/f'.run/evalset/{sample}.wav',dtype='float32')
        x=resample_poly(x,engine.RATE,rate).astype(np.float32)
        x *= .015/np.sqrt(np.mean(x*x))
        if args.trim:
            present=np.flatnonzero(np.abs(x)>.0001)
            x=x[present[0]:present[-1]+1]
        # Low first syllable and fading final syllable above a stationary room bed.
        x[:round(.4*engine.RATE)] *= .3
        x[-round(.5*engine.RATE):] *= np.linspace(1,.15,round(.5*engine.RATE))
        signal=np.r_[np.zeros(engine.RATE),x,np.zeros(engine.RATE)].astype(np.float32)
        signal += np.random.default_rng(53).standard_normal(len(signal)).astype(np.float32)*.0015
        signal=np.pad(signal,(0,-len(signal)%2646))
        sf.write(args.output/f'input-{sample}.wav',signal,engine.RATE)
        for name,(profile,offset) in variants.items():
            if name not in args.variants:
                continue
            with patch.object(local_rvc,'profile',return_value=profile):
                live=LiveVoice(args.voice,1,SPEECH_LOOKAHEAD,pitch_offset=offset)
            parts=[live.process(signal[i:i+2646]) for i in range(0,len(signal),2646)]
            parts.append(live.flush())
            output=np.concatenate(parts)
            path=args.output/f'{name}-{sample}.wav'
            sf.write(path,output,engine.RATE)
            tail=output[engine.RATE+len(x)-round(.5*engine.RATE):engine.RATE+len(x)+round(.2*engine.RATE)]
            high=sosfilt(butter(3,4500,btype='highpass',fs=engine.RATE,output='sos'),tail)
            row={'variant':name,'sentence':index,'sample':sample,'pitch_offset':offset,'target_hz':profile.target_hz,
                 'index_rate':profile.index_rate,'protect':profile.protect,
                 'tail_rms':float(np.sqrt(np.mean(tail**2))),
                 'tail_high_rms':float(np.sqrt(np.mean(high**2))),
                 'pause_rms':float(np.sqrt(np.mean(output[-round(.5*engine.RATE):]**2)))}
            rows.append(row)
            print(json.dumps(row),flush=True)
    (args.output/'metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
