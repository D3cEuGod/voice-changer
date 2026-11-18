"""Synthetic repeated-phrase stress test of recurrent state; no audio devices."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from backend import character,engine
from backend.live import LiveVoice,SPEECH_LOOKAHEAD


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--voices',nargs='+',default=['taffy','tannhauser'])
    p.add_argument('--seconds',type=int,default=90)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    phrase=engine.decode((ROOT/'.run/evalset/m2.wav').read_bytes())
    phrase=np.r_[phrase,np.zeros(round(.8*engine.RATE),np.float32)]
    x=np.tile(phrase,int(np.ceil(args.seconds*engine.RATE/len(phrase))))[:args.seconds*engine.RATE]
    rng=np.random.default_rng(76)
    x=(x+.0015*rng.standard_normal(len(x))).astype(np.float32)
    x=np.pad(x,(0,-len(x)%2646))
    reports=[]
    original_pitch=character.rmvpe_pitch
    for voice in args.voices:
        live=LiveVoice(voice,1,SPEECH_LOOKAHEAD,pitch_offset=2)
        denoise=live.model.denoiser.process
        activity=None;phantom=total=0
        def track(window,count):
            nonlocal activity
            clean,activity=denoise(window,count)
            return clean,activity
        def pitch(clean):
            nonlocal phantom,total
            f0=original_pitch(clean)
            accepted=np.interp(np.arange(len(f0))*160,np.arange(len(activity)),activity)>.005
            phantom+=int(np.count_nonzero((f0>0)&~accepted));total+=int(np.count_nonzero(f0>0))
            return f0
        live.model.denoiser.process=track
        character.rmvpe_pitch=pitch
        parts=[];rows=[]
        for i in range(0,len(x),2646):
            parts.append(live.process(x[i:i+2646]))
            if i% (2646*125)==0:
                rows.append({'seconds':round(i/engine.RATE,2),'noise_rms':float(live.model.denoiser.noise_rms),
                             'source_register_hz':float(np.exp(live.model.register)) if live.model.register else None})
        parts.append(live.flush());character.rmvpe_pitch=original_pitch
        y=np.concatenate(parts)
        sf.write(args.output/f'{voice}.wav',y,engine.RATE)
        report={'voice':voice,'seconds':args.seconds,'voiced_frames':total,'voiced_frames_outside_activity':phantom,'state':rows}
        reports.append(report);print(json.dumps(report),flush=True)
    (args.output/'metrics.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')


if __name__=='__main__':main()
