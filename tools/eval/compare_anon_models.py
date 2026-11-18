"""Compare the installed Anon, extra lookahead, and the existing fallback RVC graph."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from backend import character,engine,live,local_rvc
from tools.eval.compare_join_fades import render,replay


def main():
    out=ROOT/'.run/anon-models-053';out.mkdir(parents=True,exist_ok=True)
    alternative=engine.load_graph('anon_rvc.onnx')
    truths=(ROOT/'.run/evalset/sentences.txt').read_text(encoding='utf-8').splitlines()
    rows=[]
    for sample in ('m1','m2','m3','m4','f2','f3'):
        x=engine.decode((ROOT/f'.run/evalset/{sample}.wav').read_bytes()).astype(np.float32)
        active=np.flatnonzero(np.abs(x)>.0001);x=x[active[0]:active[-1]+1].copy()
        x*=.015/np.sqrt(np.mean(x*x));x[:round(.4*engine.RATE)]*=.3
        x[-round(.5*engine.RATE):]*=np.linspace(1,.15,round(.5*engine.RATE))
        signal=np.r_[np.zeros(engine.RATE),x,np.zeros(engine.RATE)].astype(np.float32)
        signal+=np.random.default_rng(53).standard_normal(len(signal)).astype(np.float32)*.0015
        signal=np.pad(signal,(0,-len(signal)%2646))
        model=character.CharacterModel('anon',1,pitch_offset=2)
        model.local=replace(model.local,index_rate=0,stream_lookahead_ms=0,stream_crossfade_ms=0)
        original,windows=model.render,[]
        def capture(window,count):
            y=original(window,count);windows.append(y);return y
        model.render=capture
        current=render(live,model,'anon',signal)
        future=replay(live,replace(model.local,stream_lookahead_ms=180),'anon',signal,windows)
        with patch.object(local_rvc,'graph',return_value=alternative):
            alternative_model=character.CharacterModel('anon',1,pitch_offset=2)
            alternative_model.local=replace(model.local,stream_lookahead_ms=0)
            alternate=render(live,alternative_model,'anon',signal)
        for variant,y in [('current',current),('future180',future),('alternative',alternate)]:
            name=f'{variant}-{sample}.wav';sf.write(out/name,y,engine.RATE)
            rows.append(dict(voice='anon',condition='weak-edges',variant=variant,sample=sample,file=name,
                             truth=truths[int(sample[1:])-1],pitch_offset=2,index_rate=0,
                             lookahead_ms=180 if variant=='future180' else 104.49,
                             model='TogetsuDo fallback' if variant=='alternative' else 'Wanlau Anon_v2'))
        print(sample+' complete',flush=True)
    (out/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
