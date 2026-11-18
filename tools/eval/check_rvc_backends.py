"""Compare upstream PyTorch, CPU ONNX and DirectML vocoders with matched inputs/noise.

Uses synthetic speech only; modified diagnostic graphs remain under .run.
"""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from backend import character,engine,local_rvc
from backend.live import LiveVoice,SPEECH_LOOKAHEAD


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--voice',default='tannhauser')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    profile=local_rvc.profile(args.voice)
    captured=[];original=local_rvc.graph(args.voice)
    class Capture:
        def run(self,names,inputs):
            if len(captured)<1 and np.count_nonzero(inputs['f0'])>20:
                captured.append({k:v.copy() for k,v in inputs.items()})
            return original.run(names,inputs)
    old=local_rvc.graph
    local_rvc.graph=lambda _:Capture()
    live=LiveVoice(args.voice,1,SPEECH_LOOKAHEAD,pitch_offset=2)
    signal=engine.decode((ROOT/'.run/evalset/m2.wav').read_bytes())
    try:
        for i in range(0,len(signal)-2646,2646):
            live.process(signal[i:i+2646])
            if captured:break
    finally:local_rvc.graph=old
    inputs=captured[0]
    np.savez(args.output/'inputs.npz',**inputs)
    noise=np.random.default_rng(24).standard_normal((1,profile.sample_rate,1)).astype(np.float32)
    np.save(args.output/'source-noise.npy',noise)
    graph=onnx.load(ROOT/'models'/profile.graph)
    for node in graph.graph.node:
        if node.op_type=='RandomNormalLike':
            tensor=numpy_helper.from_array(noise,name='matched_source_noise')
        elif node.op_type=='RandomUniform':
            tensor=numpy_helper.from_array(np.zeros((1,1,1),np.float32),name='matched_phase')
        else:continue
        del node.input[:];del node.attribute[:]
        node.op_type='Constant';node.attribute.append(onnx.helper.make_attribute('value',tensor))
    path=args.output/'fixed-noise.onnx';onnx.save(graph,path)
    rows=[];outputs={}
    for name,providers in [('cpu',['CPUExecutionProvider']),('dml',['DmlExecutionProvider','CPUExecutionProvider'])]:
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.enable_mem_pattern=False
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
        session=ort.InferenceSession(str(path),options,providers=providers)
        outputs[name]=session.run(None,inputs)[0].ravel()
        np.save(args.output/f'{name}.npy',outputs[name])
    # The upstream generator uses the same feature and pitch tensors; all
    # random excitation is supplied explicitly to make waveform parity useful.
    script='''import sys,json,numpy as np,torch
from pathlib import Path
root=Path(sys.argv[1]);folder=Path(sys.argv[2]);voice=sys.argv[3]
sys.path.insert(0,str(root/'.run/rvc-src'))
from infer.lib.infer_pack.models import SynthesizerTrnMs768NSFsid
data=json.loads((root/'models/local-rvc/profiles.json').read_text(encoding='utf-8'))['voices'][voice]
checkpoint=torch.load(root/'models/local-rvc'/Path(data['graph']).parent/'weights.pth',map_location='cpu',weights_only=True)
config=list(checkpoint['config']);config[-3]=checkpoint['weight']['emb_g.weight'].shape[0]
net=SynthesizerTrnMs768NSFsid(*config,is_half=False);net.load_state_dict(checkpoint['weight'],strict=False);net.eval()
x=np.load(folder/'inputs.npz');noise=torch.from_numpy(np.load(folder/'source-noise.npy'))
original=torch.randn_like
torch.randn_like=lambda t,*a,**k: noise if tuple(t.shape)==tuple(noise.shape) else torch.zeros_like(t)
torch.set_num_threads(3)
with torch.no_grad():
 y=net.infer(torch.from_numpy(x['units']),torch.tensor([100]),torch.from_numpy(x['pitch']),torch.from_numpy(x['f0']),torch.tensor([0]))[0].numpy().ravel()
np.save(folder/'torch.npy',y)
'''
    subprocess.run([str(ROOT/'.run/export-env/Scripts/python.exe'),'-X','utf8','-c',script,str(ROOT),str(args.output.resolve()),args.voice],check=True)
    outputs['torch']=np.load(args.output/'torch.npy')
    for a,b in [('torch','cpu'),('cpu','dml')]:
        error=outputs[a]-outputs[b]
        row={'comparison':a+'/'+b,'max_abs':float(np.max(np.abs(error))),
             'rmse':float(np.sqrt(np.mean(error**2))),
             'snr_db':float(20*np.log10(np.linalg.norm(outputs[a])/max(np.linalg.norm(error),1e-12)))}
        rows.append(row);print(json.dumps(row),flush=True)
    (args.output/'metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__':main()
