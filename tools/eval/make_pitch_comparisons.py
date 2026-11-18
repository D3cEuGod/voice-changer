"""Compare the available pitch presets for every installed voice, without playback."""
import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gui.backend import BackendClient, PITCH_PRESETS,DEFAULT_PITCH_PRESET


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--backend-url', default='http://127.0.0.1:8766')
    parser.add_argument('--input', type=Path, default=ROOT/'assets/pitch-comparisons/input-male.wav')
    parser.add_argument('--output', type=Path, default=ROOT/'assets/pitch-comparisons')
    args = parser.parse_args()
    client = BackendClient(args.backend_url)
    health = client.health()
    if not health.get('pitch_control'):
        raise RuntimeError('This backend does not support pitch control')
    original = args.input.read_bytes()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'input-male.wav').write_bytes(original)
    manifest = {'created_utc':datetime.now(timezone.utc).isoformat(), 'backend':health,
                'input':{'file':'input-male.wav','sha256':hashlib.sha256(original).hexdigest(),
                         'kind':'Synthetic male speech from the existing regression set; not the user microphone'},
                'strength':1, 'mode':'character', 'packet_ms':200,
                'default_pitch_preset':DEFAULT_PITCH_PRESET,'comparisons':[]}
    for voice in (v['id'] for v in client.voices()):
        pieces, segments, position = [], [], 0
        for preset, (name, shift, offset) in PITCH_PRESETS.items():
            data = client.convert(original,'audio/wav',voice,1,pitch_preset=preset)
            samples, rate = sf.read(BytesIO(data), dtype='float32')
            assert rate == 22050 and np.isfinite(samples).all() and np.max(np.abs(samples)) > .01
            segments.append({'preset':preset,'name':name,'pitch_shift':shift,'pitch_offset':offset,
                             'start_s':round(position/rate,3),'end_s':round((position+len(samples))/rate,3),
                             'result_sha256':hashlib.sha256(data).hexdigest()})
            pieces.extend((samples,np.zeros(round(.6*rate),np.float32)))
            position += len(samples)+round(.6*rate)
        path = args.output/f'{voice}.wav'
        sf.write(path,np.concatenate(pieces),rate,subtype='PCM_16')
        manifest['comparisons'].append({'voice':voice,'file':path.name,
                                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                        'segments':segments})
        print(f'{voice}: '+ ' / '.join(PITCH_PRESETS),flush=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
