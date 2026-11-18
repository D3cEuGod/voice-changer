"""Refresh the current installed voices' file examples without historical comparisons."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from gui.backend import BackendClient,DEFAULT_PITCH_PRESET


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend-url',default='http://127.0.0.1:8766')
    args=parser.parse_args()
    client=BackendClient(args.backend_url)
    health=client.health()
    folder=ROOT/'assets/previews'
    original=(folder/'original.wav').read_bytes()
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'backend':health,
              'input_sha256':hashlib.sha256(original).hexdigest(),'pitch_preset':DEFAULT_PITCH_PRESET,
              'note':'Existing synthetic Chinese input; file conversion, not real-time delay or the user microphone',
              'files':[]}
    for voice in client.voices():
        for mode in ('character','call'):
            data=client.convert(original,'audio/wav',voice['id'],1,mode=mode,pitch_preset=DEFAULT_PITCH_PRESET)
            filename=voice['id']+('-call' if mode=='call' else '')+'.wav'
            (folder/filename).write_bytes(data)
            manifest['files'].append({'file':filename,'voice':voice['id'],'mode':mode,
                                      'model':health['voice_models'][voice['id']],
                                      'sha256':hashlib.sha256(data).hexdigest()})
            print(filename,flush=True)
    (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
