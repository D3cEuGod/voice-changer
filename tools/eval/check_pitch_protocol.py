"""Check pitch presets, legacy defaults and invalid values against an idle backend."""
import argparse
import json
from pathlib import Path
import sys
from websockets.sync.client import connect
from websockets.exceptions import InvalidStatus

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gui.backend import BackendClient, BackendError, PITCH_PRESETS


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--backend-url', default='http://127.0.0.1:8766')
    parser.add_argument('--output', type=Path, default=ROOT/'.run/pitch-protocol.json')
    args = parser.parse_args()
    client = BackendClient(args.backend_url)
    assert client.health().get('pitch_control')
    rows = []
    for voice in (v['id'] for v in client.voices()):
        for preset, (_, shift, offset) in PITCH_PRESETS.items():
            with connect(client.live_url(voice,1,packet_ms=120,pitch_preset=preset), proxy=None) as ws:
                hello = json.loads(ws.recv())
                assert hello['pitch_shift'] == shift and hello['pitch_offset'] == offset, hello
                frame, total = hello['frame_samples'], 0
                for _ in range(3):
                    ws.send(bytes(frame*2))
                    reply = ws.recv()
                    assert isinstance(reply,bytes) and not any(reply)
                    total += len(reply)//2
                ws.send('finish')
                total += len(ws.recv())//2
                assert total == frame*3
                rows.append({'voice':voice,'preset':preset,'hello':hello,'samples_preserved':True})
    invalid = []
    original = (ROOT/'assets/previews/original.wav').read_bytes()
    for key in ('pitch_shift','pitch_offset'):
        for value in ('25','-25','nan','inf'):
            try:
                client._request('POST',f'/api/convert?voice_id=taffy&{key}={value}',original,
                                {'Content-Type':'audio/wav'})
                raise AssertionError(f'Invalid HTTP {key}={value} accepted')
            except BackendError as exc:
                assert exc.status == 422, exc
            try:
                with connect(client.live_url('taffy',1)+f'&{key}={value}',proxy=None):
                    raise AssertionError(f'Invalid WS {key}={value} accepted')
            except InvalidStatus as exc:
                assert exc.response.status_code == 403, exc
            invalid.append({'parameter':key,'value':value,'http':422,'websocket':403})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'health':client.health(),'presets':rows,'invalid':invalid},indent=2),encoding='utf-8')
    print(f'{len(rows)} voice/preset combinations and {len(invalid)*2} invalid requests passed',flush=True)


if __name__ == '__main__':
    main()
