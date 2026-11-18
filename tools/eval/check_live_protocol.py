"""Check real WebSocket packet sizes and tail preservation against an idle backend."""
import argparse
import json
from pathlib import Path
import sys

from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gui.backend import BackendClient


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--backend-url', default='http://127.0.0.1:8765')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    client = BackendClient(args.backend_url)
    rows = []
    for voice in ('tomori', 'taffy', 'anon'):
        for packet in (80, 100, 120, 200):
            with connect(client.live_url(voice, 1, packet_ms=packet), proxy=None, compression=None) as ws:
                hello = json.loads(ws.recv())
                frame, tail = hello['frame_samples'], hello['lookahead_samples']
                assert frame == round(22050*packet/1000), hello
                sizes = []
                for _ in range(3):
                    ws.send(bytes(frame*2))
                    reply = ws.recv()
                    assert isinstance(reply, bytes) and not any(reply), 'Nonzero output for silence'
                    sizes.append(len(reply)//2)
                ws.send('finish')
                sizes.append(len(ws.recv())//2)
                produced = [max(0,i*frame-tail) for i in range(4)]
                expected = [produced[i+1]-produced[i] for i in range(3)] + [min(3*frame,tail)]
                assert sizes == expected, (hello, sizes)
                assert sum(sizes) == 3*frame, 'Tail truncated or duplicated'
                rows.append({'voice': voice, 'packet_ms': packet, 'reply_samples': sizes})
    url = client.live_url('tomori', 1).replace('&packet_ms=200', '')
    with connect(url, proxy=None) as ws:
        assert json.loads(ws.recv())['frame_samples'] == 4410, 'Old clients changed packet size'
    try:
        with connect(url+'&packet_ms=160', proxy=None):
            raise AssertionError('Invalid packet duration accepted')
    except InvalidStatus as exc:
        assert exc.response.status_code == 403, exc
    result = {'health': client.health(), 'tests': rows,
              'legacy_default_200_ms': True, 'invalid_160_ms_rejected': True}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
