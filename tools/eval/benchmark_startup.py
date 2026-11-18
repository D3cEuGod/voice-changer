"""Time the real startup script on an unused local port, then stop only our backend."""
import argparse
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gui.backend import BackendClient, BackendError, BackendLauncher, stop_process_tree


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--label', default='startup')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.label) or not 1024 <= args.port <= 65535:
        parser.error('Use a simple label and a port between 1024 and 65535')
    client = BackendClient(f'http://127.0.0.1:{args.port}')
    try:
        client.health(connect_timeout=.1)
    except BackendError:
        pass
    else:
        parser.error('A healthy backend already uses this port; choose an unused port')
    source = (ROOT/'start-backend.cmd').read_text()
    if source.count('--port 8765') != 1 or source.count('cd /d "%~dp0"') != 1:
        parser.error('The startup script layout changed; review the benchmark first')
    script = ROOT/'.run'/f'start-benchmark-{args.label}.cmd'
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(source.replace('cd /d "%~dp0"', 'cd /d "%~dp0.."')
                      .replace('--port 8765', f'--port {args.port}'), encoding='utf-8')
    rows, outcome = [], {}
    start = time.perf_counter()
    launcher = BackendLauncher(client,
        lambda message: rows.append({'s':round(time.perf_counter()-start,3), 'message':message}),
        lambda health, owned: outcome.update(health=health, owned=owned,
                                             seconds=round(time.perf_counter()-start,3)),
        lambda error: outcome.update(error=error), script=script, wait_seconds=120)
    try:
        launcher.start()
        launcher.join(130)
        if launcher.is_alive() or 'error' in outcome or not outcome.get('owned'):
            raise RuntimeError(f'Startup did not complete in our process: {outcome}')
        outcome['events'] = rows
        output = args.output or ROOT/'.run'/f'startup-{args.label}.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(outcome,indent=2), encoding='utf-8')
        print(json.dumps(outcome), flush=True)
    finally:
        launcher.cancel()
        stop_process_tree(launcher.process)


if __name__ == '__main__':
    main()
