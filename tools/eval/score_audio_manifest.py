"""Score local synthetic fixtures listed as {file, truth, ...} with SenseVoice.

Run using .run/export-env/Scripts/python.exe. No downloads or audio devices.
"""
import argparse
from collections import defaultdict
import json
from math import gcd
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
DIGITS = str.maketrans('0123456789', '零一二三四五六七八九')


def normalise(text):
    return ''.join(c for c in text if c.isalnum()).translate(DIGITS)


def distance(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j]+1, current[-1]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    folder = next((ROOT/'.run').glob('sherpa-onnx-sense-voice-*'))
    model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        str(folder/'model.int8.onnx'), str(folder/'tokens.txt'),
        num_threads=3, language='zh', use_itn=False)
    rows = json.loads(args.manifest.read_text(encoding='utf-8'))
    groups = defaultdict(lambda: {'errors': 0, 'chars': 0})
    for row in rows:
        x, rate = sf.read(args.manifest.parent/row['file'], dtype='float32')
        g = gcd(rate, 16000)
        stream = model.create_stream()
        stream.accept_waveform(16000, resample_poly(x, 16000//g, rate//g).astype(np.float32))
        model.decode_stream(stream)
        text, truth = normalise(stream.result.text), normalise(row['truth'])
        row.update(text=text, errors=distance(truth, text), chars=len(truth))
        key = '/'.join(row.get(part, 'all') for part in ('voice', 'condition', 'variant'))
        groups[key]['errors'] += row['errors']
        groups[key]['chars'] += row['chars']
    output = args.manifest.parent/'asr.json'
    output.write_text(json.dumps({'rows': rows, 'groups': groups}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(groups, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
