"""Compare FAISS search latency and neighbors without opening audio devices."""
import argparse
import json
from pathlib import Path
import sys
import time

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.character import voice_index
from backend.engine import VOICES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=40)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    rows = []
    for voice in VOICES:
        index = voice_index(voice)
        queries = index.reconstruct_n(0, min(50, index.ntotal)).copy()
        baseline = None
        for threads in (1, 2, 4):
            faiss.omp_set_num_threads(threads)
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                distances, ids, _ = index.search_and_reconstruct(queries, 8)
                times.append((time.perf_counter()-start)*1000)
            if baseline is None:
                baseline = distances.copy(), ids.copy()
            np.testing.assert_array_equal(ids, baseline[1])
            np.testing.assert_allclose(distances, baseline[0], rtol=1e-5, atol=1e-5)
            rows.append(dict(voice=voice, threads=threads, neighbors_match=True,
                             p50_ms=float(np.percentile(times, 50)),
                             p95_ms=float(np.percentile(times, 95)), max_ms=max(times)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    main()
