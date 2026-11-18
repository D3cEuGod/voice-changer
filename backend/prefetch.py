"""Read the model files the backend needs first into the Windows file cache, while Python is still importing.

On the D: hard disk (about 155 MB/s) the first character alone needs about 1 GB: ContentVec, RMVPE, the
voice graph and its index. ONNX Runtime and FAISS read those files while holding Python's GIL, which also
stalls the web server. Reading them ahead in a plain thread (file reads release the GIL) overlaps the disk
time with the library imports and turns the later loads into memory reads.

Only the standard library is used here, so the thread starts before numpy, ONNX Runtime and FAISS load.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
LOCAL = MODELS / "local-rvc"
CHUNK = 8 << 20
LOG = logging.getLogger("uvicorn.error")
_started = False


def _signature(path: Path) -> list:
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino]


def _graph_file(name: str) -> Path | None:
    """The file backend/graph_cache.py will load for models/<name>: its optimized copy when one matches."""
    source = (MODELS / name).resolve()
    if not source.is_file():
        return None
    if os.environ.get("VOICE_ORT_CACHE") == "0":
        return source
    signature, best = _signature(source), None
    for manifest in (ROOT / ".run" / "ort-cache").glob(f"{source.stem}-*.json"):
        try:
            identity = json.loads(manifest.read_text(encoding="utf-8"))["identity"]
            candidate = manifest.with_suffix(".onnx")
            if identity[1] == str(source) and identity[2] == signature and candidate.is_file():
                if best is None or candidate.stat().st_mtime > best.stat().st_mtime:
                    best = candidate
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            continue
    return best or source


def _voice_files(voice: str) -> list[Path]:
    try:
        entry = json.loads((LOCAL / "profiles.json").read_text(encoding="utf-8"))["voices"][voice]
        files = [LOCAL / entry["graph"]]
        if float(entry.get('index_rate', .75)) > 0:
            files.append(LOCAL / entry["index"])
        return files
    except (OSError, ValueError, KeyError, TypeError):
        return []


def plan(first: str) -> list[Path]:
    """Files the backend needs before it reports ready: the shared graphs and the first voice (about 1 GB).

    The other voices are left out on purpose: on a hard disk, reading them too would compete with these
    for the same disk head. Their own background loads read them later.
    """
    files = [_graph_file(name) for name in ("contentvec_768l12.onnx", "rmvpe.onnx",
                                             "gtcrn_simple.onnx", "silero_vad_16k_op15.onnx")]
    files += _voice_files(first)
    return [f for f in dict.fromkeys(files) if f is not None and f.is_file()]


def _read(files: list[Path]) -> None:
    started, total = time.perf_counter(), 0
    for path in files:
        try:
            with path.open("rb", buffering=0) as stream:
                while chunk := stream.read(CHUNK):
                    total += len(chunk)
        except OSError:
            continue
    LOG.info("Prefetched %d model files (%.0f MB) in %.2f s", len(files), total / 1e6, time.perf_counter() - started)


def start_prefetch() -> None:
    """Start once per process, only when VOICE_PREFETCH is not 0."""
    global _started
    if _started or os.environ.get("VOICE_PREFETCH") == "0":
        return
    _started = True
    first = os.environ.get("VOICE_WARM_FIRST", "taffy")
    threading.Thread(target=_read, args=(plan(first),), daemon=True, name="model-prefetch").start()
