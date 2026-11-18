"""Install a supplied RVC v2 .pth + .index as one of the app's character voices.

Run with .venv/Scripts/python.exe. Prepares the separate export environment if needed.
Only tensor checkpoints accepted by torch.load(weights_only=True) are loaded.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import faiss
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.local_rvc import DIRECTORY, REGISTRY
from setup_models import CACHE, RVC, RVC_REV, checkout, prepare_export_environment


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--voice", required=True, help="Stable lowercase voice ID, e.g. tannhauser")
    parser.add_argument("--name", help="Display name for a newly installed voice")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--source", default="User-provided RVC model")
    parser.add_argument("--target-hz", type=float)
    parser.add_argument("--pitch", default="auto", help="auto or a fixed shift in semitones")
    parser.add_argument("--index-rate", type=float, default=.75)
    parser.add_argument("--protect", type=float, default=.33)
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument('--stream-lookahead-ms',type=int,default=0,help='Optional minimum streaming lookahead')
    parser.add_argument('--stream-crossfade-ms',type=int,default=0,help='Optional per-voice crossfade length')
    parser.add_argument('--gate-full-rms',type=float,default=0,help='Optional per-voice speech envelope level')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,47}', args.voice):
        parser.error('Voice ID must use lowercase ASCII letters, digits or underscores')
    target_hz = args.target_hz or {"tomori": 255, "anon": 310, "taffy": 429}.get(args.voice, 300)
    shift = None if args.pitch == "auto" else float(args.pitch)
    if (not 80 <= target_hz <= 650 or not 0 <= args.index_rate <= 1
            or not 0 <= args.protect <= .5 or shift is not None and not -24 <= shift <= 24
            or not 0 <= args.stream_lookahead_ms <= 250 or not 0 <= args.stream_crossfade_ms <= 60
            or args.gate_full_rms != 0 and not .00005 < args.gate_full_rms <= .01):
        parser.error("Invalid pitch, index-rate or protect setting")
    python = prepare_export_environment()
    # Inspection runs separately in the existing torch environment. No globals,
    # classes or arbitrary pickle objects are allowed from the supplied checkpoint.
    inspect = ("import json,sys,torch; c=torch.load(sys.argv[1],map_location='cpu',weights_only=True); "
               "print(json.dumps(dict(version=c.get('version'),f0=c.get('f0'),"
               "sample_rate=c['config'][-1],speakers=c['weight']['emb_g.weight'].shape[0])))")
    result = subprocess.run([str(python), "-c", inspect, str(args.checkpoint.resolve())],
                            capture_output=True, text=True, encoding="utf-8", check=True)
    metadata = json.loads(result.stdout)
    if (metadata["version"] != "v2" or metadata["f0"] != 1
            or metadata["sample_rate"] not in (32000, 40000, 48000)
            or not 0 <= args.speaker_id < metadata["speakers"]):
        raise ValueError(f"Unsupported RVC model: {metadata}")
    with args.index.open("rb") as stream:
        index = faiss.read_index(faiss.PyCallbackIOReader(stream.read))
    if index.d != 768 or not index.is_trained or index.ntotal < 8:
        raise ValueError("Expected a trained, populated RVC v2 index with 768-dimensional features")
    metadata.update(index_vectors=index.ntotal, index_dimension=index.d)
    del index
    weight_hash = digest(args.checkpoint)
    folder = DIRECTORY / f"{args.voice}-{weight_hash[:12]}"
    folder.mkdir(parents=True, exist_ok=True)
    weights, index_path = folder / "weights.pth", folder / "voice.index"
    for original, destination in ((args.checkpoint, weights), (args.index, index_path)):
        if original.resolve() != destination.resolve():
            shutil.copyfile(original, destination)
    source = checkout(CACHE / "rvc-src", RVC, RVC_REV)
    name = (folder / "voice").relative_to(ROOT / "models").as_posix()
    subprocess.run([str(python), str(ROOT / "tools/export_voices.py"), args.voice, str(source),
                    str(folder), "--checkpoint", str(weights), "--output-name", name,
                    "--speaker-id", str(args.speaker_id)], cwd=ROOT, check=True)
    graph_path = folder / "voice.onnx"
    session = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
    output = session.run(None, {"units": np.zeros((1, 100, 768), np.float32),
                               "pitch": np.full((1, 100), 80, np.int64),
                               "f0": np.full((1, 100), 220, np.float32),
                               "noise": np.zeros((1, 192, 100), np.float32)})[0]
    if output.size != metadata["sample_rate"] or not np.isfinite(output).all():
        raise ValueError("The exported model failed its one-second inference check")
    entry = {"label": args.name or args.checkpoint.stem, "display_name": args.name or args.checkpoint.stem,
             "graph": graph_path.relative_to(DIRECTORY).as_posix(), "graph_sha256": digest(graph_path),
             "index": index_path.relative_to(DIRECTORY).as_posix(), "index_sha256": digest(index_path),
             "weights_sha256": weight_hash, "source": args.source, "sample_rate": metadata["sample_rate"],
             "target_hz": target_hz, "semitones": shift, "index_rate": args.index_rate,
             "protect": args.protect, "speaker_id": args.speaker_id, "export_recipe": 1,
             "rvc_revision": RVC_REV, **metadata}
    entry.update(stream_lookahead_ms=args.stream_lookahead_ms,stream_crossfade_ms=args.stream_crossfade_ms)
    entry['output_gate_full_rms']=args.gate_full_rms
    registry = json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.exists() else {"version": 1, "voices": {}}
    registry["voices"][args.voice] = entry
    with tempfile.NamedTemporaryFile(dir=DIRECTORY, suffix=".json", delete=False, mode="w", encoding="utf-8") as stream:
        json.dump(registry, stream, ensure_ascii=False, indent=2)
        temporary = Path(stream.name)
    temporary.replace(REGISTRY)
    print(json.dumps({"voice": args.voice, **entry}, ensure_ascii=False, indent=2))
    print("Installed. Restart the backend to use this model.")


if __name__ == "__main__":
    main()
