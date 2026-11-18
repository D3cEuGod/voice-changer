"""Regenerate both GUI presets and level-matched reference / old / new comparisons."""
import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gui.backend import BackendClient


def load(data, limit=6):
    samples, rate = sf.read(BytesIO(data), dtype="float32", always_2d=True)
    divisor = gcd(rate, 22050)
    samples = resample_poly(samples.mean(axis=1), 22050//divisor, rate//divisor)[:limit*22050]
    active = samples[abs(samples) > .005]
    rms = float(np.sqrt(np.mean(active**2))) if active.size else 0
    return samples * min(.1/max(rms, 1e-9), .9/max(float(abs(samples).max()), 1e-9))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--old-revision", default="a9d9139977bbc0ab257876fe0b1f60c4abac9078")
    args = parser.parse_args()
    client = BackendClient()
    previews, comparisons = ROOT / "assets/previews", ROOT / "assets/comparisons"
    comparisons.mkdir(exist_ok=True)
    original = (previews / "original.wav").read_bytes()
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "backend": client.health(),
                "input_sha256": hashlib.sha256(original).hexdigest(), "files": []}
    metadata = []
    for voice in ("tomori", "taffy", "anon"):
        for mode in ("character", "call"):
            path = previews / f"{voice}{'-call' if mode == 'call' else ''}.wav"
            result = client.convert(original, "audio/wav", voice, 1, mode=mode)
            path.write_bytes(result)
            manifest["files"].append({"file": path.name, "voice": voice, "mode": mode,
                                      "model": manifest["backend"].get("voice_models", {}).get(voice),
                                      "sha256": hashlib.sha256(result).hexdigest()})
            print(path.name, flush=True)
        old = subprocess.run(["git", "show", f"{args.old_revision}:assets/previews/{voice}.wav"],
                             cwd=ROOT, capture_output=True, check=True).stdout
        demo = "tomori-user-demo" if voice == "tomori" else voice
        reference = (ROOT / f"assets/references/{demo}.wav").read_bytes()
        pieces, segments, position = [], [], 0
        for label, data in (("user_selected_singing_reference" if voice == "tomori" else "original_character", reference),
                            ("v011", old), ("v020_RVC", (previews / f"{voice}.wav").read_bytes())):
            clip = load(data)
            segments.append({"label": label, "start_s": round(position/22050, 3),
                             "end_s": round((position+len(clip))/22050, 3)})
            pieces.extend((clip, np.zeros(14333, np.float32)))
            position += len(clip)+14333
        sf.write(comparisons / f"{voice}.wav", np.concatenate(pieces), 22050)
        metadata.append({"voice": voice, "segments": segments})
    (comparisons / "segments.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (previews / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
