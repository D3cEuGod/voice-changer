"""Rescore two render folders against real references and disjoint held-out recordings.

Usage: .venv/Scripts/python.exe tools/eval/compare_references.py OLD NEW --heldout DIR --output FILE
OLD and NEW contain female/male/noisy-VOICE-INDEX.wav from eval_render.py.
CAM++ shares a speaker-encoder family with Seed-VC; these are diagnostic scores,
not a substitute for listening to the paired audio.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from eval_render import RATE, ROOT, load, speaker_embedding


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tomori-demo", type=Path, help="Optional user-selected AI example, scored separately from original dialogue")
    args = parser.parse_args()
    rows = []
    for voice in ("tomori", "taffy", "anon"):
        targets = {name: speaker_embedding(*sf.read(folder / f"{voice}.wav", dtype="float32"))
                   for name, folder in (("prompt", ROOT / "assets/references"), ("heldout", args.heldout))}
        if voice == "tomori" and args.tomori_demo is not None:
            targets["user_demo"] = speaker_embedding(*sf.read(args.tomori_demo, dtype="float32"))
        for condition in ("female", "male", "noisy"):
            row = {"voice": voice, "input": condition}
            for version, folder in (("old", args.old), ("new", args.new)):
                files = sorted(folder.glob(f"{condition}-{voice}-[1-8].wav"))
                if len(files) != 8:
                    raise ValueError(f"Expected eight sentences: {folder}/{condition}-{voice}")
                vector = speaker_embedding(np.concatenate([load(p) for p in files]), RATE)
                for name, target in targets.items():
                    row[f"{version}_{name}"] = round(float(vector @ target), 3)
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
