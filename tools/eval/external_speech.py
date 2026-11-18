"""Convert the same external speech samples through an old or new running server.

Example: .venv/Scripts/python.exe tools/eval/external_speech.py .run/seed-src/examples/source .run/human/old
Input source_s3.wav is Mandarin narration; source_s4.wav is Mandarin dramatic speech.
The upstream examples do not establish raw microphone provenance for every clip.
These samples are from Seed-VC revision 51383efd921027683c89e5348211d93ff12ac2a8.
Recordings remain in the local .run analysis directory.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gui.backend import BackendClient


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    client = BackendClient()
    manifest = {"server": client.health(), "inputs": []}
    for filename in ("source_s3.wav", "source_s4.wav"):
        data = (args.source / filename).read_bytes()
        manifest["inputs"].append({"file": filename, "sha256": hashlib.sha256(data).hexdigest()})
        for voice in ("tomori", "taffy", "anon"):
            result = client.convert(data, "audio/wav", voice, 1, mode="character")
            (args.output / f"{Path(filename).stem}-{voice}.wav").write_bytes(result)
            print(filename, voice, len(result), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
