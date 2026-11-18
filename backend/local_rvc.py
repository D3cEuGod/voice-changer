"""Verified, user-installed RVC v2 speakers. Weights are never unpickled by the server."""
from dataclasses import dataclass
from functools import lru_cache
import json
import re
from pathlib import Path

from tools.model_integrity import verified

ROOT = Path(__file__).resolve().parents[1]

DIRECTORY = ROOT / "models/local-rvc"
REGISTRY = DIRECTORY / "profiles.json"


def installed_voices():
    """Only installed custom speakers become API choices on this computer."""
    if not REGISTRY.exists():
        return {}
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry.get('version') != 1:
        raise ValueError('Unsupported local RVC registry version')
    return {key: str(value.get('display_name') or value.get('label') or key)
            for key, value in registry.get('voices', {}).items()
            if re.fullmatch(r'[a-z][a-z0-9_]{0,47}', key)}


@dataclass(frozen=True)
class Profile:
    label: str
    graph: str
    index: Path
    sample_rate: int
    target_hz: float
    semitones: float | None
    index_rate: float
    protect: float
    source: str
    stream_lookahead_ms: int = 0
    stream_crossfade_ms: int = 0
    output_gate_full_rms: float = 0


def verified_file(relative, expected):
    path = (DIRECTORY / relative).resolve()
    if not path.is_relative_to(DIRECTORY.resolve()) or not path.is_file():
        raise ValueError(f"Installed RVC file is missing or outside its directory: {relative}")
    if not verified(path, expected):
        raise ValueError(f"Installed RVC file failed its SHA-256 check: {relative}")
    return path


@lru_cache(maxsize=None)
def profile(voice):
    if not REGISTRY.exists():
        return None
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry.get("version") != 1:
        raise ValueError("Unsupported local RVC registry version")
    data = registry.get("voices", {}).get(voice)
    if data is None:
        return None
    graph = verified_file(data["graph"], data["graph_sha256"])
    index = verified_file(data["index"], data["index_sha256"])
    rate, target = int(data["sample_rate"]), float(data["target_hz"])
    ratio, protect = float(data["index_rate"]), float(data["protect"])
    lookahead, crossfade = int(data.get('stream_lookahead_ms',0)), int(data.get('stream_crossfade_ms',0))
    gate_full=float(data.get('output_gate_full_rms',0))
    shift = data.get("semitones")
    if (rate not in (32000, 40000, 48000) or not 80 <= target <= 650
            or not 0 <= ratio <= 1 or not 0 <= protect <= .5
            or shift is not None and not -24 <= float(shift) <= 24
            or not 0 <= lookahead <= 250 or not 0 <= crossfade <= 60
            or gate_full != 0 and not .00005 < gate_full <= .01):
        raise ValueError("Invalid local RVC inference settings")
    return Profile(str(data.get("label", voice)), graph.relative_to(ROOT / "models").as_posix(), index, rate, target,
                   None if shift is None else float(shift), ratio, protect, data.get("source", "user model"),
                   lookahead,crossfade,gate_full)


@lru_cache(maxsize=None)
def graph(voice):
    from .engine import load_graph
    return load_graph(profile(voice).graph)
