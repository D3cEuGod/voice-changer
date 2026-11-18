"""Provision the verified runtime graphs or build missing character graphs once."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.request import urlopen

from tools.model_integrity import verified

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
CACHE = ROOT / ".run"
WEIGHTS = CACHE / "rvc-input"
HF = "https://huggingface.co/"
SVC = HF + "spaces/ZilliaxOfficial/nyaru-svc-3.0"
SVC_REV = "2ded8640113ea4faec5da58426ad401e71da2885"
RVC = "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git"
RVC_REV = "d60cd3d009b7e438f92f8624fa89b710d10c6a39"
TONE = HF + "TigreGotico/voiceclonnx-openvoice-v2/resolve/34d010c192c97f763207f488f6057fd07fee42ad/"

GRAPHS = {
    "rmvpe.onnx": (
        HF + "TigreGotico/voiceclonnx-rvc/resolve/cbfbabdabe6a1414292bade8f857fb6074abf130/rmvpe.onnx",
        "5370e71ac80af8b4b7c793d27efd51fd8bf962de3a7ede0766dac0befa3660fd"),
    "silero_vad_16k_op15.onnx": (
        "https://raw.githubusercontent.com/snakers4/silero-vad/066409f3d9db45a30d21dfcbf9147826ffdd7052/src/silero_vad/data/silero_vad_16k_op15.onnx",
        "7ed98ddbad84ccac4cd0aeb3099049280713df825c610a8ed34543318f1b2c49"),
    "tomori.index": (HF + "TogetsuDo/tomori-rvc/resolve/38147a63386d278c60abf0e7f4c48559b1d2ec5d/tomori.index",
                     "89355ffbb3da0de5e8d236e3eba4b817efd3fc8e4880bc6c966f284ad9cdb6b7"),
    "anon.index": (HF + "TogetsuDo/anon-rvc/resolve/d530738e001994d21a9744a7100ff36e745f329e/anon.index",
                   "e742985d05fc4c201fe68a3df48f2533c2ef3f6864b8d4162b933840457c7827"),
    "tone_converter.onnx": (TONE + "tone_converter.onnx", "7d7ee834c230037ead5cd6b64d44fcd842fbab5cd4cf6fe3ab381fad763325e9"),
    "tone_ref_encoder.onnx": (TONE + "tone_ref_encoder.onnx", "3dd4918cab90e1acf7fa5c6f7539c27710e7a3cdfba550468c5ea49399178bf7"),
    "contentvec_768l12.onnx": (
        HF + "TigreGotico/voiceclonnx-rvc/resolve/cbfbabdabe6a1414292bade8f857fb6074abf130/contentvec_768l12.onnx",
        "cbddd8fa9352b3128df6359e2e3da6be0f9072e768e51e3efc3004e5030ea97f"),
    "gtcrn_simple.onnx": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx",
        "e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534"),
}
CHECKPOINTS = {
    "tomori.pth": (HF + "TogetsuDo/tomori-rvc/resolve/38147a63386d278c60abf0e7f4c48559b1d2ec5d/tomori.pth",
                   "7a19e21e455bc0ffc8af9219dd3aabc0cba9d4c6b9cde4e92a37b17b526ff57e"),
    "anon.pth": (HF + "TogetsuDo/anon-rvc/resolve/d530738e001994d21a9744a7100ff36e745f329e/anon.pth",
                 "9ce272c6c32faf68c97948aa80f2ffc57d208315ac34472729ba6d7e7459da62"),
    "taffy.pth": (f"{SVC}/resolve/{SVC_REV}/logs/32k/NyaruTaffy.pth",
                  "5b17b60776e745173d9ceab0eccaa9b35cbfe71a5294a6ffa173d46b4e38ee31"),
    "hubert.pt": (f"{SVC}/resolve/{SVC_REV}/hubert/hubert-soft-0d54a1f4.pt",
                  "e82e7d079df05fe3aa535f6f7d42d309bdae1d2a53324e2b2386c56721f4f649"),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(path, url, expected):
    if verified(path, expected):
        return
    print(f"Downloading {path.name} ...", flush=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as tmp:
        part = Path(tmp.name)
    try:
        with urlopen(url, timeout=120) as stream, part.open("wb") as out:
            shutil.copyfileobj(stream, out, length=1024 * 1024)
        if digest(part) != expected:
            raise RuntimeError(f"SHA-256 mismatch for {path.name}")
        part.replace(path)
        verified(path, expected)
    finally:
        part.unlink(missing_ok=True)


def checkout(folder, remote, revision):
    folder.mkdir(exist_ok=True)
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    if (folder / ".git").is_dir():
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=folder, capture_output=True, text=True)
        if head.stdout.strip() == revision:
            return folder
    else:
        subprocess.run(["git", "init", "--quiet"], cwd=folder, check=True)
    subprocess.run(["git", "fetch", "--depth", "1", remote, revision], cwd=folder, env=env, check=True)
    subprocess.run(["git", "checkout", "--detach", revision], cwd=folder, env=env, check=True)
    return folder


def prepare_export_environment():
    """Prepare only the model-export tools, without downloading compatibility speakers."""
    CACHE.mkdir(parents=True, exist_ok=True)
    env = CACHE / "export-env"
    python = env / "Scripts/python.exe" if os.name == "nt" else env / "bin/python"
    if not python.is_file():
        subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
    pins = ["torch==2.15.0", "onnx==1.23.2", "scipy==1.18.1", "matplotlib==3.11.2"]
    check = ("from importlib.metadata import version; "
             f"assert all(version(p)==v for p,v in (item.split('==') for item in {pins!r}))")
    if subprocess.run([str(python), "-c", check], capture_output=True).returncode:
        subprocess.run([str(python), "-m", "pip", "install", "--no-cache-dir", *pins], check=True)
    return python


def main():
    MODELS.mkdir(exist_ok=True)
    from backend.edition import PUBLIC
    if PUBLIC:  # 公开版只下载共用的推理图，不下载任何角色模型
        for file in ("contentvec_768l12.onnx", "rmvpe.onnx", "gtcrn_simple.onnx", "silero_vad_16k_op15.onnx"):
            fetch(MODELS / file, *GRAPHS[file])
        return
    from backend.local_rvc import profile
    fully_local = all(profile(voice) is not None for voice in ("tomori", "taffy", "anon"))
    required = {"contentvec_768l12.onnx", "rmvpe.onnx", "gtcrn_simple.onnx", "silero_vad_16k_op15.onnx"}
    for file, source in GRAPHS.items():
        if not fully_local or file in required:
            fetch(MODELS / file, *source)
    if fully_local:
        return
    from tools.svc_phase import correct_cached_phase
    outputs = {"tomori": ["tomori_rvc"], "anon": ["anon_rvc"], "taffy": ["taffy_svc", "hubert_soft"]}
    missing = [role for role, graphs in outputs.items()
               if any(not (MODELS / (name + ".onnx")).is_file() for name in graphs)]
    if not missing:
        correct_cached_phase(MODELS / "taffy_svc.onnx")
        return
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    python = prepare_export_environment()
    for role in missing:
        files = [role + ".pth"]
        if role == "taffy":
            files.append("hubert.pt")
            source = checkout(CACHE / "svc-src", SVC, SVC_REV)
        else:
            source = checkout(CACHE / "rvc-src", RVC, RVC_REV)
        for file in files:
            fetch(WEIGHTS / file, *CHECKPOINTS[file])
        print(f"Exporting {role} ...", flush=True)
        subprocess.run([str(python), str(ROOT / "tools/export_voices.py"), role, str(source),
                        str(WEIGHTS)], cwd=ROOT, check=True)
    correct_cached_phase(MODELS / "taffy_svc.onnx")


if __name__ == "__main__":
    main()
