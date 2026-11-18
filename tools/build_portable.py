r"""打包 Windows 便携版：解压后双击 start-gui.cmd 就能用，不需要先装 Python。

  python tools\build_portable.py                     # 输出到 dist\VoiceChanger-<版本>-win64.zip
  python tools\build_portable.py --python-zip D:\python-3.13.9-embed-amd64.zip

包里有什么：
- 公开版源代码（在私人仓库里运行时先按 tools/build_public_release.py 的规则导出；在公开版里运行时直接用当前提交）；
- runtime\：Python 官方的 Windows 嵌入式发行版（embeddable package），加上 requirements.txt、
  gui/requirements.txt 和 pip 的全部依赖（不含 GPL-3.0 的 praat-parselmouth，公开版用不到它）。
  start-gui.cmd / start-backend.cmd 看到 runtime\python.exe 就直接用它。

包里没有什么：模型（第一次启动时下载并校验 SHA-256）、声线、VB-CABLE。

依赖用 pip 的 --platform win_amd64 下载现成的 wheel，所以在 Linux 上也能打包；
嵌入式 Python 默认从 python.org 下载，版本和运行本脚本的 Python 相同，也可以用 --python-zip 指定本地文件。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_URL = "https://www.python.org/ftp/python/{v}/python-{v}-embed-amd64.zip"
WHEEL_TARGET = ["--platform", "win_amd64", "--python-version", "3.13", "--implementation", "cp",
                "--only-binary=:all:"]
# 不放进便携版的依赖：Praat（GPL-3.0）只给旧版兼容声线用，导入的 RVC 声线用 RMVPE 提取音高
LEFT_OUT = {"praat-parselmouth"}
# 源代码里不该出现的文件（runtime\ 里的 .dll / .exe / .pth 是 Python 和依赖本身，不在检查范围内）
BANNED_SUFFIXES = {".pth", ".pt", ".ckpt", ".safetensors", ".index", ".onnx", ".npy", ".npz",
                   ".zip", ".7z", ".rar", ".exe", ".msi", ".dll",
                   ".wav", ".flac", ".mp3", ".ogg", ".m4a", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico"}
ALLOWED = {"assets/previews/original.wav", "assets/app-icon.png", "assets/app-icon.ico"}
README_FIRST = """角色变声器 {version}（Windows 便携版）

1. 把整个文件夹放到一个路径里没有中文和空格的地方，例如 D:\\VoiceChanger。
2. 双击 start-gui.cmd。第一次启动会下载共用的推理模型（需要联网，会校验 SHA-256），
   右上角出现“引擎就绪”后就能用了。
3. 点左栏的“＋ 导入声线”，装入 RVC v2 声线包（.pth 模型 + .index 索引）。
   第一次导入要额外下载约 2 GB 的导出工具。
4. 想发到 Discord 等语音软件，需要自己从 https://vb-audio.com/Cable/ 安装 VB-CABLE。

完整说明见 README.md。本软件使用 MIT 许可，第三方组件见 THIRD_PARTY_NOTICES.md。
请只导入你有权使用的声线，不要用它冒充他人。
"""


def source_files(work: Path) -> Path:
    """把公开版源代码放到 work/src，返回这个目录。只取已提交的文件。"""
    src = work / "src"
    if (ROOT / "PUBLIC_EDITION").exists():
        data = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True).stdout
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archive.extractall(src, filter="data")
    else:
        sys.path.insert(0, str(ROOT))
        from tools import build_public_release
        src.mkdir(parents=True)
        build_public_release.export("HEAD", src)
        problems = build_public_release.audit(src)
        if problems:
            raise SystemExit("公开版检查没通过：\n  " + "\n  ".join(problems))
    return src


def fetch_python(version: str, given: str | None, cache: Path) -> Path:
    if given:
        return Path(given)
    url = PYTHON_URL.format(v=version)
    target = cache / Path(url).name
    if not target.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        print("下载", url)
        with urllib.request.urlopen(url) as response, open(target.with_suffix(".part"), "wb") as out:
            shutil.copyfileobj(response, out)
        target.with_suffix(".part").replace(target)
    return target


def runtime_requirements(app: Path) -> list[str]:
    """后端和界面两份依赖合在一起，去掉 LEFT_OUT 里的包。"""
    pins = []
    for path in (app / "requirements.txt", app / "gui" / "requirements.txt"):
        pins += [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip() and line.split("==")[0].strip().lower() not in LEFT_OUT]
    return sorted(set(pins))


def make_runtime(app: Path, python_zip: Path) -> None:
    runtime = app / "runtime"
    with zipfile.ZipFile(python_zip) as archive:
        archive.extractall(runtime)
    names = [p.name for p in runtime.glob("python3*._pth")]
    zipped = [p.name for p in runtime.glob("python3*.zip")]
    if len(names) != 1 or len(zipped) != 1 or not (runtime / "python.exe").is_file():
        raise SystemExit(f"{python_zip} 不像是 Python 的 Windows 嵌入式发行版")
    # 搜索路径：标准库、runtime 本身、装依赖的 site-packages、项目根目录（..）。import site 让 .pth 生效。
    (runtime / names[0]).write_text(f"{zipped[0]}\n.\nLib\\site-packages\n..\nimport site\n", encoding="utf-8")
    site = runtime / "Lib" / "site-packages"
    site.mkdir(parents=True)
    command = [sys.executable, "-m", "pip", "install", "--no-compile", "--disable-pip-version-check",
               "--target", str(site), *WHEEL_TARGET, "pip", *runtime_requirements(app)]
    print("安装依赖到 runtime\\Lib\\site-packages …")
    subprocess.run(command, check=True)
    for cache in site.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    shutil.rmtree(site / "bin", ignore_errors=True)  # pip 在 Linux 上生成的脚本，Windows 用不到


def audit(app: Path) -> list[str]:
    problems = []
    for path in sorted(app.rglob("*")):
        relative = path.relative_to(app).as_posix()
        if not path.is_file() or relative.startswith("runtime/") or relative in ALLOWED:
            continue
        if path.suffix.lower() in BANNED_SUFFIXES:
            problems.append(f"{relative}: 不允许的文件类型")
    for name in ("PUBLIC_EDITION", "LICENSE", "start-gui.cmd", "runtime/python.exe", "runtime/pythonw.exe"):
        if not (app / name).is_file():
            problems.append(f"{name}: 缺少")
    return problems


def write_zip(app: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".part")
    with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(app.rglob("*")):
            if path.is_file():
                archive.write(path, Path(app.name) / path.relative_to(app))
    part.replace(target)
    digest = hashlib.sha256()
    with open(target, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python-version", default=platform.python_version(),
                        help="嵌入式 Python 的版本，默认和运行本脚本的 Python 相同（必须是 3.13.x）")
    parser.add_argument("--python-zip", help="本地的 python-3.13.x-embed-amd64.zip，不从 python.org 下载")
    parser.add_argument("--out", type=Path, default=ROOT / "dist", help="输出文件夹，默认 dist")
    args = parser.parse_args(argv)
    if not args.python_zip and not args.python_version.startswith("3.13."):
        parser.error("依赖是按 Python 3.13 选的：请用 Python 3.13 运行本脚本，或用 --python-version 3.13.x 指定")
    with tempfile.TemporaryDirectory(prefix="voice-portable-") as temp:
        work = Path(temp)
        src = source_files(work)
        version = (src / "VERSION").read_text(encoding="utf-8").strip()
        app = work / f"VoiceChanger-{version}"
        src.rename(app)
        make_runtime(app, fetch_python(args.python_version, args.python_zip, ROOT / ".run" / "portable-cache"))
        (app / "先看我.txt").write_text(README_FIRST.format(version=version), encoding="utf-8-sig")
        problems = audit(app)
        if problems:
            raise SystemExit("便携版检查没通过：\n  " + "\n  ".join(problems))
        target = args.out.resolve() / f"VoiceChanger-{version}-win64.zip"
        digest = write_zip(app, target)
    size = target.stat().st_size / (1 << 20)
    print(f"完成：{target}（{size:.0f} MB）\nSHA-256：{digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
