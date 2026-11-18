"""便携版打包脚本的纯逻辑部分：依赖清单、文件检查。完整打包要下载约 200 MB 的 wheel，在发布时单独跑。

运行：.venv\\Scripts\\python.exe -m unittest tools.eval.test_portable -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools import build_portable  # noqa: E402


class PortableTests(unittest.TestCase):
    def test_requirements_merge_and_leave_out_praat(self):
        pins = build_portable.runtime_requirements(ROOT)
        names = [p.split("==")[0].lower() for p in pins]
        self.assertNotIn("praat-parselmouth", names)
        for needed in ("onnxruntime-directml", "pyside6-essentials", "fastapi", "uvicorn", "sounddevice", "soxr"):
            self.assertIn(needed, names)
        self.assertEqual(len(names), len(set(names)))  # numpy 等两份清单都有的只出现一次
        self.assertTrue(all("==" in p for p in pins))

    def test_backend_imports_without_praat(self):
        source = (ROOT / "backend" / "character.py").read_text(encoding="utf-8")
        top = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
        self.assertFalse(any("parselmouth" in line for line in top), top)

    def test_audit(self):
        app = Path(tempfile.mkdtemp())
        for name in ("PUBLIC_EDITION", "LICENSE", "start-gui.cmd", "runtime/python.exe", "runtime/pythonw.exe",
                     "runtime/Lib/site-packages/numpy/core.dll", "runtime/Lib/site-packages/distutils-precedence.pth",
                     "assets/app-icon.png", "assets/previews/original.wav"):
            (app / name).parent.mkdir(parents=True, exist_ok=True)
            (app / name).write_bytes(b"x")
        self.assertEqual(build_portable.audit(app), [])
        (app / "models").mkdir()
        (app / "models/voice.pth").write_bytes(b"x")
        (app / "assets/face.jpg").write_bytes(b"x")
        (app / "runtime/pythonw.exe").unlink()
        self.assertEqual(sorted(p.split(":")[0] for p in build_portable.audit(app)),
                         ["assets/face.jpg", "models/voice.pth", "runtime/pythonw.exe"])

    def test_launchers_prefer_bundled_runtime(self):
        gui = (ROOT / "start-gui.cmd").read_text(encoding="utf-8")
        backend = (ROOT / "start-backend.cmd").read_text(encoding="utf-8")
        self.assertLess(gui.index("runtime\\pythonw.exe"), gui.index("py -3.13"))
        self.assertLess(backend.index("runtime\\python.exe"), backend.index("py -3.13"))
        self.assertIn('if exist "setup_reference_engine.py"', backend)


if __name__ == "__main__":
    unittest.main()
