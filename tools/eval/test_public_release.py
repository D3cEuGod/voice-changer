"""公开版导出脚本：排除角色素材、覆盖公开版文件、逐个文件检查。

运行：.venv\\Scripts\\python.exe -m unittest tools.eval.test_public_release -v
导出的是 HEAD，所以要先提交 tools/public/ 和导出脚本本身的改动。
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools import build_public_release as build  # noqa: E402


class ExcludeRuleTests(unittest.TestCase):
    def test_rules(self):
        for path in ("assets/taffy-avatar.jpg", "assets/tomori-avatar.png", "assets/taffy-chibi.png",
                     "assets/references/anon.wav", "assets/previews/taffy.wav", "assets/previews/long/input.wav",
                     "notes/decisions.md", "docs/images/main-window.png", "voices/taffy.npy",
                     "third_party/installers/vb-cable/VBCABLE_Driver_Pack46.zip", "third_party/seed_vc/LICENSE",
                     "tools/public/README.md", "prepare_voices.py"):
            self.assertTrue(build.excluded(path), path)
        for path in ("assets/previews/original.wav", "assets/app-icon.png", "assets/characters.json",
                     "gui/app.py", "backend/engine.py", "tools/import_rvc.py", "tools/export_voices.py",
                     "third_party/rvc/LICENSE", "models/LICENSE.txt"):
            self.assertFalse(build.excluded(path), path)

    def test_audit_flags_weights_media_and_large_files(self):
        out = Path(tempfile.mkdtemp())
        for name in ("PUBLIC_EDITION", "LICENSE", "README.md", "THIRD_PARTY_NOTICES.md", "assets/characters.json"):
            (out / name).parent.mkdir(parents=True, exist_ok=True)
            (out / name).write_text("x", encoding="utf-8")
        self.assertEqual(build.audit(out), [])
        (out / "models").mkdir()
        (out / "models/voice.pth").write_bytes(b"x")
        (out / "assets/face.png").write_bytes(b"x")
        (out / "big.txt").write_bytes(b"x" * (build.MAX_BYTES + 1))
        self.assertEqual(sorted(p.split(":")[0] for p in build.audit(out)),
                         ["assets/face.png", "big.txt", "models/voice.pth"])


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp()) / "public"
        build.main([str(cls.out), "--git"])

    def test_only_public_files(self):
        self.assertEqual(build.audit(self.out), [])
        for gone in ("notes", "docs", "voices", "assets/references", "third_party/installers",
                     "third_party/seed_vc", "tools/public", "assets/taffy-avatar.jpg"):
            self.assertFalse((self.out / gone).exists(), gone)
        self.assertTrue((self.out / "assets/previews/original.wav").is_file())
        self.assertEqual(sorted(p.name for p in (self.out / "assets").iterdir()),
                         ["app-icon.ico", "app-icon.png", "characters.json", "previews"])

    def test_overlay_replaced_private_files(self):
        self.assertIn("MIT License", (self.out / "LICENSE").read_text(encoding="utf-8"))
        readme = (self.out / "README.md").read_text(encoding="utf-8")
        self.assertIn("这个仓库不带任何声线", readme)
        for private in ("D:\\Projects", "deng.pth", "WO Mic"):
            self.assertNotIn(private, readme)
        self.assertIn('"voices": {}', (self.out / "assets/characters.json").read_text(encoding="utf-8"))

    def test_backend_runs_as_public_edition(self):
        code = "from backend.edition import PUBLIC; from backend import engine; print(PUBLIC, sorted(engine.VOICES))"
        result = subprocess.run([sys.executable, "-c", code], cwd=self.out, capture_output=True, text=True,
                                encoding="utf-8", timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(result.stdout.split()[0], "True")

    def test_fresh_history(self):
        log = subprocess.run(["git", "log", "--oneline"], cwd=self.out, capture_output=True, text=True,
                             encoding="utf-8", check=True).stdout.splitlines()
        self.assertEqual(len(log), 1, log)

    def test_refuses_non_empty_or_inside_project(self):
        with self.assertRaises(SystemExit):
            build.main([str(self.out)])
        with self.assertRaises(SystemExit):
            build.main([str(ROOT / ".run" / "public-test")])


if __name__ == "__main__":
    unittest.main()
