"""公开版（没有内置角色）的后端：能启动、能报告就绪、角色列表为空，只下载共用模型。

运行：.venv\\Scripts\\python.exe -m unittest tools.eval.test_public_edition -v
每个检查在单独的子进程里跑，因为角色列表在导入 backend.engine 时就确定了。
"""
import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PROBE = textwrap.dedent("""
    import json, time
    from fastapi.testclient import TestClient
    import backend.startup as startup_module
    import backend.app as app_module

    class Graph:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    loaded = []
    def fake_graph(name):
        loaded.append(name)
        return Graph()
    startup_module.graph = fake_graph
    startup_module.pitch_model = lambda: loaded.append("rmvpe")
    app_module.graph = fake_graph

    with TestClient(app_module.app) as client:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            health = client.get("/api/health")
            if health.json().get("status") == "ok":
                break
            time.sleep(.05)
        voices = client.get("/api/voices").json()
        convert = client.post("/api/convert?voice_id=taffy", content=b"x").status_code
    print(json.dumps({"health": health.json(), "code": health.status_code, "voices": voices,
                      "loaded": loaded, "convert": convert}, ensure_ascii=False))
""")


def run_probe(edition: str, local_dir: Path | None = None) -> dict:
    env = dict(os.environ, VOICE_EDITION=edition, VOICE_PREFETCH="0", PYTHONUTF8="1")
    result = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env, capture_output=True, text=True,
                            encoding="utf-8", timeout=120)
    if result.returncode:
        raise AssertionError(result.stderr[-3000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


class PublicEditionTests(unittest.TestCase):
    @unittest.skipIf((ROOT / "models/local-rvc/profiles.json").exists(),
                     "本机已导入声线，空列表的检查只在没有导入声线的环境里跑")
    def test_public_backend_starts_without_any_voice(self):
        out = run_probe("public")
        self.assertEqual(out["code"], 200)
        self.assertEqual(out["health"]["status"], "ok", out["health"])
        self.assertTrue(out["health"]["public_edition"])
        self.assertEqual(out["health"]["voice_models"], {})
        self.assertEqual(out["voices"], [])
        # 只加载共用的降噪、语音检测、特征和音高模型，不碰任何角色或兼容声线
        self.assertEqual(set(out["loaded"]) - {"contentvec_768l12"},
                         {"gtcrn_simple", "silero_vad_16k_op15", "rmvpe"})
        self.assertEqual(out["convert"], 422)  # 不存在的内置角色被拒绝

    def test_setup_models_public_only_fetches_shared_graphs(self):
        script = textwrap.dedent("""
            import json, setup_models
            fetched = []
            setup_models.fetch = lambda path, url, digest: fetched.append(path.name)
            setup_models.prepare_export_environment = lambda: (_ for _ in ()).throw(AssertionError("export"))
            setup_models.main()
            print(json.dumps(sorted(fetched)))
        """)
        env = dict(os.environ, VOICE_EDITION="public", PYTHONUTF8="1")
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True,
                                text=True, encoding="utf-8", timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(json.loads(result.stdout.strip().splitlines()[-1]),
                         ["contentvec_768l12.onnx", "gtcrn_simple.onnx", "rmvpe.onnx", "silero_vad_16k_op15.onnx"])


if __name__ == "__main__":
    unittest.main()
