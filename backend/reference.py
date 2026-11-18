"""Local process and audio transport for the Seed-VC CUDA worker."""
import atexit
from functools import lru_cache
import http.client
import json
import logging
import os
from pathlib import Path
import subprocess
import socket
import tempfile
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_SAMPLES = 66591  # 3.02 seconds at 22,050 Hz
OUTPUT_START = 44100
OUTPUT_SAMPLES = 22050
# Held-out original dialogue favored Seed-VC for these voices. Anon's dedicated
# RVC remains the default: Seed-VC lost speaker identity on her high-register lines.
PREFERRED_VOICES = frozenset(("tomori", "taffy"))


class ReferenceClient:
    def __init__(self):
        self.process = None
        self.port = None
        self.failure = ""
        self._lock = threading.Lock()
        self._ready_file = None
        self._log = None
        atexit.register(self.close)

    def start(self):
        with self._lock:
            if self.port is not None:
                if self.process is not None and self.process.poll() is None:
                    return True
                self.failure = "Seed-VC 进程已退出，请重启后端并查看 .run/reference-worker.log。"
                self.close()
                return False
            if self.failure or os.environ.get("VOICE_REFERENCE_ENGINE") != "seed":
                return False
            python = ROOT / ".run/seed-env/Scripts/python.exe"
            if not python.is_file() or not (ROOT / "models/seed-vc/tiny/DiT_uvit_tat_xlsr_ema.pth").is_file():
                self.failure = "Seed-VC 尚未安装。"
                return False
            run = ROOT / ".run"
            with tempfile.NamedTemporaryFile(dir=run, prefix="reference-", suffix=".json", delete=False) as ready:
                self._ready_file = Path(ready.name)
            log_path = run / "reference-worker.log"
            self._log = log_path.open("ab")
            self.process = subprocess.Popen(
                [str(python), "-u", "-m", "backend.reference_worker", "--ready-file", str(self._ready_file)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                env=dict(os.environ, PYTHONUTF8="1", OPENBLAS_NUM_THREADS="1"))
            deadline = time.monotonic() + 120
            while self.process.poll() is None and time.monotonic() < deadline:
                try:
                    ready = json.loads(self._ready_file.read_text(encoding="utf-8"))
                    self.port = int(ready["port"])
                    return True
                except (OSError, ValueError, KeyError):
                    time.sleep(.1)
            self.failure = "Seed-VC 启动失败，请查看 .run/reference-worker.log。"
            logging.warning(self.failure)
            self.close()
            return False

    def convert(self, audio, voice):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            connection.connect()
            connection.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connection.request("POST", f"/convert?voice={voice}", body=np.asarray(audio, dtype="<f4").tobytes(),
                               headers={"Content-Type": "application/octet-stream"})
            response = connection.getresponse()
            body = response.read()
            if response.status != 200:
                raise RuntimeError(f"Seed-VC 转换失败：{body.decode('utf-8', errors='replace')[:300]}")
            result = np.frombuffer(body, dtype="<f4").copy()
            if result.size != OUTPUT_SAMPLES or not np.isfinite(result).all():
                raise RuntimeError("Seed-VC 返回的音频不完整。")
            return result
        finally:
            connection.close()

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None
        self.port = None
        if self._ready_file is not None:
            self._ready_file.unlink(missing_ok=True)
            self._ready_file = None
        if self._log is not None:
            self._log.close()
            self._log = None


@lru_cache(maxsize=1)
def client():
    return ReferenceClient()
