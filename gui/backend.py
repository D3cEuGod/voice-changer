"""本地后端的 HTTP 客户端与启动器。接口以 README.md 为准。"""
from __future__ import annotations

import http.client
import json
import os
import re
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlencode, urlsplit

DEFAULT_URL = "http://127.0.0.1:8765"
ROOT = Path(__file__).resolve().parents[1]
START_SCRIPT = ROOT / "start-backend.cmd"
# 启动脚本和后端打印的进度行：“VOICE_STARTUP <阶段名或 0–1 的进度> <说明>”
STARTUP_LINE = re.compile(r"VOICE_STARTUP\s+(\S+)\s+(.*)")
WARM_FIRST = "taffy"  # 界面一打开显示的角色；后端先加载它，其余角色在后台加载

CONTENT_TYPES = {".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg", ".ogg": "audio/ogg"}
PITCH_PRESETS = {
    "model": ("模型默认", None, 0),
    "auto2": ("自动 +2", None, 2),
    "auto4": ("自动 +4", None, 4),
}
DEFAULT_PITCH_PRESET = 'auto2'


def pitch_query(preset):
    _, shift, offset = PITCH_PRESETS[preset]
    query = {}
    if shift is not None:
        query['pitch_shift'] = shift
    if offset:
        query['pitch_offset'] = offset
    return query

STATUS_HINTS = {
    413: "文件过大",
    422: "音频或参数无效",
    429: "引擎正忙",
    500: "推理失败",
}


class BackendError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _detail_text(body: bytes) -> str:
    try:
        detail = json.loads(body.decode("utf-8")).get("detail")
    except Exception:
        return ""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):  # 参数验证错误
        parts = []
        for item in detail:
            if isinstance(item, dict):
                where = ".".join(str(x) for x in item.get("loc", ()) if x not in ("query", "body"))
                parts.append(f"{where}: {item.get('msg', '')}" if where else str(item.get("msg", "")))
            else:
                parts.append(str(item))
        return "；".join(p for p in parts if p)
    return ""


class BackendClient:
    def __init__(self, base_url: str = DEFAULT_URL):
        parts = urlsplit(base_url)
        if parts.scheme != "http" or not parts.hostname:
            raise ValueError(f"后端地址无效：{base_url}")
        self.base_url = base_url.rstrip("/")
        self.host = parts.hostname
        self.port = parts.port or 80

    def _request(self, method: str, path: str, body: bytes | None = None,
                 headers: dict | None = None, timeout: float = 5.0,
                 connect_timeout: float | None = None) -> bytes:
        # http.client 不读系统代理设置，本机回环地址不会被代理软件接管。
        connection = http.client.HTTPConnection(self.host, self.port,
                                                timeout=timeout if connect_timeout is None else connect_timeout)
        try:
            if connect_timeout is not None:
                connection.connect()
                connection.sock.settimeout(timeout)
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            data = response.read()
            status = response.status
        except ConnectionRefusedError as exc:
            raise BackendError(f"无法连接 {self.base_url}（连接被拒绝），后端没有在运行。") from exc
        except TimeoutError as exc:
            if connect_timeout is not None and connection.sock is None:
                raise BackendError(f"无法连接 {self.base_url}（连接超时）。") from exc
            raise BackendError(f"后端 {self.base_url} 在 {timeout:g} 秒内没有响应。") from exc
        except (OSError, http.client.HTTPException) as exc:
            raise BackendError(f"与后端 {self.base_url} 通信失败：{exc}") from exc
        finally:
            connection.close()
        if status != 200:
            hint = STATUS_HINTS.get(status, "请求失败")
            detail = _detail_text(data)
            raise BackendError(f"{hint}（HTTP {status}）{'：' + detail if detail else ''}", status)
        return data

    def health(self, timeout: float = 4.0, *, connect_timeout: float | None = None) -> dict:
        # Startup probes use a short connect timeout; an established connection
        # still gets the full HTTP response timeout.
        data = self._request("GET", "/api/health", timeout=timeout, connect_timeout=connect_timeout)
        try:
            info = json.loads(data.decode("utf-8"))
            # "loading"：后端已经在监听，模型还在加载，info 里有 stage 和 progress。
            if info.get("status") not in ("ok", "loading"):
                raise ValueError
        except Exception as exc:
            raise BackendError(
                f"{self.base_url}/api/health 返回了意外内容，端口可能被其他程序占用。") from exc
        return info

    def voices(self, timeout: float = 5.0) -> list[dict]:
        data = self._request("GET", "/api/voices", timeout=timeout)
        try:
            voices = json.loads(data.decode("utf-8"))
            return [{"id": str(v["id"]), "name": str(v["name"])} for v in voices]
        except Exception as exc:
            raise BackendError("GET /api/voices 返回的角色列表格式不对。") from exc

    def convert(self, data: bytes, content_type: str, voice_id: str, strength: float,
                timeout: float = 600.0, *, mode: str = "character", pitch_preset: str = "model") -> bytes:
        query = urlencode({"voice_id": voice_id, "strength": f"{strength:.2f}", "mode": mode,
                           **pitch_query(pitch_preset)})
        return self._request("POST", f"/api/convert?{query}", body=data,
                             headers={"Content-Type": content_type}, timeout=timeout)

    def live_url(self, voice_id: str, strength: float, mode: str = "character", *, packet_ms: int = 200,
                 pitch_preset: str = "model") -> str:
        if packet_ms not in (80, 100, 120, 200):
            raise ValueError("实时音频包只支持 80、100、120 或 200 毫秒。")
        query = urlencode({"voice_id": voice_id, "strength": f"{strength:.2f}", "low_latency": "true",
                           "speech_context": "true", "mode": mode, "packet_ms": packet_ms,
                           **pitch_query(pitch_preset)})
        return f"ws://{self.host}:{self.port}/api/live?{query}"


def listening(host: str, port: int, timeout: float = 0.1) -> bool:
    """端口上有没有程序在监听。Windows 拒绝连接要等约 2 秒，这里只等 timeout 秒。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def spawn_backend(script: Path, env: dict | None = None) -> subprocess.Popen:
    """在后台运行启动脚本，输出接到管道里，供启动器读取进度。"""
    # 写成 .\name：系统设置了 NoDefaultCurrentDirectoryInExePath 时，cmd 不会在当前目录里找脚本。
    return subprocess.Popen(
        ["cmd.exe", "/d", "/c", ".\\" + script.name], cwd=str(script.parent),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1", "VOICE_WARM_FIRST": WARM_FIRST, **(env or {})},
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def start_early(client: BackendClient, script: Path = START_SCRIPT) -> subprocess.Popen | None:
    """界面还在初始化时就先把后端拉起来。端口已有程序监听或脚本不存在时什么也不做。"""
    if listening(client.host, client.port) or not Path(script).is_file():
        return None
    try:
        return spawn_backend(Path(script))
    except OSError:
        return None


class BackendLauncher(threading.Thread):
    """检查本地后端：已运行就复用；否则启动 start-backend.cmd 并等到 /api/health 就绪。

    回调在本线程里调用：on_progress(text)、on_ready(health, started_here)、on_failed(reason)，
    以及可选的 on_stage(stage, fraction, text)：stage 是脚本阶段名或 "models"，fraction 是模型加载进度。
    process 是界面初始化前已经拉起的后端（见 start_early），有的话直接等它就绪。
    """

    def __init__(self, client: BackendClient, on_progress, on_ready, on_failed, *,
                 autostart: bool = True, script: Path = START_SCRIPT, wait_seconds: float = 900.0,
                 on_stage=None, process: subprocess.Popen | None = None):
        super().__init__(daemon=True, name="backend-launcher")
        self.client = client
        self.autostart = autostart
        self.script = Path(script)
        self.wait_seconds = wait_seconds
        self.process: subprocess.Popen | None = process
        self._on_progress, self._on_ready, self._on_failed = on_progress, on_ready, on_failed
        self._on_stage = on_stage or (lambda stage, fraction, text: None)
        self._cancel = threading.Event()
        self._log: deque[str] = deque(maxlen=12)

    def _stage_from_health(self, health: dict) -> None:
        try:
            fraction = min(1.0, max(0.0, float(health.get("progress", 0))))
        except (TypeError, ValueError):
            fraction = 0.0
        self._on_stage("models", fraction, str(health.get("stage", "加载模型")))

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        if self.process is not None:  # 界面初始化前已经拉起
            self._on_progress(f"正在启动 {self.script.name}…")
            self._on_stage("server", None, "启动后端")
            threading.Thread(target=self._read_output, daemon=True, name="backend-output").start()
            self._wait(owned=True)
            return
        self._on_progress("正在检查本地后端…")
        try:
            health = self.client.health(connect_timeout=.1)
            if health.get("status") == "ok":
                self._on_ready(health, False)
            else:  # 别处启动的后端还在加载模型
                self._stage_from_health(health)
                self._wait(owned=False)
            return
        except BackendError as exc:
            if exc.status is not None:
                self._on_failed(str(exc))
                return
            first_error = str(exc)
        if not self.autostart:
            self._on_failed(f"{first_error}\n已关闭自动启动，请先运行 start-backend.cmd。")
            return
        if not self.script.is_file():
            self._on_failed(f"{first_error}\n找不到启动脚本：{self.script}")
            return
        self._on_progress(f"后端未运行，正在启动 {self.script.name}…")
        self._on_stage("server", None, "启动后端")
        try:
            self.process = spawn_backend(self.script)
        except OSError as exc:
            self._on_failed(f"无法运行启动脚本 {self.script}：{exc}")
            return
        threading.Thread(target=self._read_output, daemon=True, name="backend-output").start()
        self._wait(owned=True)

    def _wait(self, owned: bool) -> None:
        """等 /api/health 报 ok。加载中把进度交给 on_stage；自己启动的进程退出了就报失败。"""
        deadline = time.monotonic() + self.wait_seconds
        while time.monotonic() < deadline:
            if self._cancel.wait(0.1):
                return
            try:
                health = self.client.health(timeout=3.0, connect_timeout=.1)
            except BackendError as exc:
                if exc.status is not None:  # 后端在回答，但说模型加载失败（HTTP 503）
                    self._on_failed(f"{exc}\n{self._tail()}")
                    return
                code = self.process.poll() if owned and self.process is not None else None
                if code is not None:
                    time.sleep(0.2)  # 让输出线程读完最后几行
                    self._on_failed(f"{self.script.name} 已退出（退出码 {code}），后端没有就绪。\n{self._tail()}")
                    return
                continue
            if health.get("status") == "ok":
                self._on_ready(health, owned)
                return
            self._stage_from_health(health)
        self._on_failed(f"等待后端就绪超过 {self.wait_seconds:g} 秒。\n{self._tail()}")

    def _read_output(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for raw in self.process.stdout:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            match = STARTUP_LINE.search(line)
            if match:
                token, text = match.groups()
                try:
                    self._on_stage("models", min(1.0, max(0.0, float(token))), text)
                except ValueError:
                    self._on_stage(token, None, text)
                continue
            self._log.append(line)
            self._on_progress(f"后端启动中：{line[:120]}")

    def _tail(self) -> str:
        lines = list(self._log)[-6:]
        return "脚本最后的输出：\n" + "\n".join(lines) if lines else "脚本没有输出。"


def stop_process_tree(process: subprocess.Popen | None) -> None:
    """结束由本程序启动的后端（cmd 及其子进程 uvicorn）。"""
    if process is None or process.poll() is not None:
        return
    try:
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=10)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass
