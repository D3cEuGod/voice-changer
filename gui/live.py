"""麦克风监听的工作线程：连续重采样、按 README 的 WebSocket 协议收发、把回包送去播放。"""
from __future__ import annotations

import json
import queue
import threading
import time

import numpy as np
import soxr
from websockets.exceptions import ConnectionClosed, WebSocketException
from websockets.sync.client import connect

CLOSE_REASONS = {
    1013: "引擎正忙（1013）：后端正在处理另一段转换或另一个监听，请稍后再开。",
    1003: "后端拒绝了音频帧（1003，帧格式错误）。",
    1001: "麦克风空闲超过 30 秒，后端已断开监听（1001）。",
    1008: "后端拒绝了这个连接来源（1008）。",
}
STALL_SECONDS = 3.0
REPLY_SECONDS = 20.0
BUSY_RETRY_SECONDS = 2.5
REFRESH_QUIET_SECONDS = .6


class LiveError(Exception):
    pass


class _Aborted(Exception):
    pass


def describe_close(exc: ConnectionClosed) -> str:
    code = exc.rcvd.code if exc.rcvd is not None else None
    if code in CLOSE_REASONS:
        return CLOSE_REASONS[code]
    if code is None:
        return "与后端的监听连接意外断开（后端可能已退出）。"
    reason = exc.rcvd.reason if exc.rcvd is not None else ""
    return f"后端关闭了监听连接（{code}{' ' + reason if reason else ''}）。"


def to_pcm16(samples: np.ndarray) -> bytes:
    return np.rint(np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


class LiveMonitor(threading.Thread):
    """一次监听会话。

    mic 提供 rate / name / blocks(有限队列) / level / dropped / active / close()；
    speaker 提供 rate / name / write() / active / close()。两者都由界面线程先打开并启动。
    回调在本线程里调用：on_connected(hello)、on_stats(dict)、on_stopped(kind, message)，
    kind 为 "user"（正常停止）、"refresh"（到期重开）或 "error"。
    无论怎样结束，都会关闭麦克风和输出流。
    """

    def __init__(self, url: str, mic, speaker, on_connected, on_stats, on_stopped,
                 *, restart_after_packets: int = 0):
        super().__init__(daemon=True, name="live-monitor")
        self.url = url
        self.mic = mic
        self.speaker = speaker
        self._on_connected, self._on_stats, self._on_stopped = on_connected, on_stats, on_stopped
        self._stop_event = threading.Event()
        self._abort = False
        self.hello: dict = {}
        self.label = ""
        self.frames_sent = 0
        self.samples_received = 0
        self.packet_samples: list[int] = []
        self._times: list[float] = []
        self._quiet_samples = 0
        if restart_after_packets < 0:
            raise ValueError("自动刷新包数不能为负数。")
        self.restart_after_packets = restart_after_packets

    def request_stop(self, abort: bool = False) -> None:
        """abort=False 按协议补零并发送 finish；abort=True 直接断开。"""
        self._abort = self._abort or abort
        self._stop_event.set()

    # ---- 线程主体 -------------------------------------------------------
    def run(self) -> None:
        kind, message = "user", ""
        ws = None
        try:
            # 刚断开的上一个会话可能还占着引擎一小会儿，1013 时短暂重试。
            deadline = time.monotonic() + BUSY_RETRY_SECONDS
            while True:
                ws = self._connect()
                try:
                    hello = self._handshake(ws)
                    break
                except ConnectionClosed as exc:
                    code = exc.rcvd.code if exc.rcvd is not None else None
                    if code != 1013 or time.monotonic() > deadline or self._stop_event.wait(0.25):
                        raise
            self.hello = hello
            self._on_connected(hello)
            refresh = self._pump(ws, hello)
            if not self._abort:
                self._finish(ws)
            if refresh and not self._stop_event.is_set():
                kind = "refresh"
        except _Aborted:
            pass
        except LiveError as exc:
            kind, message = "error", str(exc)
        except ConnectionClosed as exc:
            if not self._abort:
                kind, message = "error", describe_close(exc)
        except Exception as exc:  # 不让意外错误留下占用中的设备
            kind, message = "error", f"监听出错：{type(exc).__name__}: {exc}"
        finally:
            if self._abort and hasattr(self.speaker, 'fade_out'):
                try:
                    self.speaker.fade_out()
                except Exception:
                    pass
            for closer in (getattr(ws, "close", None), self.mic.close, self.speaker.close):
                if closer is not None:
                    try:
                        closer()
                    except Exception:
                        pass
            try:
                self._on_stopped(kind, message)
            finally:
                # GUI callbacks capture this monitor. Break the cycle so closed
                # device buffers are freed after every periodic refresh.
                self._on_connected = self._on_stats = self._on_stopped = None

    def _connect(self):
        try:
            # proxy=None：不让系统代理接管回环地址；PCM 不值得压缩。
            return connect(self.url, proxy=None, compression=None, open_timeout=5,
                           close_timeout=2, max_size=1 << 20)
        except ConnectionRefusedError as exc:
            raise LiveError("无法连接监听通道（连接被拒绝），后端没有在运行。") from exc
        except (OSError, TimeoutError, WebSocketException) as exc:
            raise LiveError(f"无法连接监听通道：{exc}") from exc

    def _handshake(self, ws) -> dict:
        try:
            first = ws.recv(timeout=10)
        except TimeoutError as exc:
            raise LiveError("后端 10 秒内没有发来监听参数。") from exc
        try:
            hello = json.loads(first) if isinstance(first, str) else None
            ok = (hello["format"] == "s16le" and int(hello["sample_rate"]) > 0
                  and int(hello["frame_samples"]) > 0)
        except Exception:
            ok = False
        if not ok:
            raise LiveError(f"后端的监听参数不符合协议：{first!r:.120}")
        return hello

    def _recv_audio(self, ws) -> np.ndarray:
        deadline = time.monotonic() + REPLY_SECONDS
        while True:
            if self._abort:
                raise _Aborted
            try:
                reply = ws.recv(timeout=0.25)
                break
            except TimeoutError:
                if time.monotonic() > deadline:
                    raise LiveError(f"后端超过 {REPLY_SECONDS:g} 秒没有返回音频。") from None
        if not isinstance(reply, (bytes, bytearray)) or len(reply) % 2:
            raise LiveError("后端返回了不是 PCM16 的数据。")
        self.packet_samples.append(len(reply) // 2)
        self.samples_received += len(reply) // 2
        return np.frombuffer(reply, dtype="<i2").astype(np.float32) / 32768.0

    def _play(self, samples: np.ndarray) -> None:
        if self._out_resampler is not None:
            samples = self._out_resampler.resample_chunk(samples)
        self.speaker.write(samples)

    def _exchange(self, ws, payload: bytes) -> None:
        started = time.perf_counter()
        ws.send(payload)
        self.frames_sent += 1
        received = self._recv_audio(ws)
        source = np.frombuffer(payload, dtype='<i2').astype(np.float32) / 32768
        source_rms = float(np.sqrt(np.mean(source**2))) if source.size else 0
        output_rms = float(np.sqrt(np.mean(received**2))) if received.size else 0
        # A packet limit requests a refresh; finish the utterance first. Check
        # both sides so the model's lookahead is not mistaken for a pause.
        if source_rms < .003 and output_rms < .001:
            self._quiet_samples += len(source)
        else:
            self._quiet_samples = 0
        self._play(received)
        self._times.append(time.perf_counter() - started)
        del self._times[:-20]
        self._on_stats({
            "frames": self.frames_sent,
            "last_ms": self._times[-1] * 1000,
            "avg_ms": sum(self._times) / len(self._times) * 1000,
            "mic_dropped": self.mic.dropped,
            "underruns": getattr(self.speaker, "underruns", 0),
            "buffered_ms": getattr(self.speaker, "buffered_seconds", 0.0) * 1000,
            "resampler_delay_ms": 1000 * (
                (self._in_resampler.delay() / self.hello["sample_rate"] if self._in_resampler is not None else 0)
                + (self._out_resampler.delay() / self.speaker.rate if self._out_resampler is not None else 0)),
        })

    def _pump(self, ws, hello: dict) -> bool:
        rate, frame = int(hello["sample_rate"]), int(hello["frame_samples"])
        self._frame_bytes = frame * 2
        self._in_resampler = (soxr.ResampleStream(self.mic.rate, rate, 1, dtype="float32")
                              if self.mic.rate != rate else None)
        self._out_resampler = (soxr.ResampleStream(rate, self.speaker.rate, 1, dtype="float32")
                               if self.speaker.rate != rate else None)
        self._pending = bytearray()
        last_block = time.monotonic()
        while not self._stop_event.is_set():
            if not self.mic.active:
                raise LiveError(f"麦克风“{self.mic.name}”已断开或不可用，监听已停止。")
            if not self.speaker.active:
                raise LiveError(f"输出设备“{self.speaker.name}”已断开或不可用，监听已停止。")
            try:
                block = self.mic.blocks.get(timeout=0.1)
            except queue.Empty:
                if time.monotonic() - last_block > STALL_SECONDS:
                    raise LiveError(
                        f"麦克风“{self.mic.name}”超过 {STALL_SECONDS:g} 秒没有送出音频，设备可能已被拔出。")
                continue
            last_block = time.monotonic()
            if self._in_resampler is not None:
                block = self._in_resampler.resample_chunk(block)
            self._pending += to_pcm16(block)
            while len(self._pending) >= self._frame_bytes and not self._stop_event.is_set():
                payload = bytes(self._pending[:self._frame_bytes])
                del self._pending[:self._frame_bytes]
                self._exchange(ws, payload)
                if (self.restart_after_packets and self.frames_sent >= self.restart_after_packets
                        and self._quiet_samples >= round(REFRESH_QUIET_SECONDS*rate)):
                    return True
        return False

    def _finish(self, ws) -> None:
        """正常停止：不足一帧的尾巴补零发送，再发 finish，读完末尾 PCM 和关闭帧。"""
        if self._in_resampler is not None:
            self._pending += to_pcm16(self._in_resampler.resample_chunk(np.zeros(0, dtype=np.float32), last=True))
        while self._pending:
            payload = bytes(self._pending[:self._frame_bytes]).ljust(self._frame_bytes, b"\0")
            del self._pending[:self._frame_bytes]
            self._exchange(ws, payload)
        ws.send("finish")
        self._play(self._recv_audio(ws))
        if self._out_resampler is not None:
            self.speaker.write(self._out_resampler.resample_chunk(np.empty(0, np.float32), last=True))
        try:
            ws.recv(timeout=2)
        except (ConnectionClosed, TimeoutError):
            pass
        # Normal stop keeps the last spoken syllable in the device FIFO.
        if hasattr(self.speaker, "begin_drain"):
            self.speaker.begin_drain()
        deadline = time.monotonic() + getattr(self.speaker, "buffered_seconds", 0) + 0.5
        while (not self._abort and getattr(self.speaker, "buffered_seconds", 0) > 0
               and self.speaker.active and time.monotonic() < deadline):
            time.sleep(.01)
        if not self._abort:
            time.sleep(min(float(getattr(self.speaker, "latency", 0)), .2))
