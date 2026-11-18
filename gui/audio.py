"""音频设备、录音、片段播放和监听用的流式缓冲。

只在 QApplication 创建之后导入本模块：PortAudio 初始化 COM 的方式会影响 Qt 的原生文件对话框。
设备回调只做复制和入队，重采样与网络收发都在别的线程。
"""
from __future__ import annotations

import io
import queue
import threading
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import soundfile as sf
import soxr

ALLOWED_FORMATS = {"WAV", "WAVEX", "FLAC", "MP3", "OGG"}
MIN_SECONDS = 0.35
# 输出流多留 0.1 秒设备缓冲：回调是 Python 写的，界面忙的时候可能晚到二三十毫秒。
OUTPUT_LATENCY = 0.1


class AudioDeviceError(RuntimeError):
    """设备打不开、被拔出或系统里没有这类设备。"""


class AudioFileError(ValueError):
    """文件不是后端支持的音频。"""


@dataclass(frozen=True)
class Device:
    index: int
    name: str        # 系统报告的完整设备名
    rate: int        # 设备默认采样率，流按这个采样率打开
    channels: int
    is_default: bool
    hostapi: str


def discord_pair(inputs, outputs):
    """Return the playback endpoint and the corresponding Discord microphone."""
    for play, record in (("cable input", "cable output"),
                         ("steam streaming microphone", "steam streaming microphone")):
        sender = next((d for d in outputs if play in d.name.casefold()), None)
        receiver = next((d for d in inputs if record in d.name.casefold()), None)
        if sender and receiver:
            return sender, receiver
    return None


def _hostapi_rank(name: str) -> int:
    # WASAPI 给出完整设备名并反映热插拔后的真实状态；MME 会把名字截断到 31 个字符。
    for rank, key in enumerate(("WASAPI", "DirectSound", "MME")):
        if key in name:
            return rank
    return 99


def list_devices(kind: str) -> list[Device]:
    """kind 取 'input' 或 'output'。没有设备时返回空列表。"""
    try:
        apis = sd.query_hostapis()
        devices = sd.query_devices()
    except Exception as exc:  # PortAudio 未能初始化
        raise AudioDeviceError(f"无法读取音频设备列表：{exc}") from exc
    channel_key = f"max_{kind}_channels"
    default_key = f"default_{kind}_device"
    for api in sorted(apis, key=lambda a: _hostapi_rank(a["name"])):
        if _hostapi_rank(api["name"]) == 99:
            continue
        found = [
            Device(index=i, name=devices[i]["name"], rate=int(round(devices[i]["default_samplerate"])),
                   channels=int(devices[i][channel_key]), is_default=(i == api[default_key]),
                   hostapi=api["name"])
            for i in api["devices"] if devices[i][channel_key] > 0
        ]
        if found:
            return sorted(found, key=lambda d: not d.is_default)
    return []


def rescan() -> None:
    """重新枚举设备（插拔后调用）。调用前必须关闭所有流。"""
    try:
        sd._terminate()
        sd._initialize()
    except Exception as exc:
        raise AudioDeviceError(f"重新扫描音频设备失败：{exc}") from exc


def _open_stream(factory, device: Device, kind_label: str, channel_options, **kwargs):
    last_error = None
    for channels in dict.fromkeys(c for c in channel_options if 0 < c <= device.channels):
        try:
            return factory(device=device.index, samplerate=device.rate, channels=channels,
                           dtype="float32", **kwargs), channels
        except Exception as exc:
            last_error = exc
    raise AudioDeviceError(f"无法打开{kind_label}“{device.name}”：{last_error or '设备没有可用声道'}")


def _close_stream(stream) -> None:
    if stream is None:
        return
    for action in (stream.abort, stream.close):
        try:
            action()
        except Exception:
            pass


def _mono(indata: np.ndarray) -> np.ndarray:
    return indata[:, 0].copy() if indata.shape[1] == 1 else indata.mean(axis=1)


def _peak(samples: np.ndarray) -> float:
    return float(np.max(np.abs(samples))) if len(samples) else 0.0


class Recorder:
    """把麦克风录成一段单声道片段，上限 max_seconds。"""

    def __init__(self, device: Device, max_seconds: float = 60.0):
        self.device = device
        self.rate = device.rate
        self.level = 0.0
        self.full = False
        self._limit = int(max_seconds * device.rate)
        self._frames = 0
        self._chunks: list[np.ndarray] = []
        self._stream, _ = _open_stream(sd.InputStream, device, "麦克风", (1, 2, device.channels),
                                       callback=self._callback)

    def _callback(self, indata, frames, time_info, status):
        mono = _mono(indata)
        self.level = _peak(mono)
        room = self._limit - self._frames
        if room <= 0:
            self.full = True
            return
        self._chunks.append(mono[:room])
        self._frames += min(len(mono), room)

    def start(self) -> None:
        try:
            self._stream.start()
        except Exception as exc:
            _close_stream(self._stream)
            raise AudioDeviceError(f"无法开始录音“{self.device.name}”：{exc}") from exc

    @property
    def active(self) -> bool:
        return bool(self._stream.active)

    @property
    def seconds(self) -> float:
        return self._frames / self.rate

    def stop(self) -> np.ndarray:
        _close_stream(self._stream)
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks).astype(np.float32, copy=False)


class ClipPlayer:
    """在选定的输出设备上播放一段单声道片段，界面线程可读取进度。"""

    def __init__(self):
        self._stream = None
        self._data = np.zeros(0, dtype=np.float32)
        self._pos = 0
        self._rate = 1
        self._done = True

    def play(self, samples: np.ndarray, rate: int, device: Device, start_seconds: float = 0.0) -> None:
        self.stop()
        data = np.asarray(samples, dtype=np.float32)
        if rate != device.rate:
            data = soxr.resample(data, rate, device.rate).astype(np.float32, copy=False)
        self._data = np.ascontiguousarray(np.clip(data, -1.0, 1.0))
        self._rate = device.rate
        self._pos = min(len(self._data), max(0, int(start_seconds * device.rate)))
        self._done = False
        self._stream, _ = _open_stream(sd.OutputStream, device, "输出设备", (2, 1, device.channels),
                                       latency=OUTPUT_LATENCY, callback=self._callback,
                                       finished_callback=self._finished)
        try:
            self._stream.start()
        except Exception as exc:
            self.stop()
            raise AudioDeviceError(f"无法在“{device.name}”上播放：{exc}") from exc

    def _callback(self, outdata, frames, time_info, status):
        pos = self._pos
        chunk = self._data[pos:pos + frames]
        count = len(chunk)
        outdata[:count] = chunk[:, None]
        self._pos = pos + count
        if count < frames:
            outdata[count:] = 0
            raise sd.CallbackStop

    def _finished(self):
        self._done = True

    @property
    def playing(self) -> bool:
        return self._stream is not None and not self._done and bool(self._stream.active)

    @property
    def position(self) -> float:
        return self._pos / self._rate

    @property
    def duration(self) -> float:
        return len(self._data) / self._rate

    def seek(self, seconds: float) -> None:
        self._pos = min(len(self._data), max(0, int(seconds * self._rate)))

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        _close_stream(stream)
        self._done = True


class MicStream:
    """监听用的采集端。回调把单声道块放进有限长度的队列，满了就丢最旧的一块。"""

    BLOCK_SECONDS = 0.02

    def __init__(self, device: Device, max_seconds: float = 1.5):
        self.device = device
        self.name = device.name
        self.rate = device.rate
        self.level = 0.0
        self.dropped = 0
        self.blocks: queue.Queue[np.ndarray] = queue.Queue(maxsize=int(max_seconds / self.BLOCK_SECONDS))
        self._stream, _ = _open_stream(sd.InputStream, device, "麦克风", (1, 2, device.channels),
                                       blocksize=int(device.rate * self.BLOCK_SECONDS),
                                       latency=.02,
                                       callback=self._callback)

    def _callback(self, indata, frames, time_info, status):
        if status.input_overflow:
            self.dropped += 1  # 设备缓冲溢出，丢了一段输入
        mono = _mono(indata)
        self.level = _peak(mono)
        try:
            self.blocks.put_nowait(mono)
        except queue.Full:
            try:
                self.blocks.get_nowait()
            except queue.Empty:
                pass
            self.dropped += 1
            try:
                self.blocks.put_nowait(mono)
            except queue.Full:
                pass

    def start(self) -> None:
        try:
            self._stream.start()
        except Exception as exc:
            self.close()
            raise AudioDeviceError(f"无法开始采集“{self.name}”：{exc}") from exc

    @property
    def active(self) -> bool:
        return self._stream is not None and bool(self._stream.active)

    @property
    def latency(self) -> float:
        return float(self._stream.latency) if self._stream is not None else 0.0

    def close(self) -> None:
        stream, self._stream = self._stream, None
        _close_stream(stream)


class AudioFifo:
    """有限长度的环形缓冲。写满时丢最旧的样本，读写各持锁很短的时间。"""

    def __init__(self, capacity: int):
        self._buffer = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._read = 0
        self._count = 0
        self._lock = threading.Lock()
        self.dropped = 0

    def __len__(self) -> int:
        return self._count

    def write(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32)
        with self._lock:
            if len(samples) > self._capacity:
                self.dropped += len(samples) - self._capacity
                samples = samples[-self._capacity:]
            overflow = self._count + len(samples) - self._capacity
            if overflow > 0:
                self._read = (self._read + overflow) % self._capacity
                self._count -= overflow
                self.dropped += overflow
            start = (self._read + self._count) % self._capacity
            first = min(len(samples), self._capacity - start)
            self._buffer[start:start + first] = samples[:first]
            self._buffer[:len(samples) - first] = samples[first:]
            self._count += len(samples)

    def read(self, count: int) -> np.ndarray:
        with self._lock:
            count = min(count, self._count)
            first = min(count, self._capacity - self._read)
            out = np.concatenate((self._buffer[self._read:self._read + first], self._buffer[:count - first]))
            self._read = (self._read + count) % self._capacity
            self._count -= count
            return out


def live_slack(reference_engine: bool, voice: str, mode: str, discord: bool,
               output_name: str = "") -> float:
    """Absorb inference and device jitter before sending audio to a virtual microphone."""
    if discord:
        if not reference_engine and "cable input" in output_name.casefold():
            return .32
        return .50
    if reference_engine and voice in ("tomori", "taffy") and mode == "character":
        return .40
    return .24


def live_packet_ms(reference_engine: bool, output_name: str) -> int:
    """Keep the verified 200 ms path for Steam and the optional reference engine."""
    return 120 if not reference_engine and "cable input" in output_name.casefold() else 200


class SpeakerStream:
    """监听用的播放端。每次断流后先留出 slack_seconds 的余量再继续，吸收推理时间的抖动。"""

    def __init__(self, device: Device, slack_seconds: float = 0.2, max_seconds: float = 3.0,
                 receiver: Device | None = None, device_latency: float | None = None):
        self.device = device
        self.name = device.name
        self._steam = receiver is not None and "steam streaming microphone" in device.name.casefold()
        self.rate = receiver.rate if self._steam else device.rate
        self.level = 0.0
        self.underruns = 0
        self.starvations = 0
        self.device_underflows = 0
        self.slack_seconds = slack_seconds
        self.fifo = AudioFifo(int(max_seconds * self.rate))
        self._slack = int(slack_seconds * self.rate)
        self._waiting = True
        self._draining = False
        self._target_fill = self._slack
        self._fade_total = 0
        self._fade_left = 0
        self._fade_done = threading.Event()
        self._muted = False
        # Large shared-mode device buffers showed sample-position jumps through
        # VB-CABLE even without a PortAudio underflow; use small native transfers.
        latency = (device_latency if device_latency is not None else
                   .02 if receiver and "cable input" in device.name.casefold() and slack_seconds < .5 else
                   max(OUTPUT_LATENCY, .15) if receiver else OUTPUT_LATENCY)
        if self._steam:
            # Steam passes raw PCM between endpoints. Match the capture format
            # instead of going through the stereo Windows playback mixer.
            try:
                self._stream = sd.OutputStream(
                    device=device.index, samplerate=self.rate, dtype="int32", channels=1,
                    extra_settings=sd.WasapiSettings(exclusive=True, explicit_sample_format=True),
                    latency=latency, callback=self._callback)
            except Exception as exc:
                raise AudioDeviceError(f"无法打开 Steam 通话输出（{self.rate} Hz 单声道）：{exc}") from exc
        else:
            self._stream, _ = _open_stream(sd.OutputStream, device, "输出设备", (2, 1, device.channels),
                                           latency=latency,
                                           callback=self._callback)

    def _callback(self, outdata, frames, time_info, status):
        if self._muted:
            outdata.fill(0)
            self.level = 0.0
            return
        if self._waiting and not self._draining and not self._fade_total and len(self.fifo) < self._target_fill:
            outdata.fill(0)
            self.level = 0.0
            return
        was_waiting = self._waiting
        self._waiting = False
        chunk = self.fifo.read(frames)
        count = len(chunk)
        ramp = min(count, int(self.rate * .003))
        if was_waiting and ramp:
            chunk[:ramp] *= np.linspace(0, 1, ramp, dtype=np.float32)
        if count < frames and ramp:
            chunk[-ramp:] *= np.linspace(1, 0, ramp, dtype=np.float32)
        if self._fade_total:
            gains = np.maximum(self._fade_left-1-np.arange(count), 0)/self._fade_total
            chunk *= gains.astype(np.float32)
            self._fade_left = max(0, self._fade_left-frames)
            if not self._fade_left:
                self._muted = True
                self.fifo.read(len(self.fifo))
                self._fade_done.set()
        if self._steam:
            outdata[:count, 0] = np.rint(np.clip(chunk, -1, 1).astype("float64") * 2147483647)
        else:
            outdata[:count] = chunk[:, None]
        self.level = _peak(chunk)
        if count < frames:  # 回包还没到：补静音，等下一包到了再留出余量继续
            outdata[count:] = 0
            self._waiting = True
            if not self._draining and not self._fade_total:
                self.underruns += 1
                self.starvations += 1
                self._target_fill = min(self._target_fill + self.rate // 16, self.rate // 2)
        elif status.output_underflow:  # 回调来晚了，设备已经断过一小段
            self.underruns += 1
            self.device_underflows += 1

    def write(self, samples: np.ndarray) -> None:
        if not self._muted:
            self.fifo.write(samples)

    def fade_out(self, seconds: float = .02) -> None:
        """Fade queued speech before an intentional switch, then discard its tail."""
        if not self.active:
            return
        self._fade_left = max(1, round(seconds*self.rate))
        self._fade_total = self._fade_left
        if self._fade_done.wait(seconds+self.latency+.1):
            # The callback filled a fading buffer; WASAPI has not necessarily
            # played it yet. stop() drains that native queue before abort/close.
            stream = self._stream
            if stream is not None:
                stream.stop()

    def begin_drain(self) -> None:
        self._draining = True

    @property
    def buffered_seconds(self) -> float:
        return len(self.fifo) / self.rate

    @property
    def target_buffer_seconds(self) -> float:
        return self._target_fill / self.rate

    @property
    def latency(self) -> float:
        try:
            return float(self._stream.latency)
        except Exception:
            return 0.0

    def start(self) -> None:
        try:
            self._stream.start()
        except Exception as exc:
            self.close()
            raise AudioDeviceError(f"无法开始播放到“{self.name}”：{exc}") from exc

    @property
    def active(self) -> bool:
        return self._stream is not None and bool(self._stream.active)

    def close(self) -> None:
        stream, self._stream = self._stream, None
        _close_stream(stream)


def probe_file(data: bytes, max_seconds: float) -> tuple[np.ndarray, int]:
    """按后端同样的限制检查并解码文件，返回 (单声道样本, 采样率)，供本地试听。"""
    try:
        with sf.SoundFile(io.BytesIO(data)) as audio:
            if audio.format not in ALLOWED_FORMATS:
                raise AudioFileError("请选择 WAV、FLAC、MP3 或 OGG 音频。")
            if audio.channels not in (1, 2) or not 8000 <= audio.samplerate <= 96000:
                raise AudioFileError("支持单声道或双声道，采样率范围为 8–96 kHz。")
            seconds = audio.frames / audio.samplerate
            if not MIN_SECONDS <= seconds <= max_seconds:
                raise AudioFileError(f"音频长度须在 {MIN_SECONDS} 至 {max_seconds:g} 秒之间（这段是 {seconds:.2f} 秒）。")
            samples = audio.read(dtype="float32", always_2d=True)
            rate = audio.samplerate
    except sf.LibsndfileError as exc:
        raise AudioFileError("无法解码音频，请重新录制或导出为 WAV。") from exc
    return np.clip(samples, -1.0, 1.0).mean(axis=1).astype(np.float32), rate


def encode_wav(samples: np.ndarray, rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(samples, -1.0, 1.0), rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class TeeSpeaker:
    """Send converted audio to the call (primary) and to headphones (ear return).

    The primary stream drives timing, buffering and statistics exactly as
    before. The ear-return stream gets its own resampler and is best effort:
    if the headphones fail, the call keeps going and only the ear return stops.
    """

    def __init__(self, primary: SpeakerStream, monitor: SpeakerStream):
        self.primary = primary
        self.monitor = monitor
        self.device = primary.device
        self.name = primary.name
        self.rate = primary.rate
        self._resampler = (soxr.ResampleStream(primary.rate, monitor.rate, 1, dtype="float32")
                           if monitor.rate != primary.rate else None)

    def __getattr__(self, name):
        # level, underruns, buffered_seconds, target_buffer_seconds, latency, slack_seconds, fifo ...
        return getattr(self.primary, name)

    def write(self, samples: np.ndarray) -> None:
        self.primary.write(samples)
        if self.monitor is not None and self.monitor.active:
            out = self._resampler.resample_chunk(samples) if self._resampler is not None else samples
            self.monitor.write(out)

    @property
    def active(self) -> bool:
        return self.primary.active

    @property
    def monitor_level(self) -> float:
        return self.monitor.level if self.monitor is not None else 0.0

    def start(self) -> None:
        self.primary.start()
        try:
            self.monitor.start()
        except AudioDeviceError:
            self.monitor.close()

    def fade_out(self, seconds: float = .02) -> None:
        self.primary.fade_out(seconds)
        if self.monitor.active:
            self.monitor.fade_out(seconds)

    def begin_drain(self) -> None:
        self.primary.begin_drain()
        self.monitor.begin_drain()

    def close(self) -> None:
        self.primary.close()
        self.monitor.close()


# 麦克风电平建议。峰值是每 20 ms 一块的最大绝对值。
QUIET_PEAK = .01     # 几秒内峰值都低于它：多半是麦克风选错、被静音或系统音量太低
LOUD_PEAK = .98      # 经常到这里：已经削波，变声会破音
ADVICE_SECONDS = 6.0


class MicAdvisor:
    """看最近几秒的麦克风峰值，给出“太小 / 爆音 / 正常”的提示。界面线程按固定间隔调用 feed()。"""

    def __init__(self, interval: float = .04, window: float = ADVICE_SECONDS):
        self._size = max(1, round(window / interval))
        self._peaks: list[float] = []

    def reset(self) -> None:
        self._peaks.clear()

    def feed(self, peak: float) -> tuple[str, str]:
        self._peaks.append(float(peak))
        del self._peaks[:-self._size]
        return self.advice()

    def advice(self) -> tuple[str, str]:
        """返回 (状态, 提示)。状态是 "" 、"quiet"、"loud" 或 "ok"；数据不够时不下结论。"""
        if len(self._peaks) < self._size:
            return "", ""
        loudest = max(self._peaks)
        clipped = sum(p >= LOUD_PEAK for p in self._peaks)
        if clipped >= 3:
            return "loud", "麦克风声音太大，已经爆音。请把系统里的麦克风音量调低一些，或离麦克风远一点。"
        if loudest < QUIET_PEAK:
            return "quiet", ("没有检测到说话声。如果你正在说话，请确认选对了麦克风、没有被静音，"
                             "或者在系统声音设置里调大麦克风音量。")
        return "ok", ""
