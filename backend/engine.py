from functools import lru_cache
from io import BytesIO
from math import gcd
from pathlib import Path
import os

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly
from scipy.fft import rfft
from scipy.signal.windows import hann

ROOT = Path(__file__).resolve().parents[1]
RATE = 22050
TONE_WINDOW = hann(1024, sym=False).astype("float32")
TONE_CONTEXT = 8192
MAX_SECONDS = 60
from .edition import PUBLIC  # noqa: E402  公开版不内置角色
VOICES = {} if PUBLIC else {"tomori": "高松灯", "taffy": "永雏塔菲", "anon": "千早爱音"}
from .local_rvc import installed_voices
VOICES.update({key: name for key, name in installed_voices().items() if key not in VOICES})


class AudioError(ValueError):
    pass


def decode(data):
    try:
        with sf.SoundFile(BytesIO(data)) as audio:
            if audio.format not in {"WAV", "WAVEX", "FLAC", "MP3", "OGG"}:
                raise AudioError("请选择 WAV、FLAC、MP3 或 OGG 音频。")
            if audio.channels not in (1, 2) or not 8000 <= audio.samplerate <= 96000:
                raise AudioError("支持单声道或双声道，采样率范围为 8–96 kHz。")
            rate = audio.samplerate
            if not 0.35 <= audio.frames / rate <= MAX_SECONDS:
                raise AudioError("音频长度须在 0.35 至 60 秒之间。")
            samples = audio.read(frames=MAX_SECONDS * rate + 1, dtype="float32", always_2d=True)
            if not 0.35 <= len(samples) / rate <= MAX_SECONDS or not np.isfinite(samples).all():
                raise AudioError("音频数据损坏或不完整。")
    except sf.LibsndfileError as exc:
        raise AudioError("无法解码音频，请重新录制或导出为 WAV。") from exc
    mono = np.clip(samples, -1, 1).mean(axis=1)
    if rate != RATE:
        divisor = gcd(rate, RATE)
        mono = resample_poly(mono, RATE // divisor, rate // divisor)
    return np.clip(mono, -1, 1)


def load_graph(filename, *, gpu=True):
    from .graph_cache import load
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.add_session_config_entry('session.intra_op.allow_spinning', os.environ.get('VOICE_ORT_SPINNING','0'))
    options.intra_op_num_threads = 4 if gpu else 1
    options.inter_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    if gpu and "DmlExecutionProvider" in ort.get_available_providers():
        options.enable_mem_pattern = False
        providers.insert(0, "DmlExecutionProvider")
    return load(ROOT / "models" / filename, options, providers)


@lru_cache(maxsize=1)
def tone_encoder():
    return load_graph("tone_ref_encoder.onnx")


@lru_cache(maxsize=1)
def tone_converter():
    return load_graph("tone_converter.onnx")


def tone_spec(samples):
    audio = np.pad(samples, 384, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(audio, 1024)[::256]
    spec = rfft(frames * TONE_WINDOW, axis=1)
    return np.sqrt(spec.real**2 + spec.imag**2 + 1e-6)


def tone_embedding(spec):
    return tone_encoder().run(None, {"spec": spec[None]})[0].reshape(1, 256, 1)


@lru_cache(maxsize=3)
def tone_target(voice_id):
    data = np.load(ROOT / "voices" / f"{voice_id}.npy", allow_pickle=False)
    if data.shape != (1, 256, 1) or data.dtype != np.float32 or not np.isfinite(data).all():
        raise ValueError(f"Invalid timbre vector: {voice_id}")
    return data


def refine_timbre(samples, voice_id, strength, source=None):
    """Pull the timbre toward the voice's speaking reference with the OpenVoice converter.

    `source` is the timbre vector of the audio being corrected. Pass a session-level
    estimate for streaming: a vector measured from one second of audio changes from
    packet to packet, and the converter then treats neighbouring packets differently.
    """
    # The converter looks about 0.37 s to either side. Mirror that much audio past the end so
    # the newest samples are not converted against silence.
    extended = np.pad(samples, (0, TONE_CONTEXT + (-len(samples)) % 256), mode="reflect")
    spec = tone_spec(extended)
    src = tone_embedding(spec[:len(samples) // 256 + 1]) if source is None else source
    tgt = src + strength * (tone_target(voice_id) - src)
    result = tone_converter().run(None, {
        "spec": spec.T[None], "spec_lengths": np.asarray([len(spec)], dtype="int64"),
        "src_g": src, "tgt_g": tgt,
    })[0].ravel()
    if len(result) < len(samples) or not np.isfinite(result).all():
        raise RuntimeError("音色校正失败。")
    return np.clip(result[:len(samples)], -1, 1)


def convert(samples, voice_id, strength, mode="character", *, pitch_shift=None, pitch_offset=0):
    if strength == 0:
        return samples
    from .live import LiveVoice, SHORT_FRAME, SPEECH_LOOKAHEAD
    voice = LiveVoice(voice_id, strength, SPEECH_LOOKAHEAD, mode,
                      pitch_shift=pitch_shift, pitch_offset=pitch_offset)
    padded = np.pad(samples, (0, (-len(samples)) % SHORT_FRAME))
    parts = [voice.process(frame) for frame in padded.reshape(-1, SHORT_FRAME)]
    parts.append(voice.flush())
    return np.concatenate(parts)[:len(samples)]


def wav_bytes(samples):
    data = BytesIO()
    sf.write(data, samples, RATE, format="WAV", subtype="PCM_16")
    return data.getvalue()
