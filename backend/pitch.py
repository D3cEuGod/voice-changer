"""RMVPE inference with the upstream 16 kHz HTK mel front end."""
from functools import lru_cache

import numpy as np
from scipy.fft import rfft
from scipy.signal.windows import hann

from .engine import load_graph


def transpose_f0(f0, strength, register, target_hz, shift=None, offset=0):
    """Fixed semitones override automatic register matching; offset adds to either.

    Zero (unvoiced) bins stay zero. All changes are applied before the RVC
    vocoder, without changing the speech duration or transposing the waveform.
    """
    changed = False
    if shift is not None:
        f0 *= 2 ** (shift * strength / 12)
        changed = True
    elif register is not None:
        ratio = np.clip(target_hz / np.exp(register), .7, 3.2)
        f0 *= ratio ** strength
        changed = True
    if offset:
        f0 *= 2 ** (offset * strength / 12)
        changed = True
    if changed:
        np.minimum(f0, 650, out=f0)
    return f0

WINDOW = hann(1024, sym=False).astype(np.float32)
CENTS = np.pad(20 * np.arange(360) + 1997.3794084376191, (4, 4))


@lru_cache(maxsize=1)
def mel_basis():
    edges = 700 * np.expm1(np.linspace(np.log1p(30 / 700), np.log1p(8000 / 700), 130))
    frequencies = np.fft.rfftfreq(1024, 1 / 16000)
    lower = (frequencies[None] - edges[:-2, None]) / np.diff(edges)[:-1, None]
    upper = (edges[2:, None] - frequencies[None]) / np.diff(edges)[1:, None]
    triangles = np.maximum(0, np.minimum(lower, upper)).astype(np.float32)
    return triangles * (2 / (edges[2:] - edges[:-2]))[:, None].astype(np.float32)


def mel_spectrogram(audio):
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(audio, 512, mode="reflect"), 1024)[::160]
    magnitude = np.abs(rfft(frames * WINDOW, axis=1))
    return np.log(np.maximum(mel_basis() @ magnitude.T, 1e-5)).astype(np.float32)


@lru_cache(maxsize=1)
def model():
    return load_graph("rmvpe.onnx")


def pitch_track(audio, threshold=.03):
    mel = mel_spectrogram(audio)
    frames = mel.shape[1]
    # RMVPE's four downsamplers require a multiple of 32 mel frames.
    mel = np.pad(mel, ((0, 0), (0, (-frames) % 32)))
    salience = model().run(None, {"input": mel[None]})[0][0, :frames]
    centers = salience.argmax(axis=1) + 4
    neighborhood = centers[:, None] + np.arange(-4, 5)
    weights = np.take_along_axis(np.pad(salience, ((0, 0), (4, 4))), neighborhood, axis=1)
    cents = (weights * CENTS[neighborhood]).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-8)
    f0 = 10 * np.exp2(cents / 1200)
    f0[salience.max(axis=1) <= threshold] = 0
    return f0[:len(audio) // 160].astype(np.float32)
