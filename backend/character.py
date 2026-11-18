"""One-second windows for the trained RVC v2 and So-VITS-SVC 3.0 speakers."""
from functools import lru_cache
from math import gcd
import os

import numpy as np
import faiss
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import butter, resample_poly, sosfilt
from scipy.signal.windows import hann

from .engine import ROOT, RATE, load_graph, refine_timbre, tone_embedding, tone_spec
from .pitch import pitch_track as rmvpe_pitch, transpose_f0
from .rvc_noise import RollingLatent
from . import reference, local_rvc

VOICE_RATES = {"tomori": 48000, "anon": 40000, "taffy": 32000}
# How far OpenVoice pulls each trained voice toward its speaking reference; 0 skips the stage.
# Legacy compatibility path only. Imported RVC speakers bypass this stage entirely.
TIMBRE_REFINE = {"tomori": 1.0, "anon": 0.0, "taffy": 1.0}
# Voiced windows averaged into the session's estimate of the trained voice's own timbre.
TONE_WINDOWS = 40
# Share of the band above 1.7 kHz taken from the denoised input, in unvoiced and in voiced frames.
CONSONANT_UNVOICED = 0.7
CONSONANT_VOICED = 0.25
# Taffy's character preset avoids mixing the caller's voiced harmonics back in.
CHARACTER_CONSONANT_UNVOICED = 0.35
CHARACTER_CONSONANT_VOICED = 0.0
# Conversational registers estimated from the project's speaker references.
REGISTER = {"tomori": 285, "anon": 310, "taffy": 315}
# Match typical speaking levels across the three independently trained vocoders.
OUTPUT_GAIN = {"tomori": .8, "anon": .8, "taffy": 3.0}
WINDOW = np.sqrt(hann(512, sym=False)).astype(np.float32)
MASK = np.ones((1, 16080), dtype=np.int64)
CONSONANT_BAND = butter(3, 1700, btype="highpass", fs=RATE, output="sos")
INPUT_HIGHPASS = butter(5, 48, btype="highpass", fs=16000, output="sos")
INPUT_RESAMPLE_GUARD = 16  # 1 ms at 16 kHz; keep the FIR's zero-padded edge out of speech.


def fresh_input(window, previous, total):
    """Resample only real interior samples on one session-wide 16 kHz grid."""
    count = round(total*320/441)-round(previous*320/441)
    left = total-len(window)
    phase = left % 441
    base = (left-phase)//441*320
    audio = resample_poly(np.pad(window, (phase, 0)), 320, 441)
    end = round(total*320/441)-base-INPUT_RESAMPLE_GUARD
    return audio[end-count:end].copy()


@lru_cache(maxsize=None)
def graph(name):
    return load_graph(name + ".onnx", gpu=name not in ("gtcrn_simple", "silero_vad_16k_op15"))


@lru_cache(maxsize=None)
def voice_index(name):
    local = local_rvc.profile(name)
    path = local.index if local is not None else ROOT / "models" / (name + ".index")
    with path.open("rb") as stream:
        return faiss.read_index(faiss.PyCallbackIOReader(stream.read))


def retrieve(units, name, f0, index_rate=.38, *, all_frames=False):
    index = voice_index(name)
    # Large IVF quantizers need more CPU time even with nprobe=1. Anon's
    # 175k-vector index lost realtime headroom at two threads on this machine.
    faiss.omp_set_num_threads(min(os.cpu_count() or 1, 4 if index.ntotal >= 100000 else 2))
    distances, ids, vectors = index.search_and_reconstruct(units[0], 8)
    weights = np.where(ids >= 0, np.reciprocal(np.maximum(distances, 1e-5))**2, 0)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    nearest = (np.nan_to_num(vectors) * weights[:, :, None]).sum(axis=1)
    mix = np.full(units.shape[1], index_rate, np.float32) if all_frames else np.where(
        f0.reshape(-1, 2).max(axis=1) > 0, index_rate, .06).astype(np.float32)
    return units + (nearest[None] - units) * mix[None, :, None]


def rvc_features(units, name, f0, profile, mode):
    original = units.repeat(2, axis=1)
    ratio = profile.index_rate if mode == 'character' else min(profile.index_rate, .5)
    if ratio == 0:
        return original
    retrieved = retrieve(units, name, f0, 1., all_frames=True).repeat(2, axis=1)
    protect = profile.protect if mode == 'character' else min(profile.protect, .15)
    mask = np.ones_like(f0) if protect >= .5 else np.where(f0 > 0, 1., protect).astype(np.float32)
    return original + (retrieved-original) * (ratio*mask)[None, :, None]


class SpeechDetector:
    def __init__(self, history_samples=16000):
        self.model = graph("silero_vad_16k_op15")
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros(64, np.float32)
        self.pending = np.empty(0, np.float32)
        self.history = np.zeros(history_samples, np.float32)
        self.speech_history = np.zeros(history_samples, np.float32)
        self.speaking = False
        self.noise_power = .001**2

    def process(self, samples):
        data = np.concatenate((self.pending, samples))
        consumed = len(data) // 512 * 512
        levels = np.empty(consumed, np.float32)
        speech = np.empty(consumed, np.float32)
        for i in range(0, consumed, 512):
            block = data[i:i+512]
            probability, self.state = self.model.run(None, {
                "input": np.concatenate((self.context, block))[None],
                "state": self.state, "sr": np.asarray(16000, dtype=np.int64)})
            levels[i:i+512] = probability.item()
            # A probability is confidence, not an audio volume control. Open on
            # speech, then keep quieter syllables until confidence really falls.
            confidence = probability.item()
            power = float(np.mean(block**2))
            if confidence < .1 and not self.speaking:
                self.noise_power = .95*self.noise_power + .05*power
            weak_onset = confidence >= .20 and power > 4*self.noise_power
            self.speaking = confidence >= .12 if self.speaking else confidence >= .35 or weak_onset
            speech[i:i+512] = float(self.speaking)
            self.context = block[-64:]
        self.pending = data[consumed:]
        self.history = np.concatenate((self.history, levels))[-len(self.history):]
        self.speech_history = np.concatenate((self.speech_history, speech))[-len(self.history):]
        scores = np.pad(self.speech_history[len(self.pending):], (0, len(self.pending)), mode="edge")
        # Keep the existing 96 ms pre-roll and retain 240 ms after speech for
        # soft endings / gaps. This uses context, without adding playback delay.
        scores = maximum_filter1d(scores, 5377, origin=1152)
        return uniform_filter1d(scores, 160)


class StreamingDenoiser:
    """Keep GTCRN's recurrent state, STFT overlap and fractional hops across packets."""
    def __init__(self, history_samples=16000):
        self.model = graph("gtcrn_simple")
        self.caches = {a.name: np.zeros(a.shape, np.float32) for a in self.model.get_inputs()[1:]}
        self.previous = np.zeros(256, np.float32)
        self.overlap = np.zeros(256, np.float32)
        self.pending = np.empty(0, np.float32)
        self.history = np.zeros(history_samples, np.float32)
        self.source_count = 0
        self.detector = SpeechDetector(history_samples)
        self.highpass_state = np.zeros((len(INPUT_HIGHPASS), 2))
        self.input_history = np.zeros(history_samples, np.float32)
        self.noise_power = 0.0

    @property
    def noise_rms(self):
        return np.sqrt(self.noise_power)

    def process(self, window, count):
        previous = self.source_count
        self.source_count += count
        fresh = fresh_input(window, previous, self.source_count)
        fresh, self.highpass_state = sosfilt(INPUT_HIGHPASS, fresh, zi=self.highpass_state)
        fresh = fresh.astype(np.float32)
        self.input_history = np.concatenate((self.input_history, fresh))[-len(self.input_history):]
        # The FIR guard is already inside the converter's speech lookahead.
        # Align both analysis tracks back to the original input timeline.
        audio = np.pad(self.input_history[INPUT_RESAMPLE_GUARD:], (0, INPUT_RESAMPLE_GUARD), mode='edge')
        activity = self.detector.process(fresh)
        activity = np.pad(activity[INPUT_RESAMPLE_GUARD:], (0, INPUT_RESAMPLE_GUARD), mode='edge')
        data = np.concatenate((self.pending, fresh))
        consumed = len(data) // 256 * 256
        clean = np.empty(consumed, np.float32)
        for i in range(0, consumed, 256):
            hop = data[i:i + 256]
            bins = np.fft.rfft(np.concatenate((self.previous, hop)) * WINDOW)
            spec = np.stack((bins.real, bins.imag), axis=-1).astype(np.float32)
            enhanced, *states = self.model.run(None, {"mix": spec[None, :, None, :], **self.caches})
            self.caches = dict(zip(self.caches, states))
            re, im = enhanced[0, :, 0].T
            block = WINDOW * np.fft.irfft(re + 1j * im, n=512)
            clean[i:i + 256] = self.overlap + block[:256]
            self.previous, self.overlap = hop, block[256:]
        self.pending = data[consumed:]
        self.history = np.concatenate((self.history, clean))[-len(self.history):]
        # The realtime converter withholds 104 ms at the right edge. The unresolved
        # 16–32 ms here is lookahead only and is never emitted as speech.
        lookahead = 256 + len(self.pending) + INPUT_RESAMPLE_GUARD
        clean = np.concatenate((self.history[lookahead:], audio[-lookahead:] * activity[-lookahead:]))
        background = clean[activity < .005]
        if background.size >= 512:
            self.noise_power = .8*self.noise_power + .2*float(np.mean(background**2))
        return clean, activity


def pitch_track(audio):
    # 只有旧版兼容声线用 Praat；导入的 RVC 声线走 RMVPE。按需导入，便携版不带这个 GPL-3.0 依赖。
    import parselmouth
    sound = parselmouth.Sound(audio, 16000)
    track = sound.to_pitch_ac(time_step=0.01, pitch_floor=65, pitch_ceiling=650,
                              voicing_threshold=0.48)
    # Praat omits the boundary analysis windows. Align by their timestamps.
    padded = np.zeros(100, np.float32)
    positions = np.rint((track.xs() - 0.005) * 100).astype(int).clip(0, 99)
    padded[positions] = track.selected_array["frequency"]
    return padded


def audio_gate(clean, activity, samples, noise_rms=0, full_rms=.00065):
    rms = np.sqrt(np.maximum(uniform_filter1d(clean**2, 480), 0))
    # GTCRN already removes background noise. The old full-gain threshold
    # (.00175 RMS) erased quiet consonants even after VAD accepted the speech.
    floor = max(.00005, 1.8*noise_rms)
    full = max(full_rms, 4*noise_rms)
    gate = np.clip((rms-floor)/(full-floor), 0, 1) * activity
    return np.interp(np.arange(samples)*16000/RATE, np.arange(len(clean)), gate)


class CharacterModel:
    def __init__(self, name, strength, mode="character", *, pitch_shift=None, pitch_offset=0):
        self.name = name
        self.strength = strength
        self.local = local_rvc.profile(name)
        self.mode = mode
        self.pitch_shift, self.pitch_offset = pitch_shift, pitch_offset
        self.fidelity = name == "taffy" and mode == "character"
        self.anon_fidelity = name == "anon" and mode == "character"
        self.reference = (reference.client() if os.environ.get("VOICE_REFERENCE_ENGINE") == "seed" and strength and mode == "character"
                          and name in reference.PREFERRED_VOICES else None)
        if self.reference is not None and not self.reference.start():
            self.reference = None
        if self.reference is not None:
            self.local = None
        self.context_samples = reference.CONTEXT_SAMPLES if self.reference is not None else RATE
        self.register = None
        self.tone = None
        self.tone_windows = 0
        self.denoiser = StreamingDenoiser(round(self.context_samples*16000/RATE)) if strength else None
        self.latent = RollingLatent() if strength else None

    def source_tone(self, wet, voiced_frames):
        """Running mean of the trained voice's own timbre vector, from windows with enough voicing."""
        if voiced_frames >= 25:
            current = tone_embedding(tone_spec(np.pad(wet, (0, (-len(wet)) % 256), mode="reflect")))
            self.tone_windows = min(self.tone_windows + 1, TONE_WINDOWS)
            self.tone = current if self.tone is None else self.tone + (current - self.tone) / self.tone_windows
        return self.tone

    def render(self, window, count):
        if self.strength == 0:
            return window.copy()
        # Advance through silent packets as well, keeping noise and speech on
        # the same timeline when the gate reopens. Overlapping windows reuse
        # their noise instead of sampling a new unrelated one every 120 ms.
        latent = self.latent.advance(count, RATE)
        clean, activity = self.denoiser.process(window, count)
        if np.max(activity) < .005 or np.max(np.abs(clean)) < 0.0001:
            return np.zeros_like(window)
        if self.reference is not None:
            rendered = self.reference.convert(clean, self.name)
            wet = np.zeros_like(window)
            # Leave headroom for the level-preserving packet crossfade.
            wet[reference.OUTPUT_START:reference.OUTPUT_START+reference.OUTPUT_SAMPLES] = .7 * rendered
            wet *= audio_gate(clean, activity, len(window), self.denoiser.noise_rms)
            dry = resample_poly(clean, 441, 320)
            return np.clip(self.strength*wet + (1-self.strength)*dry, -.99, .99)
        f0 = rmvpe_pitch(clean) if self.fidelity or self.local is not None else pitch_track(clean)
        voiced = f0[f0 > 0]
        if voiced.size >= 5:
            median = np.log(float(np.median(voiced)))
            decay = .92 ** (count / (RATE * .2))
            self.register = median if self.register is None else decay*self.register + (1-decay)*median
        shift = self.pitch_shift
        if shift is None and self.local is not None:
            shift = self.local.semitones
        target = self.local.target_hz if self.local is not None else REGISTER[self.name]
        transpose_f0(f0, self.strength, self.register, target, shift, self.pitch_offset)
        if self.name == "taffy" and self.local is None:
            units = graph("hubert_soft").run(None, {"audio": clean[None, None]})[0]
            inputs = {"units": units.repeat(2, axis=1).transpose(0, 2, 1), "f0": f0[None]}
            model = graph("taffy_svc")
        else:
            units = graph("contentvec_768l12").run(None, {
                "input_values": np.pad(clean, 40, mode="reflect")[None], "attention_mask": MASK,
            })[0]
            if self.local is not None:
                # RVC's protection happens in feature space, not by mixing the
                # caller's raw high-frequency voice into the generated waveform.
                units = rvc_features(units, self.name, f0, self.local, self.mode)
            else:
                units = retrieve(units, self.name, f0, .75 if self.anon_fidelity else .38).repeat(2, axis=1)
            mel = 1127 * np.log1p(f0 / 700)
            low, high = 1127 * np.log1p(np.array([50., 1100.]) / 700)
            pitch = np.rint((mel - low) * 254 / (high-low) + 1).clip(1, 255).astype("int64")
            inputs = {"units": units, "f0": f0[None],
                      "pitch": pitch[None], "noise": latent}
            model = local_rvc.graph(self.name) if self.local is not None else graph(self.name + "_rvc")
        generated = model.run(None, inputs)[0].ravel()
        rate = self.local.sample_rate if self.local is not None else VOICE_RATES[self.name]
        if len(generated) != rate or not np.isfinite(generated).all():
            raise RuntimeError("角色模型返回了无效音频。")
        g = gcd(rate, RATE)
        wet = resample_poly(generated, RATE // g, rate // g)
        # Prevent the vocoder from inventing hiss during pauses. Follow the
        # denoised amplitude envelope while preserving soft, unvoiced consonants.
        gate_full=(self.local.output_gate_full_rms or .00065) if self.local is not None else .00065
        gate = audio_gate(clean, activity, RATE, self.denoiser.noise_rms,gate_full)
        wet *= gate
        if self.local is not None:
            dry = resample_poly(clean, 441, 320)
            return np.clip(self.strength * .8 * wet + (1-self.strength)*dry, -.99, .99)
        refine = TIMBRE_REFINE[self.name] * self.strength
        if refine > 0 and np.max(np.abs(wet)) > 0.001:
            # Correct only what the trained voice produced, against the session's timbre
            # estimate. The input's consonants are added afterwards, so they never pass
            # through the converter.
            source = self.source_tone(wet, int(np.count_nonzero(f0)))
            wet = refine_timbre(wet, self.name, refine, source) * gate
        dry = resample_poly(clean, 441, 320)
        # Restore high-frequency consonant detail from the denoised input.
        voiced_mask = np.interp(np.arange(RATE) * 100 / RATE, np.arange(100), f0 > 0)
        unvoiced, voiced = ((CHARACTER_CONSONANT_UNVOICED, CHARACTER_CONSONANT_VOICED)
                           if self.fidelity else (CONSONANT_UNVOICED, CONSONANT_VOICED))
        if self.anon_fidelity:
            voiced = 0.0
        mix = unvoiced + (voiced - unvoiced) * voiced_mask
        level = np.clip(np.linalg.norm(wet) / max(np.linalg.norm(dry), .02), .4, 2.5)
        wet += mix * sosfilt(CONSONANT_BAND, level * dry - wet)
        return np.clip(self.strength * OUTPUT_GAIN[self.name] * wet + (1-self.strength)*dry, -.99, .99)
