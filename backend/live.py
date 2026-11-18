import numpy as np
from scipy.signal import correlate, firwin

from .engine import RATE
from .character import CharacterModel

FRAME = 8192
SHORT_FRAME = 4410
REALTIME_FRAME = 2646  # 120 ms; preserves the full 104 ms speech lookahead.
FAST_FRAME = 1764  # 80 ms; the first two packets share the same speech lookahead.
BALANCED_FRAME = 2205  # 100 ms; more inference headroom than 80 ms.
OVERLAP = 1024
SPEECH_LOOKAHEAD = 2304
JOIN = 320
SEARCH = 110
FADE = np.linspace(0, 1, JOIN, dtype="float32")
# Keep the level through each packet join (see blend). False restores the plain linear fade.
LEVEL_JOIN = True
SPLIT_PAD = 31
SPLIT = firwin(2 * SPLIT_PAD + 1, 2500, fs=RATE).astype("float32")


def pcm(samples):
    return np.rint(np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


def blend(old, new, fade=FADE):
    """Crossfade two renderings of the same overlap without a level dip in the middle.

    old and new carry SPLIT_PAD samples of context on both sides. A linear fade only keeps
    the level when both signals are identical; the vocoder's noise-like components differ
    between renderings and lose up to half their power halfway through. Each band is
    therefore scaled by the gain that restores its level for the correlation measured
    between the two renderings: 1 where they agree, up to sqrt(2) where they are unrelated.
    """
    count=len(fade)
    core = slice(SPLIT_PAD, SPLIT_PAD + count)
    out = np.zeros(count, np.float32)
    old_low, new_low = np.convolve(old, SPLIT, mode="valid"), np.convolve(new, SPLIT, mode="valid")
    for a, b in ((old_low, new_low), (old[core] - old_low, new[core] - new_low)):
        power = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
        agreement = float(np.clip(np.dot(a, b) / power, 0, 1)) if power > 1e-9 else 1.0
        gain = 1 / np.sqrt((1 - fade)**2 + fade**2 + 2 * fade * (1 - fade) * agreement)
        out += (a * (1 - fade) + b * fade) * gain
    return out


class LiveVoice:
    """Keep acoustic context and a short lookahead at packet edges."""
    def __init__(self, voice_id, strength, lookahead=OVERLAP, mode="character", *, pitch_shift=None, pitch_offset=0):
        self.model = CharacterModel(voice_id, strength, mode, pitch_shift=pitch_shift, pitch_offset=pitch_offset)
        profile=getattr(self.model,'local',None)
        if strength and profile is not None:
            lookahead=max(lookahead,round(profile.stream_lookahead_ms*RATE/1000))
        crossfade=round(profile.stream_crossfade_ms*RATE/1000) if strength and profile is not None else 0
        if crossfade:
            lookahead=max(lookahead,crossfade+SPLIT_PAD)
        self.join=crossfade or JOIN
        self.fade=np.linspace(0,1,self.join,dtype=np.float32) if crossfade else FADE
        self.lookahead = lookahead
        self.history = np.zeros(self.model.context_samples, np.float32)
        self.tail = np.empty(0, np.float32)
        self.sent = np.zeros(SPLIT_PAD, np.float32)  # the samples emitted just before the tail
        self.received = 0
        self.emitted = 0
        self.finished = False

    def process(self, frame):
        if self.finished:
            raise RuntimeError("Live session has already finished")
        count = len(frame)
        self.history[:-count] = self.history[count:]
        self.history[-count:] = frame
        self.received += count
        rendered = self.model.render(self.history, count)
        output_count = max(0, self.received-self.lookahead)-self.emitted
        if not output_count:
            return np.empty(0, np.float32)
        start = len(self.history)-self.lookahead-output_count
        end = len(self.history)-self.lookahead
        shift = 0
        if self.tail.size and self.model.strength > 0:
            # SOLA: align the overlapping voiced cycles before mixing them.
            probe = rendered[start-SEARCH:start+self.join+SEARCH]
            old = self.tail[:self.join]
            energy = np.convolve(probe**2, np.ones(self.join, np.float32), mode="valid")
            scores = correlate(probe, old, mode="valid") / np.sqrt(np.maximum(energy, 1e-9))
            if np.mean(old**2) > 1e-7 and scores.max() > 0.3 * np.linalg.norm(old):
                shift = int(scores.argmax()) - SEARCH
        result = rendered[start+shift:end+shift].copy()
        if self.tail.size and LEVEL_JOIN and self.model.strength > 0:
            first = start + shift
            result[:self.join] = blend(np.concatenate((self.sent, self.tail[:self.join+SPLIT_PAD])),
                                      rendered[first-SPLIT_PAD:first+self.join+SPLIT_PAD],self.fade)
        elif self.tail.size:
            result[:self.join] = self.tail[:self.join]*(1-self.fade) + result[:self.join]*self.fade
        if self.model.strength > 0:
            np.clip(result, -.99, .99, out=result)
        self.sent = result[-SPLIT_PAD:].copy()
        self.emitted += len(result)
        # Keep nominal duration even when the best join lies a few samples later.
        self.tail = rendered[end+shift:].copy()
        if len(self.tail) < self.lookahead:
            self.tail = np.pad(self.tail, (0, self.lookahead-len(self.tail)), mode="edge")
        return result

    def feed(self, data):
        frame = np.frombuffer(data, dtype="<i2").astype("float32") / 32768
        return pcm(self.process(frame))

    def flush(self):
        if self.finished or not self.received:
            return np.empty(0, np.float32)
        remaining = self.received-self.emitted
        # Supply real lookahead to the denoiser and vocoder for the last syllable.
        last = self.process(np.zeros(SHORT_FRAME, np.float32))[:remaining]
        self.tail = np.empty(0, np.float32)
        self.finished = True
        return last

    def finish(self):
        return pcm(self.flush())
