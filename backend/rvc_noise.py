"""RVC's inference noise, kept on a continuous 100 Hz session timeline."""
import numpy as np

# Upstream SynthesizerTrnMs768NSFsid.infer samples N(0, 1) * 0.66666.
# The exported ONNX forward takes that already scaled tensor as its `rnd` input.
NOISE_SCALE = .66666


class RollingLatent:
    def __init__(self, *, seed=None):
        self.rng = np.random.default_rng(seed)
        self.values = self._new(100)
        self.samples = self.end_frame = 0

    def _new(self, frames):
        # Generate frame-major so changing packet sizes cannot change the noise
        # attached to a particular instant of speech.
        return (self.rng.standard_normal((frames, 192)).astype(np.float32).T[None]
                * NOISE_SCALE)

    def advance(self, count, rate=22050):
        self.samples += count
        end = (self.samples * 100 + rate // 2) // rate
        advance = end - self.end_frame
        if advance:
            fresh = self._new(advance)
            self.values = np.concatenate((self.values, fresh), axis=2)[:, :, -100:]
        self.end_frame = end
        return self.values
