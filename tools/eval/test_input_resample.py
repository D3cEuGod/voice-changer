"""Packet resampling must match one continuous FIR resample, including fractional hops."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.character import fresh_input, INPUT_RESAMPLE_GUARD, StreamingDenoiser
from backend.engine import RATE


class ResampleTests(unittest.TestCase):
    def test_denoiser_consumes_guard_inside_existing_lookahead(self):
        class IdentityGraph:
            def get_inputs(self):
                return [SimpleNamespace(name='mix')]
            def run(self, names, inputs):
                return [inputs['mix']]
        class AlwaysSpeech:
            def __init__(self, count):self.count=count
            def process(self, fresh):return np.ones(self.count, np.float32)
        rng = np.random.default_rng(66)
        x = rng.uniform(-.2, .2, 2*RATE).astype(np.float32)
        expected = resample_poly(np.pad(x, (RATE, 0)), 320, 441)
        for frame in (2646, 8192):
            with self.subTest(frame=frame), \
                 patch('backend.character.graph', return_value=IdentityGraph()), \
                 patch('backend.character.SpeechDetector', AlwaysSpeech), \
                 patch('backend.character.INPUT_HIGHPASS', np.array([[1., 0, 0, 1, 0, 0]])):
                denoiser = StreamingDenoiser()
                history, total = np.zeros(RATE, np.float32), 0
                for start in range(0, len(x), frame):
                    block=x[start:start+frame]
                    history[:-len(block)]=history[len(block):]
                    history[-len(block):]=block
                    clean,_=denoiser.process(history,len(block))
                    total+=len(block)
                    position=round(total*320/441)
                    # The unresolved right edge is lookahead, never emitted.
                    np.testing.assert_allclose(clean[:-1024], expected[position:position+16000-1024],
                                               rtol=0, atol=2e-7)

    def test_fixed_and_fractional_packets_match_continuous_reference(self):
        rng = np.random.default_rng(12)
        t = np.arange(3*RATE)/RATE
        x = (.2*np.sin(2*np.pi*737*t)+.03*rng.standard_normal(len(t))).astype(np.float32)
        full = resample_poly(np.pad(x, (RATE, 0)), 320, 441)
        expected = full[16000-INPUT_RESAMPLE_GUARD:]
        for frame in (1764, 2205, 2646, 4410, 8192):
            with self.subTest(frame=frame):
                history = np.zeros(RATE, np.float32)
                total, parts = 0, []
                for i in range(0, len(x), frame):
                    block = x[i:i+frame]
                    history[:-len(block)] = history[len(block):]
                    history[-len(block):] = block
                    parts.append(fresh_input(history, total, total+len(block)))
                    total += len(block)
                actual = np.concatenate(parts)
                self.assertEqual(len(actual), round(len(x)*320/441))
                np.testing.assert_allclose(actual, expected[:len(actual)], rtol=0, atol=2e-7)

    def test_changing_packet_sizes_preserve_phase_and_sample_count(self):
        rng = np.random.default_rng(23)
        sizes = [333, 440, 8192, 2646, 103, 4410, 11]
        x = rng.uniform(-.2, .2, sum(sizes)).astype(np.float32)
        expected = resample_poly(np.pad(x, (RATE, 0)), 320, 441)[16000-INPUT_RESAMPLE_GUARD:]
        history = np.zeros(RATE, np.float32)
        parts, total = [], 0
        for count in sizes:
            history[:-count] = history[count:]
            history[-count:] = x[total:total+count]
            parts.append(fresh_input(history, total, total+count))
            total += count
        actual = np.concatenate(parts)
        self.assertEqual(len(actual), round(len(x)*320/441))
        np.testing.assert_allclose(actual, expected[:len(actual)], rtol=0, atol=2e-7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
