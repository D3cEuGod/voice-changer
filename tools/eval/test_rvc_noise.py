"""Verify the native RVC noise scale and its overlap across packet boundaries."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.rvc_noise import RollingLatent, NOISE_SCALE
from backend.character import CharacterModel


class NoiseTests(unittest.TestCase):
    def test_native_distribution_and_shape(self):
        noise = RollingLatent(seed=53).values
        self.assertEqual(noise.shape, (1, 192, 100))
        self.assertEqual(noise.dtype, np.float32)
        self.assertLess(abs(float(noise.mean())), .015)
        self.assertAlmostEqual(float(noise.std()), NOISE_SCALE, delta=.015)
        self.assertTrue(np.isfinite(noise).all())

    def test_adjacent_windows_keep_the_same_overlap(self):
        noise = RollingLatent(seed=71)
        first = noise.advance(2646).copy()
        second = noise.advance(2646)
        np.testing.assert_array_equal(first[:, :, 12:], second[:, :, :-12])
        self.assertFalse(np.array_equal(first[:, :, -12:], second[:, :, -12:]))

    def test_noise_is_independent_of_packet_partitioning(self):
        total = 5 * 22050 + 101
        expected = RollingLatent(seed=12).advance(total)
        for packet in (1764, 2205, 2646, 4410, 8192):
            noise = RollingLatent(seed=12)
            for start in range(0, total, packet):
                actual = noise.advance(min(packet, total-start))
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(noise.samples, total)

    def test_reset_sessions_have_private_noise_and_silence_keeps_clock(self):
        first, second = RollingLatent(seed=1), RollingLatent(seed=2)
        self.assertFalse(np.array_equal(first.values, second.values))
        first.advance(20*2646)  # e.g. packets whose speech gate stayed closed
        self.assertEqual(first.end_frame, 240)
        np.testing.assert_array_equal(first.advance(2646), RollingLatent(seed=1).advance(21*2646))

    def test_gated_packets_advance_model_clock_without_running_vocoder(self):
        denoiser = SimpleNamespace(process=lambda *a: (np.zeros(16000, np.float32),
                                                      np.zeros(16000, np.float32)))
        with patch('backend.character.StreamingDenoiser', return_value=denoiser), \
             patch('backend.character.local_rvc.profile', return_value=SimpleNamespace()), \
             patch('backend.character.local_rvc.graph', side_effect=AssertionError('Silence ran vocoder')):
            model = CharacterModel('anon', 1)
            for _ in range(5):
                result = model.render(np.zeros(22050, np.float32), 2646)
                self.assertFalse(result.any())
            self.assertEqual(model.latent.samples, 5*2646)
            self.assertEqual(model.latent.end_frame, 60)


if __name__ == '__main__':
    unittest.main(verbosity=2)
