"""Feature-space protection and the no-index path; no neural graphs or devices."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.character import rvc_features
from backend.startup import Startup


class FeatureTests(unittest.TestCase):
    def test_zero_ratio_keeps_features_and_never_searches(self):
        units = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        profile = SimpleNamespace(index_rate=0, protect=.1)
        with patch('backend.character.retrieve', side_effect=AssertionError('Index must stay unused')):
            for mode in ('character', 'call'):
                result = rvc_features(units, 'anon', np.zeros(6), profile, mode)
                np.testing.assert_array_equal(result, units.repeat(2, axis=1))

    def test_retrieval_preserves_voiced_and_unvoiced_protection(self):
        units = np.zeros((1, 2, 3), np.float32)
        retrieved = np.ones_like(units)
        f0 = np.array([100, 0, 100, 0], np.float32)
        profile = SimpleNamespace(index_rate=.75, protect=.33)
        with patch('backend.character.retrieve', return_value=retrieved):
            result = rvc_features(units, 'tomori', f0, profile, 'character')
            np.testing.assert_allclose(result[0, :, 0], [.75, .2475, .75, .2475])
            result = rvc_features(units, 'tomori', f0, profile, 'call')
            np.testing.assert_allclose(result[0, :, 0], [.5, .075, .5, .075])

    def test_startup_does_not_read_unused_large_index(self):
        profile = SimpleNamespace(index_rate=0, sample_rate=40000, target_hz=310)
        graph = SimpleNamespace(run=lambda *args: [np.zeros((1, 1, 40000), np.float32)])
        with patch('backend.startup.local_rvc.profile', return_value=profile), \
             patch('backend.startup.local_rvc.graph', return_value=graph), \
             patch('backend.startup.voice_index', side_effect=AssertionError('Unused index loaded')):
            startup = Startup()
            startup._load_voice('anon')
            self.assertIn('anon', startup._loaded)


if __name__ == '__main__':
    unittest.main(verbosity=2)
