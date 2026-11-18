"""Fixed RVC transposition and offsets must not be confused with register matching."""
from pathlib import Path
import sys
import unittest
from urllib.parse import parse_qs, urlsplit
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.pitch import transpose_f0
from gui.backend import BackendClient, pitch_query,DEFAULT_PITCH_PRESET


class PitchTests(unittest.TestCase):
    def pitch(self, values, *, shift=None, offset=0, strength=1):
        return transpose_f0(np.array(values, np.float32), strength, np.log(150), 300, shift, offset)

    def test_existing_automatic_mapping_is_unchanged(self):
        np.testing.assert_allclose(self.pitch([0,150,200]), [0,300,400], rtol=1e-6)

    def test_fixed_four_replaces_automatic_mapping(self):
        np.testing.assert_allclose(self.pitch([0,150,200],shift=4),
                                   np.array([0,150,200])*2**(4/12), rtol=1e-6)

    def test_offset_four_adds_to_automatic_mapping(self):
        np.testing.assert_allclose(self.pitch([0,150,200],offset=4),
                                   np.array([0,300,400])*2**(4/12), rtol=1e-6)

    def test_zero_shift_overrides_automatic_pitch_and_keeps_unvoiced_zero(self):
        np.testing.assert_array_equal(self.pitch([0,150,200],shift=0), [0,150,200])

    def test_pitch_strength_and_safety_ceiling(self):
        self.assertAlmostEqual(self.pitch([150],shift=4,strength=.5)[0],150*2**(2/12),places=4)
        np.testing.assert_array_equal(self.pitch([0,600],offset=12),[0,650])

    def test_client_presets_encode_the_two_different_operations(self):
        self.assertEqual(DEFAULT_PITCH_PRESET,'auto2')
        self.assertEqual(pitch_query('auto2'),{'pitch_offset':2})
        self.assertEqual(pitch_query('model'),{})
        with self.assertRaises(KeyError):
            pitch_query('fixed4')
        self.assertEqual(pitch_query('auto4'),{'pitch_offset':4})
        client = BackendClient()
        query = parse_qs(urlsplit(client.live_url('taffy',1,pitch_preset='auto4')).query)
        self.assertEqual(query['pitch_offset'],['4'])
        self.assertNotIn('pitch_shift',query)

    def test_new_default_adds_two_without_replacing_automatic_register(self):
        np.testing.assert_allclose(self.pitch([0,150,200],offset=2),
                                   np.array([0,300,400])*2**(2/12),rtol=1e-6)
        query=parse_qs(urlsplit(BackendClient().live_url('tannhauser',1,pitch_preset='auto2')).query)
        self.assertEqual(query['pitch_offset'],['2'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
