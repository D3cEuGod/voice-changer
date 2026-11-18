"""Quiet syllables, VAD confidence and speech boundaries; no GPU or devices."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.character import SpeechDetector, audio_gate


class Probabilities:
    def __init__(self, values):
        self.values = iter(values)

    def run(self, _, inputs):
        return np.array([[next(self.values)]], np.float32), inputs['state']


class GateTests(unittest.TestCase):
    def detect(self, values):
        with patch('backend.character.graph', return_value=Probabilities(values)):
            detector = SpeechDetector()
        output = detector.process(np.zeros(512*len(values), np.float32))
        return detector, output

    def test_quiet_speech_opens_without_turning_confidence_into_volume(self):
        _, gate = self.detect([.4]*20)
        self.assertGreater(gate[-5000:].mean(), .99)

    def test_soft_syllables_after_speech_are_retained(self):
        _, gate = self.detect([.8]+[.18]*20)
        self.assertGreater(gate[-5000:].mean(), .99)

    def test_background_does_not_open_gate(self):
        _, gate = self.detect([.1]*30)
        self.assertEqual(float(gate.max()), 0)

    def test_boundaries_retain_consonants_but_close_in_long_pause(self):
        _, gate = self.detect([.01]*8+[.8]*3+[.01]*15)
        onset = 16000-26*512+8*512
        self.assertGreater(gate[onset-1000], .99)
        self.assertGreater(gate[onset+3*512+2500], .99)
        self.assertEqual(float(gate[-1000:].max()), 0)

    def test_accepted_soft_audio_is_not_muted_by_second_volume_gate(self):
        activity = np.ones(16000, np.float32)
        gate = audio_gate(np.full(16000, .0008, np.float32), activity, 22050)
        self.assertGreater(float(gate.min()), .99)
        self.assertEqual(float(audio_gate(np.zeros(16000), activity, 22050).max()), 0)

    def test_pending_partial_vad_block_does_not_lose_onset(self):
        with patch('backend.character.graph', return_value=Probabilities([.8]*32)):
            detector = SpeechDetector()
        detector.process(np.zeros(500, np.float32))
        self.assertEqual(len(detector.pending), 500)
        gate = detector.process(np.zeros(500, np.float32))
        self.assertGreater(gate[-100], .99)
        self.assertEqual(len(detector.pending), 488)

    def test_weak_onset_above_background_opens_early(self):
        with patch('backend.character.graph', return_value=Probabilities([.01]*20+[.25]*4)):
            detector = SpeechDetector()
        detector.process(np.full(20*512,.0005,np.float32))
        gate = detector.process(np.full(4*512,.004,np.float32))
        self.assertGreater(gate[-1000:].mean(), .99)

    def test_low_confidence_background_does_not_use_weak_onset_path(self):
        with patch('backend.character.graph', return_value=Probabilities([.01]*20+[.25]*4)):
            detector = SpeechDetector()
        detector.process(np.full(20*512,.001,np.float32))
        gate = detector.process(np.full(4*512,.001,np.float32))
        self.assertEqual(float(gate.max()),0)

    def test_residual_background_is_suppressed_after_speech(self):
        clean=np.full(16000,.0003,np.float32)
        activity=np.ones(16000,np.float32)
        self.assertGreater(audio_gate(clean,activity,22050).mean(),.3)
        self.assertEqual(float(audio_gate(clean,activity,22050,noise_rms=.00025).max()),0)

    def test_soft_ending_preset_preserves_speech_but_keeps_silence_zero(self):
        clean=np.full(16000,.0003,np.float32);active=np.ones(16000,np.float32)
        ordinary=audio_gate(clean,active,22050)
        protected=audio_gate(clean,active,22050,full_rms=.00045)
        self.assertGreater(float(protected.mean()),float(ordinary.mean())*1.4)
        self.assertEqual(float(audio_gate(np.zeros(16000),active,22050,full_rms=.00045).max()),0)
        self.assertEqual(float(audio_gate(clean,np.zeros(16000),22050,full_rms=.00045).max()),0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
