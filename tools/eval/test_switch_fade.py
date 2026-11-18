"""Intentional role switches must fade and discard old audio before closing."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from gui.__main__ import create_app, cleanup_qt_styles
APP=create_app([])
from gui.audio import Device, SpeakerStream


class FadeTests(unittest.TestCase):
    def speaker(self):
        native=Mock(active=True,latency=.02)
        with patch('gui.audio._open_stream',return_value=(native,2)):
            speaker=SpeakerStream(Device(1,'test output',48000,2,False,'WASAPI'))
        speaker._waiting=False
        return speaker

    def test_switch_fades_across_callbacks_then_discards_old_queue(self):
        speaker=self.speaker()
        speaker.write(np.full(8000,.2,np.float32))
        with patch.object(speaker._fade_done,'wait'):
            speaker.fade_out()
        parts=[]
        for _ in range(6):
            output=np.empty((256,2),np.float32)
            speaker._callback(output,256,None,SimpleNamespace(output_underflow=False))
            parts.append(output[:,0].copy())
        samples=np.concatenate(parts)
        self.assertLessEqual(float(np.max(np.diff(samples))),1e-7)
        self.assertGreater(samples[0],.19)
        self.assertEqual(float(np.max(np.abs(samples[960:]))),0)
        self.assertEqual(len(speaker.fifo),0)
        self.assertTrue(speaker._fade_done.is_set())
        self.assertEqual(speaker.underruns,0)
        speaker.write(np.ones(100,np.float32))
        self.assertEqual(len(speaker.fifo),0)
        speaker.close()

    def test_empty_queue_during_switch_finishes_with_silence(self):
        speaker=self.speaker()
        with patch.object(speaker._fade_done,'wait'):
            speaker.fade_out()
        output=np.ones((2048,2),np.float32)
        speaker._callback(output,2048,None,SimpleNamespace(output_underflow=False))
        self.assertEqual(float(np.max(np.abs(output))),0)
        self.assertTrue(speaker._fade_done.is_set())
        self.assertEqual(speaker.underruns,0)
        speaker.close()


if __name__=='__main__':
    try:
        unittest.main(verbosity=2)
    finally:
        cleanup_qt_styles()
