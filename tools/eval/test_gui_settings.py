"""1.0 界面的设置保存、麦克风电平提示和耳返分路。不打开真实设备。

运行：.venv-gui\\Scripts\\python.exe -m unittest tools.eval.test_gui_settings -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gui import audio
from gui.settings import Settings


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "nested" / "settings.json"

    def test_round_trip_creates_folder(self):
        settings = Settings.load(self.path)
        settings.voice_id, settings.strength, settings.monitor, settings.mic = "anon", 55, True, "麦克风 (USB)"
        self.assertTrue(settings.save())
        loaded = Settings.load(self.path)
        self.assertEqual((loaded.voice_id, loaded.strength, loaded.monitor, loaded.mic),
                         ("anon", 55, True, "麦克风 (USB)"))
        self.assertFalse(self.path.with_name("settings.json.tmp").exists())

    def test_missing_or_broken_file_uses_defaults(self):
        self.assertEqual(Settings.load(self.path), Settings())
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(Settings.load(self.path), Settings())
        self.path.write_text("[1, 2]", encoding="utf-8")
        self.assertEqual(Settings.load(self.path), Settings())

    def test_invalid_values_fall_back_per_field(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps({
            "strength": 250, "mode": "robot", "live_packet_ms": 80, "live_restart_packets": 7,
            "monitor": 1, "send_to_call": "yes", "voice_id": 3, "output": "耳机", "unknown": True}), encoding="utf-8")
        loaded = Settings.load(self.path)
        self.assertEqual(loaded.strength, 100)
        self.assertEqual(loaded.mode, "character")
        self.assertEqual(loaded.live_packet_ms, 120)
        self.assertEqual(loaded.live_restart_packets, 3000)
        self.assertFalse(loaded.monitor)          # 1 不是布尔值
        self.assertTrue(loaded.send_to_call)
        self.assertIsNone(loaded.voice_id)
        self.assertEqual(loaded.output, "耳机")

    def test_memory_only_settings_never_write(self):
        settings = Settings()
        self.assertIsNone(settings.path)
        self.assertTrue(settings.save())


class MicAdvisorTests(unittest.TestCase):
    def feed(self, peaks):
        advisor = audio.MicAdvisor(interval=.04, window=1.0)
        result = ("", "")
        for peak in peaks:
            result = advisor.feed(peak)
        return result

    def test_needs_a_full_window_before_judging(self):
        self.assertEqual(self.feed([0.0] * 10), ("", ""))

    def test_silence_is_reported_as_quiet(self):
        self.assertEqual(self.feed([.002] * 25)[0], "quiet")

    def test_normal_speech_is_ok(self):
        self.assertEqual(self.feed([.01, .3, .5, .02] * 7)[0], "ok")

    def test_repeated_clipping_is_loud_but_a_single_peak_is_not(self):
        self.assertEqual(self.feed([.4] * 22 + [1.0] * 3)[0], "loud")
        self.assertEqual(self.feed([.4] * 24 + [1.0])[0], "ok")


class _FakeSpeaker:
    def __init__(self, rate, name):
        self.rate, self.name, self.device = rate, name, name
        self.written, self.active, self.level, self.underruns = [], True, .5, 2
        self.drained = self.faded = False

    def write(self, samples):
        self.written.append(np.asarray(samples, dtype=np.float32))

    def start(self):
        pass

    def fade_out(self, seconds=.02):
        self.faded = True

    def begin_drain(self):
        self.drained = True

    def close(self):
        self.active = False


class TeeSpeakerTests(unittest.TestCase):
    def test_primary_drives_stats_and_monitor_gets_resampled_copy(self):
        call, ear = _FakeSpeaker(48000, "CABLE Input"), _FakeSpeaker(44100, "耳机")
        tee = audio.TeeSpeaker(call, ear)
        block = np.sin(np.arange(4800) / 10).astype(np.float32)
        for _ in range(10):
            tee.write(block)
        self.assertEqual(tee.rate, 48000)
        self.assertEqual(tee.name, "CABLE Input")
        self.assertEqual(tee.underruns, 2)        # 统计来自主输出
        self.assertEqual(sum(map(len, call.written)), 48000)
        ear_samples = sum(map(len, ear.written))  # 重采样器会暂存不到 1,000 个样本
        self.assertLessEqual(ear_samples, 44100)
        self.assertGreater(ear_samples, 44100 - 1000)

    def test_failed_monitor_does_not_stop_the_call(self):
        call, ear = _FakeSpeaker(48000, "CABLE Input"), _FakeSpeaker(48000, "耳机")
        tee = audio.TeeSpeaker(call, ear)
        ear.active = False
        tee.write(np.zeros(480, np.float32))
        self.assertTrue(tee.active)
        self.assertEqual(len(call.written), 1)
        self.assertEqual(ear.written, [])

    def test_close_and_drain_reach_both(self):
        call, ear = _FakeSpeaker(48000, "a"), _FakeSpeaker(48000, "b")
        tee = audio.TeeSpeaker(call, ear)
        tee.begin_drain()
        tee.fade_out()
        tee.close()
        self.assertTrue(call.drained and ear.drained and call.faded)
        self.assertFalse(call.active or ear.active)


if __name__ == "__main__":
    unittest.main()
