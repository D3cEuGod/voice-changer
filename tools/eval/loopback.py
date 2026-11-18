"""Pace files as a microphone, convert over WebSocket, record the real virtual microphone.

Run with .venv-gui/Scripts/python.exe. Requires paired WASAPI virtual audio devices.
This checks transport and playback, not Discord's remote audio or human listening.
"""
import argparse
import json
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import soxr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gui.audio import SpeakerStream, discord_pair, list_devices, live_slack
from gui.backend import BackendClient
from gui.live import LiveMonitor


class FileMic:
    active = True
    name = "paced test file"
    dropped = 0
    level = 0

    def __init__(self, rate=22050):
        self.rate = rate
        self.blocks = queue.Queue()

    def close(self):
        self.active = False


def test(client, path, voice, mode, sender, receiver, output, repeat,
         slack_override=None, device_latency=None, packet_ms=200, input_rate=22050, pitch_preset='model'):
    signal, rate = sf.read(path, dtype="float32", always_2d=True)
    signal = np.tile(soxr.resample(signal.mean(axis=1), rate, input_rate), repeat)
    mic = FileMic(input_rate)
    health = client.health()
    reference_voices = health.get("reference_voices", ("tomori", "taffy") if health.get("reference_engine") else ())
    slack = (slack_override if slack_override is not None else
             live_slack(voice in reference_voices, voice, mode, True, sender.name))
    speaker = SpeakerStream(sender, slack_seconds=slack, receiver=receiver, device_latency=device_latency)
    device_latency = speaker.latency
    expected, recorded, ticks, stops, capture_origin = [], [], [], [], []
    capture_ready = threading.Event()
    capture_overflows = []

    def capture_audio(block, frames, clock, status):
        if not capture_origin:
            capture_origin.append(clock.inputBufferAdcTime)
            capture_ready.set()
        recorded.append(block[:, 0].copy())
        if status.input_overflow:
            capture_overflows.append(True)
    write = speaker.write

    def tee(samples):
        expected.append(samples.copy())
        write(samples)

    speaker.write = tee
    ready = threading.Event()
    url = client.live_url(voice, 1, mode, packet_ms=packet_ms, pitch_preset=pitch_preset)
    monitor = LiveMonitor(url, mic, speaker,
                          lambda hello: ready.set(), ticks.append, lambda *args: stops.append(args))
    capture_rate = 48000
    try:
        with sd.InputStream(device=receiver.index, samplerate=capture_rate, channels=1, latency=.02,
                            extra_settings=sd.WasapiSettings(auto_convert=True),
                            callback=capture_audio) as capture:
            speaker.start()
            monitor.start()
            if not ready.wait(15):
                raise RuntimeError(f"WebSocket handshake failed: {stops}")
            if not capture_ready.wait(2):
                raise RuntimeError("Capture device did not start.")
            capture_prefix = capture.time - capture_origin[0]
            start = time.perf_counter()
            padded = np.r_[np.zeros(round(.2*input_rate), np.float32), signal, np.zeros(input_rate, np.float32)]
            block = round(.02*input_rate)
            for offset in range(0, len(padded), block):
                time.sleep(max(0, start+(offset+block)/input_rate-time.perf_counter()))
                mic.blocks.put(padded[offset:offset+block])
            monitor.request_stop()
            monitor.join(15)
            time.sleep(.2)
    finally:
        monitor.request_stop(abort=True)
        if monitor.ident is not None:
            monitor.join(5)
        speaker.close()
    if stops != [("user", "")] or not expected or not recorded:
        raise RuntimeError(f"Conversion did not complete: {stops}")
    sent = soxr.resample(np.concatenate(expected), speaker.rate, capture_rate)
    heard = np.concatenate(recorded)
    size = 1 << (len(heard)+len(sent)-1).bit_length()
    correlation = np.fft.irfft(np.fft.rfft(heard, size)*np.conj(np.fft.rfft(sent, size)), size)
    lag = int(correlation.argmax())
    norm = np.linalg.norm(sent.astype(np.float64))*np.linalg.norm(heard.astype(np.float64))
    agreement = float(np.clip(correlation[lag]/max(norm, 1e-12), -1., 1.))
    segments = []
    for offset in range(0,len(sent)-capture_rate+1,capture_rate//2):
        expected_part = sent[offset:offset+capture_rate].astype(np.float64)
        heard_part = heard[lag+offset:lag+offset+capture_rate].astype(np.float64)
        if len(heard_part) == capture_rate and np.sqrt(np.mean(expected_part**2)) > .005:
            denominator = np.linalg.norm(expected_part)*np.linalg.norm(heard_part)
            segments.append(float(np.dot(expected_part,heard_part)/max(denominator,1e-12)))
    name = f"{path.stem}-{voice}-{mode}"
    sf.write(output / f"{name}-received.wav", heard, capture_rate)
    sf.write(output / f"{name}-sent.wav", sent, capture_rate)
    (output / f"{name}-trace.json").write_text(json.dumps(ticks), encoding="utf-8")
    return {"input": str(path), "voice": voice, "mode": mode, "repeat": repeat,
            "correlation": round(agreement, 5), "signal_delay_s": round(lag/capture_rate-capture_prefix, 3),
            "worst_voiced_second_correlation": round(min(segments),5) if segments else None,
            "last_voiced_second_correlation": round(segments[-1],5) if segments else None,
            "capture_prefix_s": round(capture_prefix, 4), "capture_overflows": len(capture_overflows),
            "input_rate": input_rate, "frame_samples": monitor.hello["frame_samples"],
            "p95_ms": round(float(np.percentile([tick["last_ms"] for tick in ticks], 95)), 1),
            "max_ms": round(float(max(tick["last_ms"] for tick in ticks)), 1),
            "starvations": speaker.starvations, "device_underflows": speaker.device_underflows,
            "fifo_dropped": speaker.fifo.dropped,
            "playback_slack_s": slack, "device_latency_s": round(device_latency, 4),
            "underruns": speaker.underruns, "full_scale_pct": round(float(np.mean(abs(heard)>=.999)*100), 5),
            "audio_present": bool(np.max(abs(sent))>.02)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("character", "call"), default="character")
    parser.add_argument("--voices", nargs="+", default=("tomori", "taffy", "anon"))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--slack", type=float)
    parser.add_argument("--device-latency", type=float)
    parser.add_argument("--packet-ms", type=int, choices=(80, 100, 120, 200), default=200)
    parser.add_argument("--input-rate", type=int, default=22050)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8765")
    parser.add_argument("--pitch-preset", choices=('model','auto2','auto4'), default='model')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pair = discord_pair(list_devices("input"), list_devices("output"))
    if pair is None:
        raise RuntimeError("No paired virtual microphone devices were found.")
    client = BackendClient(args.backend_url)
    results = {"health": client.health(), "sender": pair[0].name, "receiver": pair[1].name, "tests": []}
    for path in args.inputs:
        for voice in args.voices:
            row = test(client, path, voice, args.mode, *pair, args.output, args.repeat,
                       args.slack, args.device_latency, args.packet_ms, args.input_rate, args.pitch_preset)
            results["tests"].append(row)
            print(json.dumps(row), flush=True)
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return int(any(row["underruns"] or row["capture_overflows"] or row["correlation"] < .99 or not row["audio_present"]
                   or row["full_scale_pct"] for row in results["tests"]))


if __name__ == "__main__":
    sys.exit(main())
