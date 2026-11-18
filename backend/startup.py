"""Load models after the server starts listening, and report progress through /api/health.

The server opens its port before any model is loaded. A background thread loads the shared graphs and the
voice the GUI will show first (VOICE_WARM_FIRST, default taffy), runs one real warm-up render as a self-test,
and only then reports ready. The other voices load afterwards in the same thread. A request for a voice that
is not loaded yet loads it on the spot, under the same lock, so nothing is loaded twice.

Background loading also primes each voice's private generator with fixed tensors. It never runs a shared
feature or pitch graph, because DirectML forbids concurrent Run calls on one session.
"""
import logging
import os
import threading
import time

import numpy as np

from . import engine, local_rvc
from .character import graph, voice_index
from .live import SHORT_FRAME, SPEECH_LOOKAHEAD, LiveVoice
from .pitch import model as pitch_model

LOG = logging.getLogger("uvicorn.error")


class Startup:
    def __init__(self):
        self._busy = lambda: False
        self._client_seen = threading.Event()  # set when a client has fetched the voice list
        self._load_lock = threading.Lock()   # one model load at a time, startup or on demand
        self._state_lock = threading.Lock()
        self._loaded: set[str] = set()
        self._started = time.perf_counter()
        self._state = {"ready": False, "progress": 0.0, "stage": "准备加载模型", "error": "",
                       "pending": [], "first_voice": "", "ready_seconds": None}

    # ---- state shared with request handlers ---------------------------------------------
    def snapshot(self) -> dict:
        with self._state_lock:
            return dict(self._state, pending=list(self._state["pending"]))

    @property
    def ready(self) -> bool:
        with self._state_lock:
            return self._state["ready"]

    def _update(self, **values) -> None:
        with self._state_lock:
            self._state.update(values)

    # ---- loading -------------------------------------------------------------------------
    @staticmethod
    def first_voice() -> str:
        """先加载的声线；还没有导入任何声线（公开版刚装好）时为空字符串。"""
        wanted = os.environ.get("VOICE_WARM_FIRST", "taffy")
        return wanted if wanted in engine.VOICES else next(iter(engine.VOICES), "")

    def start(self, busy=None) -> None:
        """busy() is true while a conversion or live session runs; background voices wait for it."""
        if busy is not None:
            self._busy = busy
        threading.Thread(target=self._run, daemon=True, name="model-startup").start()

    def client_ready(self) -> None:
        """The GUI has fetched the voice list: background loading can start without delaying it."""
        self._client_seen.set()

    def ensure_voice(self, voice: str) -> None:
        """Called before every conversion or live session: wait for a background load in progress, then
        load this voice if the background loader has not reached it yet.

        Taking the lock even for a voice that is already loaded keeps a session from starting while another
        voice's session is being created: ONNX Runtime and FAISS hold the GIL while they load, which would
        stall the session's audio, and a live session once failed when it overlapped a cold index load.
        """
        with self._load_lock:
            self._load_voice(voice)

    def _load_voice(self, voice: str) -> None:
        if voice in self._loaded:
            return
        if local_rvc.profile(voice) is not None:
            profile = local_rvc.profile(voice)
            model = local_rvc.graph(voice)
            if profile.index_rate > 0:
                voice_index(voice)
            # Session construction does not compile every DirectML kernel. Do
            # this before marking a voice loaded, so its first spoken syllable
            # does not pay that cost and empty the playback queue. Only this
            # voice's private generator runs here; shared graphs stay untouched.
            generated = model.run(None, {
                'units': np.zeros((1, 100, 768), np.float32),
                'f0': np.full((1, 100), profile.target_hz, np.float32),
                'pitch': np.full((1, 100), 128, np.int64),
                'noise': np.zeros((1, 192, 100), np.float32),
            })[0]
            if generated.size != profile.sample_rate or not np.isfinite(generated).all():
                raise RuntimeError(f'角色模型预热返回无效音频：{voice}')
        # Voices without a local profile use the legacy graphs, which load on first use.
        self._loaded.add(voice)
        with self._state_lock:
            if voice in self._state["pending"]:
                self._state["pending"].remove(voice)

    def _load_background_voice(self, voice: str) -> float:
        while voice not in self._loaded:
            # A request can claim the engine while this thread waits for the
            # load lock. Recheck under the lock before starting disk / ORT work.
            with self._load_lock:
                if not self._busy():
                    started = time.perf_counter()
                    self._load_voice(voice)
                    return time.perf_counter()-started
            time.sleep(.05)
        return 0.0

    def _warm(self, voice: str) -> None:
        """One real render of the bundled sentence: compiles the GPU kernels and checks the voice answers."""
        sample = engine.decode((engine.ROOT / "assets/previews/original.wav").read_bytes())
        frames = np.pad(sample, (0, (-len(sample)) % SHORT_FRAME)).reshape(-1, SHORT_FRAME)
        live = LiveVoice(voice, 1, SPEECH_LOOKAHEAD)
        for frame in frames:
            if np.max(np.abs(live.process(frame))) > .002:
                return
        raise RuntimeError(f"角色模型启动自检没有检测到语音：{voice}")

    def _run(self) -> None:
        first = self.first_voice()
        name = engine.VOICES.get(first, "")
        others = [v for v in engine.VOICES if v != first]
        self._update(first_voice=first, pending=list(engine.VOICES))
        local = all(local_rvc.profile(v) is not None for v in engine.VOICES)
        # Weights are the measured seconds of each step on the development machine; they only shape the bar.
        steps = []
        if local:
            steps += [(0.1, "加载降噪和语音检测模型",
                       lambda: (graph("gtcrn_simple"), graph("silero_vad_16k_op15"))),
                      (0.9, "加载发音特征模型（ContentVec）", lambda: graph("contentvec_768l12")),
                      (0.4, "加载音高模型（RMVPE）", pitch_model)]
        if first:
            steps += [(1.0, f"加载{name}的声线模型", lambda: self._load_voice(first)),
                      (0.4, f"预热{name}", lambda: self._warm(first))]
        total = sum(weight for weight, _, _ in steps)
        done = 0.0
        try:
            with self._load_lock:
                for weight, stage, action in steps:
                    self._update(stage=stage, progress=round(done / total, 3))
                    LOG.info('VOICE_STARTUP %.3f %s', done / total, stage)  # 界面读这一行更新加载页
                    started = time.perf_counter()
                    action()
                    LOG.info("Startup: %s in %.2f s", stage, time.perf_counter() - started)
                    done += weight
            seconds = round(time.perf_counter() - self._started, 2)
            self._update(ready=True, progress=1.0, stage="就绪", ready_seconds=seconds)
            LOG.info('VOICE_STARTUP 1.000 就绪')
            LOG.info("Startup: ready for %s after %.2f s; %d more voices load in the background",
                     first, seconds, len(others))
            # Loading a voice holds the GIL for about a second and would delay the GUI's own health and
            # voice-list requests right when it is about to show the main window. Give it up to 3 s first.
            self._client_seen.wait(3.0)
            for voice in others:
                # Creating a session holds Python's GIL for about a second, which would stall a live
                # session's audio. Wait until the engine is idle; a voice the user picks meanwhile is
                # loaded on demand by ensure_voice().
                elapsed = self._load_background_voice(voice)
                LOG.info("Startup: background voice %s in %.2f s", voice, elapsed)
        except Exception as exc:  # report through /api/health instead of leaving a silent server
            LOG.exception("Model startup failed")
            self._update(error=f"{type(exc).__name__}: {exc}", stage="模型加载失败")


startup = Startup()
