"""A pending background load must recheck engine ownership after taking its lock."""
from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.startup import Startup


class IdleTests(unittest.TestCase):
    def test_session_becoming_busy_while_loader_waits_defers_loading(self):
        startup = Startup()
        busy, attempted, loaded = threading.Event(), threading.Event(), threading.Event()
        lock = threading.Lock()
        class ObservedLock:
            def __enter__(self):
                attempted.set()
                lock.acquire()
            def __exit__(self, *args):
                lock.release()
        startup._load_lock = ObservedLock()
        startup._busy = busy.is_set
        def load(voice):
            self.assertFalse(busy.is_set())
            startup._loaded.add(voice)
            loaded.set()
        startup._load_voice = load
        lock.acquire()
        worker = threading.Thread(target=startup._load_background_voice, args=('anon',), daemon=True)
        worker.start()
        self.assertTrue(attempted.wait(1))
        busy.set()
        lock.release()
        try:
            self.assertFalse(loaded.wait(.15))
        finally:
            busy.clear()
            worker.join(2)
        self.assertTrue(loaded.is_set())
        self.assertFalse(worker.is_alive())

    def test_on_demand_voice_finishes_background_wait_without_waiting_for_call_end(self):
        startup = Startup()
        busy, attempted = threading.Event(), threading.Event()
        busy.set()
        def is_busy():
            attempted.set()
            return busy.is_set()
        startup._busy = is_busy
        startup._load_voice = lambda voice: startup._loaded.add(voice)
        worker = threading.Thread(target=startup._load_background_voice, args=('anon',), daemon=True)
        worker.start()
        self.assertTrue(attempted.wait(1))
        startup.ensure_voice('anon')
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertTrue(busy.is_set())


if __name__ == '__main__':
    unittest.main(verbosity=2)
