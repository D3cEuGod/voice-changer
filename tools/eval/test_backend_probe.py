"""A fast connect probe must still allow normal HTTP response latency."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from gui.backend import BackendClient, BackendError, BackendLauncher


class DelayedHealth(BaseHTTPRequestHandler):
    def do_GET(self):
        time.sleep(.25)
        data=b'{"status":"ok","version":"test"}'
        self.send_response(200)
        self.send_header('Content-Length',str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self,*args):
        pass


class ProbeTests(unittest.TestCase):
    def test_existing_loading_server_reports_progress_without_second_process(self):
        class LoadingHealth(DelayedHealth):
            calls=0
            def do_GET(self):
                type(self).calls+=1
                data=(b'{"status":"loading","progress":0.4,"stage":"models"}'
                      if self.calls<3 else b'{"status":"ok"}')
                self.send_response(200)
                self.send_header('Content-Length',str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        server=ThreadingHTTPServer(('127.0.0.1',0),LoadingHealth)
        worker=threading.Thread(target=server.serve_forever,daemon=True)
        worker.start()
        try:
            ready,failed,stages=[],[],[]
            launcher=BackendLauncher(BackendClient(f'http://127.0.0.1:{server.server_port}'),
                lambda _:None,lambda *args:ready.append(args),failed.append,
                on_stage=lambda *args:stages.append(args),wait_seconds=3)
            with patch('gui.backend.spawn_backend') as spawn:
                launcher.run()
                spawn.assert_not_called()
            self.assertEqual(failed,[])
            self.assertFalse(ready[0][1])
            self.assertIn(('models',.4,'models'),stages)
        finally:
            server.shutdown();server.server_close();worker.join(2)

    def test_existing_failed_server_does_not_spawn_duplicate_backend(self):
        class FailedHealth(DelayedHealth):
            def do_GET(self):
                data=b'{"detail":"model failed"}'
                self.send_response(503)
                self.send_header('Content-Length',str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        server=ThreadingHTTPServer(('127.0.0.1',0),FailedHealth)
        worker=threading.Thread(target=server.serve_forever,daemon=True)
        worker.start()
        try:
            ready,failed=[],[]
            launcher=BackendLauncher(BackendClient(f'http://127.0.0.1:{server.server_port}'),
                lambda _:None,lambda *args:ready.append(args),failed.append)
            with patch('gui.backend.spawn_backend') as spawn:
                launcher.run()
                spawn.assert_not_called()
            self.assertEqual(ready,[])
            self.assertIn('model failed',failed[0])
        finally:
            server.shutdown();server.server_close();worker.join(2)

    def test_closed_port_fails_quickly(self):
        with socket.socket() as reserved:
            reserved.bind(('127.0.0.1',0))
            client=BackendClient(f'http://127.0.0.1:{reserved.getsockname()[1]}')
            start=time.perf_counter()
            with self.assertRaises(BackendError):
                client.health(connect_timeout=.1)
            self.assertLess(time.perf_counter()-start,.8)

    def test_slow_http_response_is_not_mistaken_for_dead_backend(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),DelayedHealth)
        worker=threading.Thread(target=server.serve_forever,daemon=True)
        worker.start()
        try:
            ready,failed=[],[]
            client=BackendClient(f'http://127.0.0.1:{server.server_port}')
            launcher=BackendLauncher(client,lambda _:None,lambda *args:ready.append(args),failed.append)
            launcher.run()
            self.assertEqual(failed,[])
            self.assertEqual(ready[0][0]['status'],'ok')
            self.assertFalse(ready[0][1])
            self.assertIsNone(launcher.process)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(2)


if __name__=='__main__':
    unittest.main(verbosity=2)
