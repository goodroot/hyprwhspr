"""Recording cleanup must release status and capture clients before teardown."""

import socket
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

from tests.test_suspend_resume_recovery import _import_main_isolated


class RecordingCleanupStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def _app(self, status_path):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app.playback_suppressor = types.SimpleNamespace(is_active=False)
        app._autostop_stop_silence_monitor = mock.Mock()
        app._clear_mic_osd_preview_text = mock.Mock()
        app._stop_audio_level_monitoring = mock.Mock()
        patcher = mock.patch.object(self.main, 'RECORDING_STATUS_FILE', status_path)
        patcher.start()
        self.addCleanup(patcher.stop)
        return app

    def test_blocked_hide_releases_capture_client_and_subscriber_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = Path(tmp) / 'recording_status'
            status.write_text('true')
            app = self._app(status)
            server = self.main.RecordingControlServer(
                fifo_path=Path(tmp) / 'recording_control',
                socket_path=Path(tmp) / 'capture.sock',
                on_command=lambda *args: None,
                is_recording=lambda: False,
            )
            app._recording_control_server = server
            stop = threading.Event()
            server._stop_event = stop
            capture_started = threading.Event()
            server._write_fifo = lambda command: capture_started.set() or True
            hide_entered = threading.Event()
            release_hide = threading.Event()
            errors = []
            clients = []
            workers = []

            def block_hide():
                hide_entered.set()
                release_hide.wait()

            def cleanup():
                try:
                    app._cleanup_recording_state()
                except Exception as exc:
                    errors.append(exc)

            def connect_capture():
                capture_started.clear()
                client, connection = socket.socketpair()
                clients.append(client)
                client.settimeout(2)
                worker = threading.Thread(
                    target=server._handle_capture_connection,
                    args=(connection, stop),
                )
                worker.start()
                workers.append(worker)
                client.sendall(b'capture\n')
                self.assertTrue(capture_started.wait(2), 'capture slot was not acquired')
                return client, worker

            app._hide_mic_osd = block_hide
            try:
                client, subscriber = connect_capture()
                cleanup_worker = threading.Thread(target=cleanup)
                cleanup_worker.start()
                workers.append(cleanup_worker)
                self.assertTrue(hide_entered.wait(2), 'cleanup did not reach hide')

                self.assertFalse(status.exists())
                self.assertEqual(client.recv(1), b'', 'capture client did not receive EOF')
                subscriber.join(timeout=2)
                self.assertFalse(subscriber.is_alive())
                self.assertFalse(server.has_capture_subscriber())

                # A new client can acquire the slot even while hide is stalled.
                connect_capture()
                self.assertTrue(server.has_capture_subscriber())
            finally:
                release_hide.set()
                stop.set()
                for client in clients:
                    client.close()
                for worker in workers:
                    worker.join(timeout=2)
                self.assertFalse(any(worker.is_alive() for worker in workers))
            self.assertEqual(errors, [])
