import os
import socket
import stat
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "src"))

from recording_control_server import RecordingControlServer


class RecordingControlServerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.fifo = root / "recording_control"
        self.sock = root / "capture.sock"
        self.commands = []
        self.recording = False
        self.server = RecordingControlServer(
            self.fifo, self.sock, lambda action, language: self.commands.append((action, language)),
            lambda: self.recording,
        )

    def tearDown(self):
        self.server.stop(fifo_timeout=1, capture_timeout=2)
        self.tempdir.cleanup()

    def _connect(self, request=b"capture\n"):
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(2)
        client.connect(str(self.sock))
        client.sendall(request)
        return client

    def _wait_for(self, predicate, timeout=2):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def test_parsing_preserves_language_and_selects_last_valid(self):
        self.assertEqual(
            RecordingControlServer.parse_commands("start:pt-BR\ninvalid\nSTOP\n"),
            ("stop", None),
        )
        self.assertEqual(
            RecordingControlServer.parse_commands("stop\nSTART:pt-BR\n"),
            ("start", "pt-BR"),
        )
        self.assertIsNone(RecordingControlServer.parse_commands("\ninvalid\nstartling\n"))

    def test_prepare_fifo_reuses_fifo_and_replaces_regular_file(self):
        self.fifo.write_text("legacy")
        self.assertTrue(self.server.prepare_fifo())
        self.assertTrue(self.fifo.is_fifo())
        inode = self.fifo.stat().st_ino
        self.assertTrue(self.server.prepare_fifo())
        self.assertEqual(self.fifo.stat().st_ino, inode)

    def test_prepare_fifo_failure_is_reported(self):
        with mock.patch("recording_control_server.os.mkfifo", side_effect=PermissionError):
            self.assertFalse(self.server.prepare_fifo())

    def test_fifo_dispatch_and_recreation(self):
        self.assertTrue(self.server.prepare_fifo())
        self.assertTrue(self.server.start())
        fd = os.open(self.fifo, os.O_WRONLY)
        os.write(fd, b"invalid\nstart:EN-us\n")
        os.close(fd)
        self.assertTrue(self._wait_for(lambda: self.commands == [("start", "EN-us")]))
        self.fifo.unlink()
        self.assertTrue(self._wait_for(self.fifo.is_fifo))

    def test_empty_fifo_writer_reopens_reader_after_eof(self):
        self.assertTrue(self.server.prepare_fifo())
        real_open = os.open
        with mock.patch("recording_control_server.os.open", wraps=real_open) as open_mock:
            self.assertTrue(self.server.start())
            self.assertTrue(self._wait_for(lambda: any(
                call.args[1] == os.O_RDONLY | os.O_NONBLOCK
                for call in open_mock.call_args_list
            )))
            fd = real_open(self.fifo, os.O_WRONLY)
            os.close(fd)

            def reader_open_count():
                return sum(
                    call.args[1] == os.O_RDONLY | os.O_NONBLOCK
                    for call in open_mock.call_args_list
                )

            self.assertTrue(self._wait_for(lambda: reader_open_count() >= 2))

    def test_start_stop_start_and_socket_permissions_cleanup(self):
        self.assertTrue(self.server.prepare_fifo())
        self.assertTrue(self.server.start())
        self.assertEqual(stat.S_IMODE(self.sock.stat().st_mode), 0o600)
        self.assertFalse(self.server.start())
        self.assertTrue(self.server.stop())
        self.assertTrue(self._wait_for(lambda: not self.sock.exists()))
        self.assertTrue(self.server.start())

    def test_capture_stream_and_occupied_slot(self):
        self.assertTrue(self.server.prepare_fifo())
        self.assertTrue(self.server.start())
        first = self._connect()
        self.assertTrue(self._wait_for(self.server.has_capture_subscriber))
        second = self._connect()
        self.assertEqual(second.recv(100), b"ERROR:slot_occupied\n")
        second.close()
        self.server.notify_capture("hello", final=True)
        self.assertEqual(first.recv(100), b"hello")
        self.assertEqual(first.recv(1), b"")
        first.close()

    def test_invalid_capture_request_is_closed(self):
        self.assertTrue(self.server.prepare_fifo())
        self.assertTrue(self.server.start())
        client = self._connect(b"unknown:en\n")
        self.assertEqual(client.recv(1), b"")
        self.assertFalse(self.server.has_capture_subscriber())
        client.close()

    def test_active_capture_joins_without_self_trigger(self):
        self.recording = True
        self.assertTrue(self.server.prepare_fifo())
        self.assertTrue(self.server.start())
        client = self._connect(b"capture:fr\n")
        self.assertTrue(self._wait_for(self.server.has_capture_subscriber))
        time.sleep(0.05)
        self.assertEqual(self.commands, [])
        self.server.notify_capture("bonjour", final=True)
        self.assertEqual(client.recv(100), b"bonjour")
        client.close()

    def test_idle_capture_self_triggers_with_language(self):
        self.assertTrue(self.server.prepare_fifo())
        self.assertTrue(self.server.start())
        client = self._connect(b"capture:Ja-JP\n")
        self.assertTrue(self._wait_for(lambda: self.commands == [("start", "Ja-JP")]))
        self.server.notify_capture("", final=True)
        client.close()

    def test_disconnect_sends_cancel(self):
        def record_after_start(action, language):
            self.commands.append((action, language))
            if action == "start":
                self.recording = True

        self.server._on_command = record_after_start
        self.assertTrue(self.server.prepare_fifo())
        self.assertTrue(self.server.start())
        client = self._connect()
        self.assertTrue(self._wait_for(lambda: self.recording))
        client.close()
        self.assertTrue(self._wait_for(lambda: ("cancel", None) in self.commands))

    def test_stop_before_start_is_safe(self):
        self.assertTrue(self.server.stop())


if __name__ == "__main__":
    unittest.main()
