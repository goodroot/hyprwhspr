import io
import sys
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from src.cli import record


class FakeSocket:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.sent = b""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def connect(self, path):
        self.path = path

    def sendall(self, data):
        self.sent += data

    def recv(self, size):
        return next(self.chunks)


class FakeStdout:
    def __init__(self):
        self.buffer = io.BytesIO()

    def flush(self):
        pass


class RecordCaptureCliTests(unittest.TestCase):
    def _capture(self, chunks, **kwargs):
        sock = FakeSocket(chunks)
        stdout = FakeStdout()
        socket_path = mock.MagicMock()
        socket_path.exists.return_value = True
        socket_path.__str__ = mock.Mock(return_value="/tmp/capture.sock")
        with (
            mock.patch.object(record, "SOCKET_FILE", socket_path),
            mock.patch.object(record.socket, "socket", return_value=sock),
            mock.patch.object(record.sys, "stdout", stdout),
        ):
            record.record_capture_command(**kwargs)
        return sock.sent, stdout.buffer.getvalue()

    def test_ordinary_capture_request_and_output_remain_raw(self):
        sent, output = self._capture(["Café\n東京".encode(), b""])
        self.assertEqual(sent, b"capture\n")
        self.assertEqual(output, "Café\n東京".encode())

    def test_trace_capture_uses_private_trace_verb_and_preserves_json(self):
        payload = b'{"raw":"hello","preprocessed":"hello"}\n'
        sent, output = self._capture(
            [payload, b""], language="fr", trace_processing=True
        )
        self.assertEqual(sent, b"capture_trace:fr\n")
        self.assertEqual(output, payload)


if __name__ == "__main__":
    unittest.main()
