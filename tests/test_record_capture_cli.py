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

    def settimeout(self, timeout):
        self.timeout = timeout

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
    def test_recovery_commands_send_once_and_accept_split_response(self):
        for action in ('copy-last', 'paste-last', 'clear-last'):
            with self.subTest(action=action):
                sock = FakeSocket([b'{"ok":', b'true,"message":"done"}\n'])
                with (
                    mock.patch.object(record.socket, 'socket', return_value=sock),
                    mock.patch.object(record, 'log_success') as success,
                ):
                    record.record_recovery_command(action)
                self.assertEqual(sock.sent, ('{"verb": "' + action.replace('-', '_') + '"}\n').encode())
                self.assertEqual(sock.timeout, 15.0)
                success.assert_called_once_with('done')

    def test_recovery_errors_exit_without_retry(self):
        cases = [
            [b'{"ok":false,"error":"unknown request"}\n'],
            [b''], [b'not json\n'], [b'[]\n'],
            [b'{"ok":"yes"}\n'], [b'x' * 65537],
        ]
        for chunks in cases:
            with self.subTest(chunks=chunks[:1]):
                sock = FakeSocket(chunks)
                with (
                    mock.patch.object(record.socket, 'socket', return_value=sock) as factory,
                    mock.patch.object(record, 'log_error'),
                    self.assertRaises(SystemExit) as raised,
                ):
                    record.record_recovery_command('paste-last')
                self.assertEqual(raised.exception.code, 1)
                factory.assert_called_once()

    def test_recovery_connection_and_timeout_errors_exit(self):
        for error in (FileNotFoundError(), ConnectionRefusedError(), TimeoutError()):
            with self.subTest(error=type(error).__name__):
                sock = FakeSocket([])
                with (
                    mock.patch.object(record.socket, 'socket', return_value=sock),
                    mock.patch.object(sock, 'connect', side_effect=error) as connect,
                    mock.patch.object(record, 'log_error'),
                    self.assertRaises(SystemExit) as raised,
                ):
                    record.record_recovery_command('paste-last')
                self.assertEqual(raised.exception.code, 1)
                connect.assert_called_once()

    def test_busy_response_has_no_uncertain_paste_warning(self):
        sock = FakeSocket([b'{"ok":false,"error":"Text delivery is busy"}\n'])
        with (
            mock.patch.object(record.socket, 'socket', return_value=sock),
            mock.patch.object(record, 'log_error') as error,
            self.assertRaises(SystemExit),
        ):
            record.record_recovery_command('paste-last')
        error.assert_called_once()
        self.assertIn('busy', error.call_args.args[0])
        self.assertNotIn('dispatched', error.call_args.args[0])

    def test_only_post_request_paste_timeout_warns_about_dispatch(self):
        for action, during_reply in (('paste-last', True), ('paste-last', False), ('copy-last', True)):
            with self.subTest(action=action, during_reply=during_reply):
                sock = FakeSocket([])
                with (
                    mock.patch.object(record.socket, 'socket', return_value=sock),
                    mock.patch.object(sock, 'recv' if during_reply else 'connect', side_effect=TimeoutError),
                    mock.patch.object(record, 'log_error') as error,
                    self.assertRaises(SystemExit),
                ):
                    record.record_recovery_command(action)
                messages = ' '.join(call.args[0] for call in error.call_args_list)
                self.assertEqual('already have been dispatched' in messages, during_reply and action == 'paste-last')

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
