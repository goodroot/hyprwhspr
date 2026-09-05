import json
import socket
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib' / 'src'))

from recording_control_server import RecordingControlServer  # noqa: E402


class FileTranscriptionProtocolTests(unittest.TestCase):
    def _exchange(self, request, callback=None, recover=None, stopping=False):
        callback = callback or (lambda path, language, clean: (True, 'text'))
        stop_event = threading.Event()
        server = RecordingControlServer(
            '/tmp/unused-fifo', '/tmp/unused-socket', lambda *_: None,
            lambda: False, on_file_transcribe=callback, on_recover=recover,
        )
        server._stop_event = stop_event
        if stopping:
            stop_event.set()
        client, peer = socket.socketpair()
        worker = threading.Thread(
            target=server._handle_capture_connection, args=(peer, stop_event)
        )
        worker.start()
        client.sendall(request)
        client.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        worker.join(timeout=2)
        client.close()
        self.assertFalse(worker.is_alive())
        return json.loads(b''.join(chunks).decode('utf-8'))

    def test_recovery_verbs_do_not_transcribe_files(self):
        transcribe = mock.Mock()
        for verb in ('copy_last', 'paste_last', 'clear_last'):
            with self.subTest(verb=verb):
                recover = mock.Mock(return_value=(True, 'done'))
                response = self._exchange(
                    json.dumps({'verb': verb}).encode() + b'\n',
                    callback=transcribe, recover=recover,
                )
                self.assertEqual(response, {'ok': True, 'message': 'done'})
                recover.assert_called_once_with(verb)
        transcribe.assert_not_called()

    def test_recovery_failure_unavailable_and_shutdown(self):
        request = b'{"verb":"copy_last"}\n'
        recover = mock.Mock(return_value=(False, 'nothing saved'))
        self.assertEqual(self._exchange(request, recover=recover),
                         {'ok': False, 'error': 'nothing saved'})
        self.assertIn('unavailable', self._exchange(request)['error'])
        recover.reset_mock()
        self.assertIn('shutting down', self._exchange(request, recover=recover, stopping=True)['error'])
        recover.assert_not_called()

    def test_structured_request_and_success_response(self):
        seen = []
        request = json.dumps({
            'verb': 'transcribe_file', 'path': '/tmp/audio.mp3',
            'language': 'fr', 'clean': True,
        }).encode() + b'\n'
        response = self._exchange(
            request,
            lambda path, language, clean: (
                seen.append((path, language, clean)) or True, 'bonjour'
            ),
        )
        self.assertEqual(seen, [('/tmp/audio.mp3', 'fr', True)])
        self.assertEqual(response, {'ok': True, 'text': 'bonjour'})

    def test_large_transcript_response_is_not_truncated(self):
        text = 'word ' * 250000
        request = json.dumps({
            'verb': 'transcribe_file', 'path': '/tmp/audio.wav', 'clean': False,
        }).encode() + b'\n'
        response = self._exchange(request, lambda *_: (True, text))
        self.assertTrue(response['ok'])
        self.assertEqual(response['text'], text)

    def test_callback_rejection_and_malformed_request_are_structured(self):
        request = json.dumps({
            'verb': 'transcribe_file', 'path': '/tmp/audio.wav', 'clean': False,
        }).encode() + b'\n'
        response = self._exchange(request, lambda *_: (False, 'busy recording'))
        self.assertEqual(response, {'ok': False, 'error': 'busy recording'})

        response = self._exchange(b'{not-json}\n')
        self.assertFalse(response['ok'])
        self.assertIn('invalid request', response['error'])

    def test_oversized_request_is_rejected(self):
        response = self._exchange(b'{' + b'x' * 65536 + b'\n')
        self.assertFalse(response['ok'])
        self.assertIn('too large', response['error'])

    def test_json_response_clears_request_timeout_before_send(self):
        conn = mock.Mock()
        stop_event = threading.Event()
        server = RecordingControlServer(
            '/tmp/unused-fifo', '/tmp/unused-socket', lambda *_: None,
            lambda: False, on_file_transcribe=lambda *_: (True, 'text'),
        )
        server._stop_event = stop_event
        server._handle_json_request(conn, json.dumps({
            'verb': 'transcribe_file', 'path': '/tmp/audio.wav', 'clean': False,
        }), stop_event)
        conn.settimeout.assert_called_once_with(None)
        conn.sendall.assert_called_once()


if __name__ == '__main__':
    unittest.main()
