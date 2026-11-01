import json
import os
import socket
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib' / 'src'))
from voice_activity import VoiceActivity, _Session, default_socket_path


class FakeSocket:
    def __init__(self, response=None, connect_error=None):
        self.requests = []
        self.response = response
        self.connect_error = connect_error
        self.closed = False
        self.connected = None

    def settimeout(self, value):
        self.timeout = value

    def connect(self, path):
        if self.connect_error:
            raise self.connect_error
        self.connected = path

    def sendall(self, data):
        self.requests.append(json.loads(data))

    def recv(self, size):
        if self.response is not None:
            return self.response
        return (json.dumps({
            'v': 1, 'request_id': self.requests[-1]['request_id'],
            'ok': True, 'type': 'input', 'active': True,
        }) + '\n').encode()

    def shutdown(self, how):
        pass

    def close(self):
        self.closed = True


class VoiceActivityTests(unittest.TestCase):
    def test_default_socket_resolution_and_explicit_path(self):
        with mock.patch.dict(os.environ, {'VOXD_SOCKET': '/test/override'}, clear=True):
            self.assertEqual(default_socket_path(), '/test/override')
            self.assertEqual(VoiceActivity('/test/explicit').path, '/test/explicit')
        with mock.patch.dict(os.environ, {'XDG_RUNTIME_DIR': '/test/run'}, clear=True):
            self.assertEqual(default_socket_path(), '/test/run/voxd.sock')
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch('voice_activity.os.getuid', return_value=42):
            self.assertEqual(default_socket_path(), '/tmp/voxd-42.sock')

    def test_acknowledged_start_and_renew_use_same_connection_then_eof(self):
        sock = FakeSocket()
        with mock.patch('voice_activity.socket.socket', return_value=sock), mock.patch('voice_activity.threading.Thread'):
            activity = VoiceActivity('/test/socket')
            session = activity.begin()
            self.assertEqual(sock.connected, '/test/socket')
            self.assertEqual(sock.requests[0]['op'], 'input_begin')
            session._renew()
            self.assertEqual(len(sock.requests), 2)
            self.assertNotEqual(sock.requests[0]['request_id'], sock.requests[1]['request_id'])
            session.close()
            session.close()
            session._renew()
            self.assertTrue(sock.closed)
            self.assertEqual(len(sock.requests), 2)
            self.assertFalse(activity._sessions)

    def test_old_session_cleanup_does_not_release_new_take(self):
        first, second = FakeSocket(), FakeSocket()
        with mock.patch('voice_activity.socket.socket', side_effect=[first, second]), mock.patch('voice_activity.threading.Thread'):
            activity = VoiceActivity('/test/socket')
            a, b = activity.begin(), activity.begin()
            a.close()
            self.assertTrue(first.closed)
            self.assertFalse(second.closed)
            self.assertEqual(activity._sessions, {b})
            activity.close()
            self.assertTrue(second.closed)
            activity.close()
            # Shutdown is sticky and does not deadlock or open another socket.
            late = activity.begin()
            self.assertTrue(late._stop.is_set())

    def test_failed_connect_is_nonfatal_and_retries_with_fresh_socket(self):
        first = FakeSocket(connect_error=FileNotFoundError('missing daemon'))
        second = FakeSocket()
        with mock.patch('voice_activity.socket.socket', side_effect=[first, second]), mock.patch('voice_activity.threading.Thread'), mock.patch('builtins.print') as log:
            activity = VoiceActivity('/test/socket')
            session = activity.begin()
            self.assertTrue(first.closed)
            log.assert_called_once()
            session._renew()
            self.assertEqual(second.requests[0]['op'], 'input_begin')
            self.assertFalse(session._warned)
            activity.close()

    def test_bad_or_old_daemon_replies_close_the_connection(self):
        for response in [b'{}\n', b'garbage\n', b'[]\n', b'', b'x' * 4096,
                         b'{"v":1,"request_id":"other","ok":true,"type":"input","active":true}\n']:
            with self.subTest(response=response[:40]):
                sock = FakeSocket(response=response)
                with mock.patch('voice_activity.socket.socket', return_value=sock), mock.patch('builtins.print'):
                    session = _Session('/test/socket', lambda _: None)
                    session._renew()
                    self.assertTrue(sock.closed)
                    self.assertIsNone(session._socket)

    def test_silent_peer_timeout_releases_lease(self):
        sock = FakeSocket()
        sock.recv = mock.Mock(side_effect=socket.timeout('no response'))
        with mock.patch('voice_activity.socket.socket', return_value=sock), mock.patch('builtins.print'):
            session = _Session('/test/socket', lambda _: None)
            session._renew()
            self.assertTrue(sock.closed)
            self.assertLessEqual(sock.timeout, .5)

    def test_stopped_or_replaced_capture_does_not_keep_renewing(self):
        active = True
        sock = FakeSocket()
        with mock.patch('voice_activity.socket.socket', return_value=sock), mock.patch('voice_activity.threading.Thread'):
            activity = VoiceActivity('/test/socket')
            session = activity.begin(active=lambda: active)
            active = False
            session._renew()
            self.assertTrue(sock.closed)
            self.assertTrue(session._stop.is_set())
            self.assertFalse(activity._sessions)

    def test_close_during_start_cannot_reconnect(self):
        sock = FakeSocket()
        with mock.patch('voice_activity.socket.socket', return_value=sock):
            activity = VoiceActivity('/test/socket')
            activity.close()
            session = activity.begin()
            session._renew()
            self.assertFalse(sock.requests)
            self.assertIsNone(sock.connected)


if __name__ == '__main__':
    unittest.main()
