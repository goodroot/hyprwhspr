import sys
import json
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))
sys.modules.setdefault("websocket", types.SimpleNamespace(WebSocketApp=object))

from realtime_client import RealtimeClient  # noqa: E402


class ReconnectPolicyTests(unittest.TestCase):
    def _client(self, connected=True, receiver_running=True, last_chunk_age=None):
        client = RealtimeClient(mode="transcribe")
        client.connected = connected
        client.receiver_running = receiver_running
        if last_chunk_age is not None:
            client._last_audio_chunk_time = time.time() - last_chunk_age
        return client

    def _close_and_join(self, client, code=1006):
        """Fire _on_close and wait for any spawned reconnect thread."""
        attempted = threading.Event()
        client._attempt_reconnect = attempted.set
        client._on_close(None, code, '')
        attempted.wait(timeout=1)
        return attempted.is_set()

    def test_idle_close_suppresses_reconnect(self):
        client = self._client(last_chunk_age=30)
        self.assertFalse(self._close_and_join(client))

    def test_zero_chunk_close_treated_as_idle(self):
        client = self._client()  # _last_audio_chunk_time == 0.0
        self.assertFalse(self._close_and_join(client))

    def test_mid_recording_close_reconnects(self):
        client = self._client(last_chunk_age=1)
        self.assertTrue(self._close_and_join(client))

    def test_normal_close_1000_never_reconnects(self):
        client = self._client(last_chunk_age=1)
        self.assertFalse(self._close_and_join(client, code=1000))

    def test_close_before_connect_never_reconnects(self):
        client = self._client(connected=False, last_chunk_age=1)
        self.assertFalse(self._close_and_join(client))

    def test_obsolete_socket_close_does_not_touch_active_connection(self):
        client = self._client(last_chunk_age=1)
        active_ws = object()
        client.ws = active_ws
        client._sender_running = True
        client._audio_queue.append(b'audio')

        client._on_close(object(), 1006, '')

        self.assertTrue(client.connected)
        self.assertTrue(client._sender_running)
        self.assertEqual(list(client._audio_queue), [b'audio'])
        self.assertIs(client.ws, active_ws)

    def test_obsolete_socket_open_does_not_mark_client_connected(self):
        client = self._client(connected=False, receiver_running=False)
        client.ws = object()

        client._on_open(object())

        self.assertFalse(client.connected)
        self.assertFalse(client.receiver_running)

    def test_obsolete_socket_message_is_not_queued(self):
        client = self._client()
        client.ws = object()

        client._on_message(object(), json.dumps({'type': 'stale'}))

        self.assertTrue(client.event_queue.empty())

    def test_reconnect_loop_is_bounded_and_guarded(self):
        client = self._client()
        calls = []
        client._connect_internal = lambda: calls.append(1) is None and False

        with mock.patch.object(client._stop_event, 'wait', return_value=False):
            self.assertFalse(client._attempt_reconnect())

        self.assertEqual(len(calls), client.max_reconnect_attempts)

        # A concurrent attempt returns immediately while the lock is held
        client.reconnect_attempts = 0
        with client._reconnect_lock:
            self.assertFalse(client._attempt_reconnect())

    def test_reconnect_stops_at_first_success(self):
        client = self._client()
        results = iter([False, True])
        calls = []
        client._connect_internal = lambda: calls.append(1) is None and next(results)

        with mock.patch.object(client._stop_event, 'wait', return_value=False):
            self.assertTrue(client._attempt_reconnect())

        self.assertEqual(len(calls), 2)

    def test_close_during_backoff_aborts(self):
        client = self._client()
        client._connect_internal = lambda: self.fail("must not connect after close")

        def wait_then_close(_delay):
            client.close()
            return True

        with mock.patch.object(client._stop_event, 'wait', side_effect=wait_then_close):
            self.assertFalse(client._attempt_reconnect())

    def test_late_open_after_close_is_ignored(self):
        client = self._client(connected=False, receiver_running=False)
        socket = object()
        client.ws = socket
        generation = client._active_generation

        client.close()
        client._on_open(socket, generation)

        self.assertFalse(client.connected)
        self.assertFalse(client.receiver_running)

    def test_append_audio_updates_last_audio_chunk_time(self):
        client = self._client()
        client.ws = object()
        client.append_audio(np.zeros(160, dtype=np.float32))
        self.assertGreater(client._last_audio_chunk_time, 0.0)

    def test_reset_stream_state_clears_last_audio_chunk_time(self):
        client = self._client(last_chunk_age=1)
        with client.lock:
            client._reset_stream_state_locked()
        self.assertEqual(client._last_audio_chunk_time, 0.0)


if __name__ == '__main__':
    unittest.main()
