"""Recovery from a dead realtime connection (regressions for issue #229).

Two independent paths used to leave realtime permanently unusable until the
service restarted:

- close() latched ``_closed`` and ``_connect_internal()`` refused while latched,
  so the backend's on-demand reconnect (which tears down stale state first)
  could never succeed;
- a recording failure ran ``close_realtime_connection()``, which drops the
  client entirely, and nothing outside the suspend/resume path rebuilt it.

Both are driven here against the real classes over a fake transport.
"""

import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))
sys.modules.setdefault("websocket", types.SimpleNamespace(WebSocketApp=object))

from backends.realtime_ws_backend import RealtimeWsBackend  # noqa: E402
from realtime_client import RealtimeClient  # noqa: E402
from whisper_manager import WhisperManager  # noqa: E402


class FakeWebSocketApp:
    """Transport whose run_forever completes the handshake immediately."""

    def __init__(self, url, on_open=None, on_message=None, on_error=None, on_close=None, **kwargs):
        self.url = url
        self.on_open = on_open
        self.on_close = on_close
        self._stop = threading.Event()

    def run_forever(self):
        self.on_open(self)
        self._stop.wait(5)

    def close(self):
        self._stop.set()

    def send(self, payload):
        pass


class DeadTransport:
    """Transport whose socket is refused, so connect() fails immediately.

    Refusing at construction rather than hanging keeps these tests off the
    client's 10s connect timeout.
    """

    class WebSocketApp:
        def __init__(self, url, **kwargs):
            raise ConnectionRefusedError("endpoint down")


def _client():
    """A RealtimeClient wired to the fake transport, with I/O stubbed out."""
    client = RealtimeClient(mode="transcribe")
    client._websocket_transport = types.SimpleNamespace(WebSocketApp=FakeWebSocketApp)
    client._send_session_update = lambda *a, **k: None
    return client


def _connected_client():
    client = _client()
    assert client.connect("wss://example.test/rt", "key", "model")
    return client


def _idle_close(client):
    """Simulate the server dropping an idle session."""
    client._last_audio_chunk_time = 0.0
    client._on_close(client.ws, None, "", client._active_generation)


class ClientReopenTests(unittest.TestCase):
    """close() must not make a client permanently unusable (issue #229)."""

    def test_connect_reopens_a_closed_client(self):
        client = _connected_client()
        _idle_close(client)
        self.assertFalse(client.connected)

        # What the backend's on-demand reconnect used to do.
        client.close()
        self.assertTrue(client._closed)

        self.assertTrue(client.connect("wss://example.test/rt", "key", "model"))
        self.assertTrue(client.connected)
        self.assertFalse(client._closed)

    def test_reset_leaves_client_reconnectable(self):
        client = _connected_client()
        client.reset()

        self.assertFalse(client._closed)
        self.assertFalse(client._stop_event.is_set())
        self.assertFalse(client.connected)
        self.assertTrue(client._connect_internal())
        self.assertTrue(client.connected)

    def test_close_still_latches_against_internal_reconnect(self):
        """Real shutdown must still stand background reconnect loops down."""
        client = _connected_client()
        client.close()

        self.assertTrue(client._closed)
        self.assertTrue(client._stop_event.is_set())
        self.assertFalse(client._connect_internal())
        self.assertFalse(client.connected)

    def test_attempt_reconnect_aborts_after_close(self):
        client = _connected_client()
        client.reconnect_delays = [0]
        client.close()

        self.assertFalse(client._attempt_reconnect())

    def test_teardown_clears_stuck_connecting_flag(self):
        """An abandoned attempt must not leave `connecting` latched True.

        _connect_internal only clears the flag when its generation is still
        current, and teardown bumps the generation — so without an explicit
        clear the backend would report "still connecting" forever.
        """
        client = _client()
        client._websocket_transport = DeadTransport()
        client._connect_internal()  # fails, leaving attempt state behind
        client.connecting = True

        client.reset()
        self.assertFalse(client.connecting)


class FakeConfig:
    def __init__(self, backend="realtime-ws"):
        self._backend = backend

    def get_setting(self, key, default=None):
        if key == "transcription_backend":
            return self._backend
        return default

    def get_temp_directory(self):
        return "/tmp"


def _backend(config=None):
    manager = WhisperManager(config_manager=config or FakeConfig())
    return RealtimeWsBackend(manager)


class EnsureClientTests(unittest.TestCase):
    """A torn-down client is rebuilt on the next recording, not left dead."""

    def test_streaming_callback_rebuilds_destroyed_client(self):
        backend = _backend()
        sentinel = object()

        def fake_initialize():
            backend._realtime_client = _connected_client()
            backend._realtime_streaming_callback = sentinel
            return True

        with mock.patch.object(backend, "initialize", side_effect=fake_initialize) as init:
            self.assertIs(backend.get_streaming_callback(), sentinel)

        self.assertEqual(init.call_count, 1)
        self.assertIsNone(backend.last_connect_failure)

    def test_rebuild_is_not_retried_within_cooldown(self):
        backend = _backend()

        with mock.patch.object(backend, "initialize", return_value=False) as init:
            self.assertIsNone(backend.get_streaming_callback())
            self.assertEqual(backend.last_connect_failure, "failed")

            self.assertIsNone(backend.get_streaming_callback())
            self.assertEqual(init.call_count, 1)
            self.assertEqual(backend.last_connect_failure, "cooldown")

    def test_rebuild_retried_after_cooldown_expires(self):
        backend = _backend()

        with mock.patch.object(backend, "initialize", return_value=False) as init:
            self.assertIsNone(backend.get_streaming_callback())
            backend._last_rebuild_attempt = time.monotonic() - backend.REBUILD_COOLDOWN_SECS - 1
            self.assertIsNone(backend.get_streaming_callback())

        self.assertEqual(init.call_count, 2)

    def test_connected_client_is_used_as_is(self):
        backend = _backend()
        backend._realtime_client = _connected_client()
        backend._realtime_streaming_callback = object()

        with mock.patch.object(backend, "initialize") as init:
            self.assertIs(
                backend.get_streaming_callback(), backend._realtime_streaming_callback
            )
        init.assert_not_called()

    def test_non_realtime_backend_returns_none_without_rebuilding(self):
        backend = _backend(config=FakeConfig(backend="pywhispercpp"))

        with mock.patch.object(backend, "initialize") as init:
            self.assertIsNone(backend.get_streaming_callback())
        init.assert_not_called()


class OnDemandReconnectTests(unittest.TestCase):
    """The on-demand path and the background loop must agree about recovery."""

    def _backend_with_idle_closed_client(self):
        backend = _backend()
        client = _connected_client()
        backend._realtime_client = client
        backend._realtime_streaming_callback = object()
        backend._realtime_connect_params = {
            "websocket_url": "wss://example.test/rt",
            "api_key": "key",
            "model_id": "model",
            "instructions": None,
        }
        _idle_close(client)
        return backend, client

    def test_on_demand_reconnect_recovers_idle_close(self):
        backend, client = self._backend_with_idle_closed_client()

        self.assertIs(
            backend.get_streaming_callback(), backend._realtime_streaming_callback
        )
        self.assertTrue(client.connected)
        self.assertIsNone(backend.last_connect_failure)

    def test_background_reconnect_recovers_same_state(self):
        """Same starting state, the other path: both must end connected."""
        _, client = self._backend_with_idle_closed_client()
        client.reconnect_delays = [0]
        client.receiver_running = True

        self.assertTrue(client._attempt_reconnect())
        self.assertTrue(client.connected)

    def test_reconnect_failure_is_reported_as_failed(self):
        backend, client = self._backend_with_idle_closed_client()
        client._websocket_transport = DeadTransport()

        self.assertIsNone(backend.get_streaming_callback())
        self.assertEqual(backend.last_connect_failure, "failed")

    def test_in_flight_handshake_is_not_torn_down(self):
        backend, client = self._backend_with_idle_closed_client()
        client.connecting = True

        with mock.patch.object(client, "connect") as connect:
            self.assertIsNone(backend.get_streaming_callback())

        connect.assert_not_called()
        self.assertEqual(backend.last_connect_failure, "connecting")


if __name__ == "__main__":
    unittest.main()
