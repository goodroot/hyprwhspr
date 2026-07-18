import json
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))
sys.modules.setdefault("websocket", types.SimpleNamespace(WebSocketApp=object))

from realtime_client import RealtimeClient  # noqa: E402
from whisper_manager import WhisperManager  # noqa: E402


class FakeWebSocket:
    def __init__(self):
        self.sent = []

    def send(self, payload):
        self.sent.append(json.loads(payload))


class FakeConfig:
    def get_setting(self, key, default=None):
        return default

    def get_temp_directory(self):
        return "/tmp"


class FakeRealtimeBackend:
    name = 'realtime-ws'

    def __init__(self, loaded=True):
        self.is_loaded = loaded
        self.discarded = 0
        self.closed = 0

    def discard_audio(self):
        self.discarded += 1

    def close(self):
        self.closed += 1


class StaleTranscriptGuardTests(unittest.TestCase):
    """A transcript for a cancelled/previous take must not leak into the next one."""

    def _connected_client(self):
        client = RealtimeClient(mode="transcribe")
        client.connected = True
        client.ws = FakeWebSocket()
        return client

    def test_current_take_transcript_is_kept(self):
        client = self._connected_client()
        client._handle_event({'type': 'input_audio_buffer.committed', 'item_id': 'item_a'})
        client._handle_event({
            'type': 'conversation.item.input_audio_transcription.completed',
            'item_id': 'item_a', 'transcript': 'hello world',
        })
        self.assertEqual(client._committed_segments, ['hello world'])

    def test_retired_item_transcript_is_dropped(self):
        client = self._connected_client()
        client._handle_event({'type': 'input_audio_buffer.committed', 'item_id': 'item_a'})

        # Cancel discards the take; next recording start clears again
        client.clear_audio_buffer()

        client._handle_event({
            'type': 'conversation.item.input_audio_transcription.completed',
            'item_id': 'item_a', 'transcript': 'delete everything',
        })
        self.assertEqual(client._committed_segments, [])
        self.assertEqual(client._transcript_generation, 0)

    def test_retired_item_delta_does_not_reach_preview(self):
        previews = []
        client = self._connected_client()
        client.set_partial_transcript_callback(previews.append)
        client._handle_event({'type': 'input_audio_buffer.speech_started', 'item_id': 'item_a'})
        client.clear_audio_buffer()
        previews.clear()

        client._handle_event({
            'type': 'conversation.item.input_audio_transcription.delta',
            'item_id': 'item_a', 'delta': 'stale',
        })
        self.assertEqual(previews, [])
        self.assertEqual(client._partial_transcript, "")

    def test_new_take_after_retirement_is_kept(self):
        client = self._connected_client()
        client._handle_event({'type': 'input_audio_buffer.committed', 'item_id': 'item_a'})
        client.clear_audio_buffer()
        client._handle_event({'type': 'input_audio_buffer.committed', 'item_id': 'item_b'})
        client._handle_event({
            'type': 'conversation.item.input_audio_transcription.completed',
            'item_id': 'item_b', 'transcript': 'fresh take',
        })
        self.assertEqual(client._committed_segments, ['fresh take'])

    def test_retirement_happens_even_when_disconnected(self):
        client = self._connected_client()
        client._handle_event({'type': 'input_audio_buffer.committed', 'item_id': 'item_a'})
        client.connected = False

        client.clear_audio_buffer()

        self.assertIn('item_a', client._retired_item_ids)
        self.assertEqual(client._session_item_ids, set())


class CancelRecoveryManagerTests(unittest.TestCase):
    """Cancel keeps the connection; a destroyed client is reported for gate-driven rebuild."""

    def _manager(self, backend):
        manager = WhisperManager(config_manager=FakeConfig())
        manager._backend = backend
        return manager

    def test_discard_keeps_connection(self):
        backend = FakeRealtimeBackend(loaded=True)
        manager = self._manager(backend)
        manager.discard_realtime_audio()
        self.assertEqual(backend.discarded, 1)
        self.assertEqual(backend.closed, 0)

    def test_discard_noops_without_client(self):
        backend = FakeRealtimeBackend(loaded=False)
        manager = self._manager(backend)
        manager.discard_realtime_audio()
        self.assertEqual(backend.discarded, 0)

    def test_client_missing_only_for_unloaded_realtime_backend(self):
        self.assertTrue(self._manager(FakeRealtimeBackend(loaded=False)).realtime_client_missing())
        self.assertFalse(self._manager(FakeRealtimeBackend(loaded=True)).realtime_client_missing())
        self.assertFalse(self._manager(None).realtime_client_missing())

        local = FakeRealtimeBackend(loaded=False)
        local.name = 'pywhispercpp'
        self.assertFalse(self._manager(local).realtime_client_missing())


if __name__ == '__main__':
    unittest.main()
