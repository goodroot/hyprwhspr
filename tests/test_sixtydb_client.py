import json
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))
sys.modules.setdefault("websocket", types.SimpleNamespace(WebSocketApp=object))

from sixtydb_realtime_client import SixtyDbRealtimeClient


class FakeWebSocket:
    def __init__(self):
        self.sent = []

    def send(self, payload):
        self.sent.append(json.loads(payload))


class SixtyDbRealtimeClientTests(unittest.TestCase):
    def _client_with_ws(self):
        client = SixtyDbRealtimeClient()
        client.connected = True
        client._socket_open = True
        client.ws = FakeWebSocket()
        return client

    def test_auth_url_appends_api_key_query_param(self):
        client = SixtyDbRealtimeClient()
        client.url = "wss://api.60db.ai/ws/stt"
        client.api_key = "sk_live_abc"
        self.assertEqual(client._auth_url(), "wss://api.60db.ai/ws/stt?apiKey=sk_live_abc")

    def test_start_message_carries_language_and_config(self):
        client = self._client_with_ws()
        client.language = "en"
        client.diarize = True
        client.utterance_end_ms = 700

        client._send_start()

        start = client.ws.sent[-1]
        self.assertEqual(start["type"], "start")
        self.assertEqual(start["languages"], ["en"])
        self.assertEqual(start["config"]["encoding"], "linear")
        self.assertEqual(start["config"]["sample_rate"], 16000)
        self.assertEqual(start["config"]["utterance_end_ms"], 700)
        self.assertTrue(start["config"]["diarize"])

    def test_start_message_omits_languages_for_auto_detect(self):
        client = self._client_with_ws()
        client.language = None

        client._send_start()

        self.assertNotIn("languages", client.ws.sent[-1])

    def test_interim_is_preview_only_and_final_commits(self):
        previews = []
        client = self._client_with_ws()
        client.set_partial_transcript_callback(previews.append)

        client._handle_event({"type": "transcription", "text": "hello", "is_final": False, "speech_final": False})
        client._handle_event({"type": "transcription", "text": "hello world", "is_final": True, "speech_final": True})

        self.assertEqual(previews, ["hello", ""])
        self.assertEqual(client.commit_and_get_text(timeout=0.1), "hello world")

    def test_fast_prefinal_is_not_committed(self):
        # is_final True but speech_final False = fast pre-refinement text -> preview only.
        previews = []
        client = self._client_with_ws()
        client.set_partial_transcript_callback(previews.append)

        client._handle_event({"type": "transcription", "text": "draft text", "is_final": True, "speech_final": False})

        self.assertEqual(client._partial_transcript, "draft text")
        self.assertEqual(client._committed_segments, [])
        self.assertEqual(previews, ["draft text"])

    def test_empty_final_is_skipped(self):
        client = self._client_with_ws()
        client._handle_event({"type": "transcription", "text": "", "is_final": True, "speech_final": True})
        self.assertEqual(client._committed_segments, [])

    def test_multiple_finals_are_stitched_in_order(self):
        client = self._client_with_ws()
        client._handle_event({"type": "transcription", "text": "first.", "is_final": True, "speech_final": True})
        client._handle_event({"type": "transcription", "text": "second.", "is_final": True, "speech_final": True})
        self.assertEqual(client.commit_and_get_text(timeout=0.1), "first. second.")

    def test_speech_started_clears_stale_partial(self):
        previews = []
        client = self._client_with_ws()
        client.set_partial_transcript_callback(previews.append)

        client._handle_event({"type": "transcription", "text": "stale", "is_final": False, "speech_final": False})
        client._handle_event({"type": "speech_started"})

        self.assertEqual(client._partial_transcript, "")
        self.assertEqual(previews[-1], "")

    def test_commit_sends_stop(self):
        client = self._client_with_ws()
        # No committed text yet and no queued audio -> falls through to sending stop.
        client._handle_event({"type": "transcription", "text": "done", "is_final": True, "speech_final": True})
        # Mark there is "new audio" so the fast path is skipped and stop is sent.
        client._audio_activity_id = 5
        client._last_transcript_audio_activity_id = 4
        client.commit_and_get_text(timeout=0.1)
        self.assertIn({"type": "stop"}, client.ws.sent)

    def test_clear_audio_buffer_resets_state(self):
        previews = []
        client = self._client_with_ws()
        client.set_partial_transcript_callback(previews.append)
        client._handle_event({"type": "transcription", "text": "x", "is_final": True, "speech_final": True})

        client.clear_audio_buffer()

        self.assertEqual(client._committed_segments, [])
        self.assertEqual(client._partial_transcript, "")
        self.assertEqual(client._transcript_generation, 0)
        self.assertEqual(previews[-1], "")

    def test_provider_registry_has_60db_realtime_model(self):
        from provider_registry import get_provider
        provider = get_provider("60db")
        self.assertIsNotNone(provider)
        self.assertEqual(provider["websocket_endpoint"], "wss://api.60db.ai/ws/stt")
        realtime = [m for m, d in provider["models"].items() if d.get("realtime_model")]
        self.assertTrue(realtime)


if __name__ == "__main__":
    unittest.main()
