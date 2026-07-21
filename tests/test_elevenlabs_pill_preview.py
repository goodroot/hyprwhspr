import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))
sys.path.insert(0, str(ROOT / "lib"))

from backends.realtime_ws_backend import RealtimeWsBackend
from elevenlabs_realtime_client import ElevenLabsRealtimeClient


class FakeConfig:
    def __init__(self, values=None):
        self.values = {
            "transcription_backend": "realtime-ws",
            "websocket_provider": "elevenlabs",
            "websocket_model": "scribe_v2_realtime",
            "realtime_mode": "transcribe",
            "mic_osd_enabled": True,
            "mic_osd_style": "pill",
            "mic_osd_pill_transcript_enabled": True,
        }
        self.values.update(values or {})

    def get_setting(self, key, default=None):
        return self.values.get(key, default)


class FakeManager:
    def __init__(self, config=None):
        self.config = config or FakeConfig()
        self.temp_dir = None
        self.ready = True
        self.current_model = None
        self._last_use_time = 0.0
        self._realtime_partial_callback = None


class FakeRealtimeClient:
    def __init__(self):
        self.connected = True
        self.callback = None
        self.callback_updates = []

    def set_partial_transcript_callback(self, callback):
        self.callback = callback
        self.callback_updates.append(callback)

    def clear_audio_buffer(self):
        if self.callback:
            self.callback("")


class ElevenLabsClientPreviewTests(unittest.TestCase):
    def setUp(self):
        self.client = ElevenLabsRealtimeClient()

    def test_combines_committed_and_partial_text(self):
        previews = []
        self.client.set_partial_transcript_callback(previews.append)
        with self.client.lock:
            self.client._committed_segments = ["This is already"]
            self.client._partial_transcript = "committed and live"

        self.client._emit_partial_transcript()

        self.assertEqual(previews[-1], "This is already committed and live")

    def test_callback_survives_connection_replacement(self):
        previews = []
        self.client.set_partial_transcript_callback(previews.append)
        self.client._connection = object()
        self.client._connection = object()
        with self.client.lock:
            self.client._partial_transcript = "after reconnect"

        self.client._emit_partial_transcript()

        self.assertEqual(previews[-1], "after reconnect")

    def test_clear_audio_buffer_clears_preview(self):
        previews = []
        self.client.set_partial_transcript_callback(previews.append)
        with self.client.lock:
            self.client._partial_transcript = "stale"

        self.client.clear_audio_buffer()

        self.assertEqual(previews[-1], "")


class ElevenLabsRealtimePreviewTests(unittest.TestCase):
    def setUp(self):
        self.manager = FakeManager()
        self.backend = RealtimeWsBackend(self.manager)
        self.client = FakeRealtimeClient()
        self.backend._realtime_client = self.client

    def _apply(self):
        previews = []
        self.manager._realtime_partial_callback = previews.append
        self.backend.apply_partial_callback(previews.append)
        return previews

    def test_enabled_pill_registers_client_callback(self):
        previews = self._apply()

        self.client.callback("live words")

        self.assertEqual(previews, ["live words"])

    def test_callback_is_updated_without_duplicate_sdk_handlers(self):
        first = self._apply()
        second = []
        self.manager._realtime_partial_callback = second.append

        self.backend.apply_partial_callback(second.append)
        self.client.callback("latest")

        self.assertEqual(first, [])
        self.assertEqual(second, ["latest"])
        self.assertEqual(len(self.client.callback_updates), 2)

    def test_disabled_pill_preview_unregisters_callback_and_clears(self):
        self.manager.config.values["mic_osd_pill_transcript_enabled"] = False
        previews = self._apply()

        self.assertIsNone(self.client.callback)
        self.assertEqual(previews, [""])

    def test_pill_preview_is_disabled_when_setting_is_absent(self):
        self.manager.config.values.pop("mic_osd_pill_transcript_enabled")
        previews = self._apply()

        self.assertIsNone(self.client.callback)
        self.assertEqual(previews, [""])

    def test_non_pill_style_does_not_enable_elevenlabs_preview(self):
        self.manager.config.values["mic_osd_style"] = "waveform"
        previews = self._apply()

        self.assertIsNone(self.client.callback)
        self.assertEqual(previews, [""])

    def test_openai_preview_remains_enabled_for_waveform(self):
        self.manager.config.values.update({
            "websocket_provider": "openai",
            "websocket_model": "gpt-realtime-whisper",
            "mic_osd_style": "waveform",
        })

        self._apply()

        self.assertIsNotNone(self.client.callback)

    def test_openai_preview_is_not_shown_in_elevenlabs_only_pill(self):
        self.manager.config.values.update({
            "websocket_provider": "openai",
            "websocket_model": "gpt-realtime-whisper",
            "mic_osd_style": "pill",
        })
        previews = self._apply()

        self.assertIsNone(self.client.callback)
        self.assertEqual(previews, [""])


if __name__ == "__main__":
    unittest.main()
