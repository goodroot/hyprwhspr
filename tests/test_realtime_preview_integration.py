import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))
sys.path.insert(0, str(ROOT / "lib"))
sys.modules.setdefault("websocket", types.SimpleNamespace(WebSocketApp=object))

from realtime_client import RealtimeClient
from whisper_manager import WhisperManager
import mic_osd.runner as runner_module


class FakeConfig:
    def __init__(self, values):
        self.values = values

    def get_setting(self, key, default=None):
        return self.values.get(key, default)


class FakeWebSocket:
    def __init__(self):
        self.sent = []

    def send(self, payload):
        self.sent.append(json.loads(payload))


class RealtimePreviewIntegrationTests(unittest.TestCase):
    def test_openai_realtime_whisper_delta_updates_preview_file(self):
        config = FakeConfig(
            {
                "transcription_backend": "realtime-ws",
                "websocket_provider": "openai",
                "websocket_model": "gpt-realtime-whisper",
                "realtime_mode": "transcribe",
                "mic_osd_enabled": True,
            }
        )
        manager = WhisperManager(config_manager=config)
        client = RealtimeClient(mode="transcribe")
        client.connected = True
        client.ws = FakeWebSocket()
        client.model = "gpt-realtime-whisper"
        manager._realtime_client = client

        with tempfile.TemporaryDirectory() as tmp:
            preview_file = Path(tmp) / "hyprwhspr" / "transcript_preview"
            original = runner_module.TRANSCRIPT_PREVIEW_FILE
            runner_module.TRANSCRIPT_PREVIEW_FILE = preview_file
            try:
                runner = runner_module.MicOSDRunner()
                manager.set_realtime_partial_callback(runner.set_preview_text)

                client._handle_event({"type": "conversation.item.input_audio_transcription.delta", "delta": "hello "})
                client._handle_event({"type": "conversation.item.input_audio_transcription.delta", "delta": "world"})
                runner._flush_pending_preview_text()

                self.assertEqual(preview_file.read_text(encoding="utf-8"), "hello world")
            finally:
                runner_module.TRANSCRIPT_PREVIEW_FILE = original

    def test_non_matching_realtime_config_clears_preview_callback(self):
        config = FakeConfig(
            {
                "transcription_backend": "realtime-ws",
                "websocket_provider": "openai",
                "websocket_model": "gpt-4o-mini-transcribe",
                "realtime_mode": "transcribe",
                "mic_osd_enabled": True,
            }
        )
        manager = WhisperManager(config_manager=config)
        client = RealtimeClient(mode="transcribe")
        manager._realtime_client = client

        callback = mock.Mock()
        manager.set_realtime_partial_callback(callback)

        self.assertIsNone(client.partial_transcript_callback)
        callback.assert_called_once_with("")


if __name__ == "__main__":
    unittest.main()
