import json
import sys
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))
sys.modules.setdefault("websocket", types.SimpleNamespace(WebSocketApp=object))

from realtime_client import RealtimeClient


class FakeWebSocket:
    def __init__(self):
        self.sent = []

    def send(self, payload):
        self.sent.append(json.loads(payload))


class RealtimeClientTests(unittest.TestCase):
    def _client_with_ws(self, model="gpt-realtime-whisper"):
        client = RealtimeClient(mode="transcribe")
        client.connected = True
        client.ws = FakeWebSocket()
        client.model = model
        return client

    def test_gpt_realtime_whisper_session_payload(self):
        client = self._client_with_ws()
        client.language = "en"
        client.set_transcription_delay("minimal")
        client.ws.sent.clear()

        client._send_session_update()

        payload = client.ws.sent[-1]
        session = payload["session"]
        audio_input = session["audio"]["input"]

        self.assertEqual(payload["type"], "session.update")
        self.assertEqual(session["type"], "transcription")
        self.assertEqual(audio_input["format"], {"type": "audio/pcm", "rate": 24000})
        self.assertEqual(audio_input["turn_detection"], None)
        self.assertEqual(
            audio_input["transcription"],
            {
                "model": "gpt-realtime-whisper",
                "language": "en",
                "delay": "minimal",
            },
        )

    def test_non_whisper_transcription_session_keeps_vad_and_configured_model(self):
        client = self._client_with_ws("gpt-4o-mini-transcribe")
        client.language = "fr"

        client._send_session_update()

        audio_input = client.ws.sent[-1]["session"]["audio"]["input"]
        self.assertEqual(audio_input["transcription"], {"model": "gpt-4o-mini-transcribe", "language": "fr"})
        self.assertEqual(audio_input["turn_detection"]["type"], "server_vad")
        self.assertNotIn("delay", audio_input["transcription"])

    def test_invalid_delay_falls_back_to_low(self):
        client = self._client_with_ws()
        client.set_transcription_delay("fastest")
        client.ws.sent.clear()

        client._send_session_update()

        transcription = client.ws.sent[-1]["session"]["audio"]["input"]["transcription"]
        self.assertEqual(transcription["delay"], "low")

    def test_delta_updates_preview_and_completed_is_final_text(self):
        previews = []
        client = self._client_with_ws()
        client.set_partial_transcript_callback(previews.append)

        client._handle_event({"type": "conversation.item.input_audio_transcription.delta", "delta": "hello"})
        client._handle_event({"type": "conversation.item.input_audio_transcription.delta", "delta": " wor"})
        client._handle_event({"type": "conversation.item.input_audio_transcription.completed", "transcript": "hello world"})

        self.assertEqual(previews, ["hello", "hello wor", ""])
        self.assertEqual(client.commit_and_get_text(timeout=0.1), "hello world")

    def test_completed_without_transcript_uses_accumulated_delta_text(self):
        previews = []
        client = self._client_with_ws()
        client.set_partial_transcript_callback(previews.append)

        client._handle_event({"type": "conversation.item.input_audio_transcription.delta", "delta": "delta"})
        client._handle_event({"type": "conversation.item.input_audio_transcription.delta", "delta": " only"})
        client._handle_event({"type": "conversation.item.input_audio_transcription.completed"})

        self.assertEqual(previews, ["delta", "delta only", ""])
        self.assertEqual(client.commit_and_get_text(timeout=0.1), "delta only")

    def test_unicode_delta_text_is_preserved(self):
        previews = []
        client = self._client_with_ws()
        client.set_partial_transcript_callback(previews.append)

        client._handle_event({"type": "conversation.item.input_audio_transcription.delta", "delta": "cafe "})
        client._handle_event({"type": "conversation.item.input_audio_transcription.delta", "delta": "東京"})
        client._handle_event({"type": "conversation.item.input_audio_transcription.completed"})

        self.assertEqual(previews, ["cafe ", "cafe 東京", ""])
        self.assertEqual(client.commit_and_get_text(timeout=0.1), "cafe 東京")

    def test_partial_preview_preserves_trailing_spaces(self):
        previews = []
        client = self._client_with_ws()
        client.set_partial_transcript_callback(previews.append)

        client._handle_event({"type": "conversation.item.input_audio_transcription.delta", "delta": "hello "})

        self.assertEqual(previews, ["hello "])

    def test_speech_started_clears_stale_partial(self):
        previews = []
        client = self._client_with_ws()
        client.set_partial_transcript_callback(previews.append)

        client._handle_event({"type": "conversation.item.input_audio_transcription.delta", "delta": "first segment"})
        client._handle_event({"type": "input_audio_buffer.speech_started"})
        client._handle_event({"type": "conversation.item.input_audio_transcription.delta", "delta": "next"})

        self.assertEqual(client._partial_transcript, "next")
        self.assertEqual(previews, ["first segment", "", "next"])

    def test_clear_audio_buffer_clears_stale_partial(self):
        previews = []
        client = self._client_with_ws()
        client.set_partial_transcript_callback(previews.append)
        client._handle_event({"type": "conversation.item.input_audio_transcription.delta", "delta": "stale"})

        client.clear_audio_buffer()

        self.assertEqual(client._partial_transcript, "")
        self.assertEqual(previews[-1], "")
        self.assertEqual(client.ws.sent[-1]["type"], "input_audio_buffer.clear")

    def test_schema_declares_realtime_transcription_delay_values(self):
        schema = json.loads((ROOT / "share" / "config.schema.json").read_text())
        delay_schema = schema["properties"]["realtime_transcription_delay"]

        self.assertEqual(delay_schema["default"], "low")
        self.assertEqual(delay_schema["enum"], ["minimal", "low", "medium", "high", "xhigh"])

    def test_append_audio_uses_configured_input_sample_rate_for_duration(self):
        client = self._client_with_ws()
        client.set_input_sample_rate(48000)
        client.max_buffer_seconds = 2.0

        client.append_audio(np.zeros(4800, dtype=np.float32))

        self.assertAlmostEqual(client.audio_buffer_seconds, 0.1)

    def test_resample_for_output_handles_48khz_capture_to_24khz_provider_rate(self):
        client = self._client_with_ws()
        client.set_input_sample_rate(48000)
        client.sample_rate = 24000
        audio = np.zeros(4800, dtype=np.float32)

        resampled = client._resample_for_output(audio)

        self.assertEqual(len(resampled), 2400)
        self.assertEqual(resampled.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
