import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))
sys.modules.setdefault("websocket", types.SimpleNamespace(WebSocketApp=object))

from realtime_client import RealtimeClient
import realtime_base


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

    @staticmethod
    def _response_created(request_id, response_id):
        return {
            "type": "response.created",
            "response": {
                "id": response_id,
                "metadata": {"hyprwhspr_request_id": request_id},
            },
        }

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

    def test_gpt_live_transcribe_session_payload(self):
        client = self._client_with_ws("gpt-live-transcribe")
        client.update_transcription_config(
            "en",
            "Linux and software-development dictation.",
        )
        client.set_transcription_delay("high")
        client.ws.sent.clear()

        client._send_session_update()

        self.assertEqual(
            client.ws.sent[-1],
            {
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "transcription": {
                                "model": "gpt-live-transcribe",
                                "languages": ["en"],
                                "delay": "high",
                                "prompt": "Linux and software-development dictation.",
                            },
                            "turn_detection": None,
                        }
                    },
                },
            },
        )

    def test_gpt_transcribe_session_payload(self):
        client = self._client_with_ws("gpt-transcribe")
        client.update_transcription_config(
            "en",
            "Linux and software-development dictation.",
        )
        client.set_transcription_delay("xhigh")
        client.ws.sent.clear()

        client._send_session_update()

        self.assertEqual(
            client.ws.sent[-1],
            {
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "transcription": {
                                "model": "gpt-transcribe",
                                "languages": ["en"],
                                "prompt": "Linux and software-development dictation.",
                            },
                            "turn_detection": None,
                        }
                    },
                },
            },
        )

    def test_gpt_transcribe_omits_unconfigured_context(self):
        client = self._client_with_ws("gpt-transcribe")

        client._send_session_update()

        transcription = client.ws.sent[-1]["session"]["audio"]["input"]["transcription"]
        self.assertEqual(transcription, {"model": "gpt-transcribe"})
        self.assertNotIn("language", transcription)
        self.assertNotIn("languages", transcription)
        self.assertNotIn("prompt", transcription)
        self.assertNotIn("keywords", transcription)
        self.assertNotIn("delay", transcription)

    def test_gpt_live_transcribe_omits_unset_prompt(self):
        client = self._client_with_ws("gpt-live-transcribe")

        client._send_session_update()

        transcription = client.ws.sent[-1]["session"]["audio"]["input"]["transcription"]
        self.assertNotIn("prompt", transcription)

    def test_gpt_live_transcribe_uses_languages_array(self):
        client = self._client_with_ws("gpt-live-transcribe")
        client.language = "fr"

        client._send_session_update()

        transcription = client.ws.sent[-1]["session"]["audio"]["input"]["transcription"]
        self.assertEqual(transcription["languages"], ["fr"])
        self.assertNotIn("language", transcription)

    def test_gpt_live_transcribe_propagates_delay(self):
        client = self._client_with_ws("gpt-live-transcribe")
        client.set_transcription_delay("xhigh")
        client.ws.sent.clear()

        client._send_session_update()

        transcription = client.ws.sent[-1]["session"]["audio"]["input"]["transcription"]
        self.assertEqual(transcription["delay"], "xhigh")

    def test_gpt_live_transcribe_uses_manual_turn_detection(self):
        client = self._client_with_ws("gpt-live-transcribe")

        client._send_session_update()

        audio_input = client.ws.sent[-1]["session"]["audio"]["input"]
        self.assertIsNone(audio_input["turn_detection"])

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

    def test_converse_session_history_keeps_completed_items(self):
        client = RealtimeClient(mode="converse")
        client.connected = True
        client.ws = FakeWebSocket()
        client.set_conversation_history("session")
        client._handle_event({"type": "input_audio_buffer.committed", "item_id": "input_1"})
        client._request_transcript({"buffer_was_committed": True})
        request_id = client.ws.sent[-1]["response"]["metadata"]["hyprwhspr_request_id"]
        client._handle_event(self._response_created(request_id, "response_1"))
        client._handle_event({
            "type": "response.done",
            "response": {
                "id": "response_1",
                "metadata": {"hyprwhspr_request_id": request_id},
                "output": [{"id": "output_1"}],
            },
        })

        self.assertEqual([event for event in client.ws.sent if event["type"] == "conversation.item.delete"], [])
        self.assertTrue(client.response_complete)

    def test_converse_defaults_to_deleting_completed_turn_history(self):
        client = RealtimeClient(mode="converse")

        self.assertEqual(client.conversation_history, "turn")

    def test_converse_turn_history_deletes_completed_input_and_outputs(self):
        client = RealtimeClient(mode="converse")
        client.connected = True
        client.ws = FakeWebSocket()
        client.set_conversation_history("turn")
        client._handle_event({"type": "input_audio_buffer.committed", "item_id": "input_1"})
        client._request_transcript({"buffer_was_committed": True})
        request_id = client.ws.sent[-1]["response"]["metadata"]["hyprwhspr_request_id"]
        client._handle_event(self._response_created(request_id, "response_1"))
        client._handle_event({
            "type": "response.output_item.added",
            "response_id": "response_1",
            "item": {"id": "output_lifecycle"},
        })

        client._handle_event({
            "type": "response.done",
            "response": {
                "id": "response_1",
                "metadata": {"hyprwhspr_request_id": request_id},
                "output": [
                    {"id": "output_1"},
                    {"id": "output_2"},
                    {"id": "output_1"},
                    {},
                ],
            },
        })

        self.assertEqual(
            {event["item_id"] for event in client.ws.sent if event["type"] == "conversation.item.delete"},
            {"input_1", "output_1", "output_2", "output_lifecycle"},
        )
        self.assertTrue(all(
            event["type"] == "conversation.item.delete"
            for event in client.ws.sent
            if event["type"] != "response.create"
        ))
        self.assertEqual(client._take_item_ids, set())
        self.assertTrue(client.response_complete)

    def test_converse_request_metadata_correlates_response_text(self):
        client = RealtimeClient(mode="converse")
        client.connected = True
        client.ws = FakeWebSocket()

        client._request_transcript({"buffer_was_committed": True})
        request = client.ws.sent[-1]["response"]
        request_id = request["metadata"]["hyprwhspr_request_id"]
        self.assertEqual(request["output_modalities"], ["text"])

        client._handle_event(self._response_created(request_id, "response_1"))
        client._handle_event({
            "type": "response.output_text.delta",
            "response_id": "response_1",
            "delta": "hello",
        })
        client._handle_event({
            "type": "response.done",
            "response": {
                "id": "response_1",
                "metadata": {"hyprwhspr_request_id": request_id},
            },
        })

        self.assertEqual(client.current_response_text, "hello")
        self.assertTrue(client.response_event.is_set())

    def test_converse_timeout_cancels_and_stale_events_do_not_complete_next_request(self):
        client = RealtimeClient(mode="converse")
        client.connected = True
        client.ws = FakeWebSocket()
        client.set_conversation_history("turn")
        client._handle_event({"type": "input_audio_buffer.committed", "item_id": "input_1"})
        client._request_transcript({"buffer_was_committed": True})
        first_request_id = client.ws.sent[-1]["response"]["metadata"]["hyprwhspr_request_id"]
        client._handle_event(self._response_created(first_request_id, "response_1"))
        client._handle_event({
            "type": "response.output_text.delta", "response_id": "response_1",
            "item_id": "output_1", "delta": "late",
        })

        client._on_response_timeout()

        self.assertIn(
            {"type": "response.cancel", "response_id": "response_1"}, client.ws.sent,
        )
        self.assertEqual(
            {event["item_id"] for event in client.ws.sent if event["type"] == "conversation.item.delete"},
            {"input_1", "output_1"},
        )

        client._request_transcript({"buffer_was_committed": True})
        second_request_id = client.ws.sent[-1]["response"]["metadata"]["hyprwhspr_request_id"]
        client._handle_event(self._response_created(second_request_id, "response_2"))
        client.response_event.clear()
        client.current_response_text = ""
        client._handle_event({
            "type": "response.output_text.delta", "response_id": "response_1", "delta": "stale",
        })
        client._handle_event({
            "type": "response.done",
            "response": {"id": "response_1", "metadata": {"hyprwhspr_request_id": first_request_id}},
        })

        self.assertEqual(client.current_response_text, "")
        self.assertFalse(client.response_event.is_set())
        self.assertFalse(client.response_complete)

    def test_converse_wait_timeout_emits_response_cancel(self):
        client = RealtimeClient(mode="converse")
        client.connected = True
        client.ws = FakeWebSocket()

        with mock.patch.object(realtime_base.time, "sleep"):
            self.assertEqual(client.commit_and_get_text(timeout=0), "")

        self.assertEqual(client.ws.sent[-1]["type"], "response.cancel")

    def test_converse_ignores_uncorrelated_provider_events(self):
        client = RealtimeClient(mode="converse")
        client.connected = True
        client.ws = FakeWebSocket()
        client._request_transcript({"buffer_was_committed": True})

        client._handle_event({"type": "response.created", "response": {"id": "unknown"}})
        client._handle_event({"type": "response.output_text.delta", "response_id": "unknown", "delta": "wrong"})
        client._handle_event({"type": "response.done", "response": {"id": "unknown"}})

        self.assertEqual(client.current_response_text, "")
        self.assertFalse(client.response_event.is_set())

    def test_delta_updates_preview_and_completed_is_final_text(self):
        previews = []
        client = self._client_with_ws()
        client.set_partial_transcript_callback(previews.append)

        client._handle_event({"type": "conversation.item.input_audio_transcription.delta", "delta": "hello"})
        client._handle_event({"type": "conversation.item.input_audio_transcription.delta", "delta": " wor"})
        client._handle_event({"type": "conversation.item.input_audio_transcription.completed", "transcript": "hello world"})

        self.assertEqual(previews, ["hello", "hello wor", ""])
        self.assertEqual(client.commit_and_get_text(timeout=0.1), "hello world")

    def test_gpt_transcribe_commit_accepts_detected_languages_metadata(self):
        for detected_languages in ([{"code": "fr"}], []):
            with self.subTest(detected_languages=detected_languages):
                client = self._client_with_ws("gpt-transcribe")

                client._request_transcript({"buffer_was_committed": False})
                client._handle_event({
                    "type": "input_audio_buffer.committed",
                    "item_id": "item_003",
                })
                client._handle_event({
                    "type": "conversation.item.input_audio_transcription.delta",
                    "item_id": "item_003",
                    "delta": "Bonjour, ",
                })
                client._handle_event({
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "item_003",
                    "transcript": "Bonjour, pouvez-vous m'entendre ?",
                    "languages": detected_languages,
                })

                self.assertEqual(
                    client.commit_and_get_text(timeout=0.1),
                    "Bonjour, pouvez-vous m'entendre ?",
                )
                self.assertEqual(
                    client.ws.sent,
                    [{"type": "input_audio_buffer.commit"}],
                )

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

    def test_schema_declares_realtime_conversation_history_values(self):
        schema = json.loads((ROOT / "share" / "config.schema.json").read_text())
        history_schema = schema["properties"]["realtime_conversation_history"]

        self.assertEqual(history_schema["default"], "turn")
        self.assertEqual(history_schema["enum"], ["session", "turn"])

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

        fake_soxr = types.SimpleNamespace(
            resample=lambda samples, source, target, quality="HQ": samples[::2]
        )
        with mock.patch.dict(sys.modules, {"soxr": fake_soxr}):
            resampled = client._resample_for_output(audio)

        self.assertEqual(len(resampled), 2400)
        self.assertEqual(resampled.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
