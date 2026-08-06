import json
import struct
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))
sys.path.insert(0, str(ROOT / "lib"))

websocket_stub = types.SimpleNamespace(
    WebSocketApp=object,
    ABNF=types.SimpleNamespace(OPCODE_BINARY=2),
)
websocket_module = sys.modules.setdefault("websocket", websocket_stub)
if not hasattr(websocket_module, "ABNF"):
    websocket_module.ABNF = websocket_stub.ABNF

from backends.realtime_ws_backend import RealtimeWsBackend
from realtimestt_realtime_client import RealtimeSTTRealtimeClient


class FakeWebSocket:
    def __init__(self):
        self.text_messages = []
        self.binary_messages = []

    def send(self, payload, opcode=None):
        if opcode == 2:
            self.binary_messages.append(payload)
        else:
            self.text_messages.append(json.loads(payload))


class RealtimeSTTClientTests(unittest.TestCase):
    def _client(self):
        client = RealtimeSTTRealtimeClient()
        client.connected = True
        client.ws = FakeWebSocket()
        return client

    def test_binary_audio_packet_matches_server_protocol(self):
        client = self._client()
        pcm = np.array([1, -2, 3], dtype=np.int16).tobytes()

        client._send_audio_payload(client.ws, pcm)

        packet = client.ws.binary_messages[-1]
        metadata_length = struct.unpack("<I", packet[:4])[0]
        metadata = json.loads(packet[4:4 + metadata_length])
        self.assertEqual(
            metadata,
            {
                "sampleRate": 16000,
                "channels": 1,
                "format": "pcm_s16le",
                "frames": 3,
            },
        )
        self.assertEqual(packet[4 + metadata_length:], pcm)

    def test_new_take_sends_clear_then_start(self):
        previews = []
        client = self._client()
        client.set_partial_transcript_callback(previews.append)

        client.start_audio_stream()

        self.assertEqual(
            client.ws.text_messages,
            [{"type": "clear"}, {"type": "start"}],
        )
        self.assertEqual(previews, [""])

    def test_realtime_and_final_events_assemble_preview_and_result(self):
        previews = []
        client = self._client()
        client.set_partial_transcript_callback(previews.append)

        client._handle_event({"type": "realtime", "segmentId": 1, "text": "hello wor"})
        client._handle_event({"type": "final", "segmentId": 1, "text": "hello world"})
        client._handle_event({"type": "realtime", "segmentId": 2, "displayText": "next seg"})
        client._handle_event({"type": "final", "segmentId": 2, "text": "next segment"})

        self.assertEqual(
            previews,
            [
                "hello wor",
                "hello world",
                "hello world next seg",
                "hello world next segment",
            ],
        )
        self.assertEqual(
            client.commit_and_get_text(timeout=0.1),
            "hello world next segment",
        )
        self.assertEqual(client.ws.text_messages[-1], {"type": "stop"})

    def test_idle_stop_returns_vad_final_instead_of_waiting_full_timeout(self):
        client = self._client()
        client.STOP_IDLE_GRACE_SECONDS = 0.01
        client._handle_event({"type": "final", "segmentId": 1, "text": "already final"})
        # A silence chunk can arrive between server VAD finalization and key-up.
        client._audio_activity_id += 1
        threading.Timer(
            0.08,
            lambda: client._handle_event({"type": "status", "state": "idle"}),
        ).start()

        started = time.monotonic()
        result = client.commit_and_get_text(timeout=3.0)

        self.assertEqual(result, "already final")
        self.assertLess(time.monotonic() - started, 1.0)

    def test_final_arriving_during_idle_grace_extends_existing_text(self):
        client = self._client()
        client.STOP_IDLE_GRACE_SECONDS = 0.2
        client._handle_event({"type": "final", "segmentId": 1, "text": "first"})
        client._audio_activity_id += 1
        threading.Timer(
            0.08,
            lambda: client._handle_event({"type": "status", "state": "idle"}),
        ).start()
        threading.Timer(
            0.12,
            lambda: client._handle_event(
                {"type": "final", "segmentId": 2, "text": "second"}
            ),
        ).start()

        self.assertEqual(client.commit_and_get_text(timeout=1.0), "first second")

    def test_clear_ack_rejects_in_flight_previous_take_segments(self):
        previews = []
        client = self._client()
        client.set_partial_transcript_callback(previews.append)
        client.start_audio_stream()

        client._handle_event({"type": "final", "segmentId": 1, "text": "stale"})
        client._handle_event({"type": "clear", "nextSegmentId": 2})
        client._handle_event({"type": "final", "segmentId": 1, "text": "still stale"})
        client._handle_event({"type": "realtime", "segmentId": 2, "text": "fresh"})

        self.assertEqual(previews, ["", "fresh"])

    def test_older_structured_realtime_sequence_is_ignored(self):
        previews = []
        client = self._client()
        client.set_partial_transcript_callback(previews.append)

        client._handle_event({
            "type": "realtime",
            "segmentId": 1,
            "sequence": 4,
            "displayText": "newest",
        })
        client._handle_event({
            "type": "realtime",
            "segmentId": 1,
            "sequence": 3,
            "displayText": "stale",
        })

        self.assertEqual(previews, ["newest"])

    def test_cancel_sends_stop_and_clear_without_restart(self):
        client = self._client()

        client.discard_audio_buffer()

        self.assertEqual(
            client.ws.text_messages,
            [{"type": "stop"}, {"type": "clear"}],
        )


class FakeConfig:
    def __init__(self, values):
        self.values = values

    def get_setting(self, key, default=None):
        return self.values.get(key, default)


class FakeManager:
    def __init__(self, config):
        self.config = config
        self.temp_dir = None
        self.ready = False
        self.current_model = None
        self._last_use_time = 0.0
        self._realtime_partial_callback = lambda _text: None


class RealtimeSTTBackendTests(unittest.TestCase):
    def test_initializes_without_api_key_or_client_model(self):
        manager = FakeManager(FakeConfig({
            "transcription_backend": "realtime-ws",
            "websocket_provider": "realtimestt",
            "websocket_url": "ws://127.0.0.1:8010/ws/transcribe",
            "realtime_mode": "transcribe",
            "mic_osd_enabled": True,
            "mic_osd_style": "waveform",
        }))
        backend = RealtimeWsBackend(manager)

        with mock.patch(
            "realtimestt_realtime_client.RealtimeSTTRealtimeClient.connect",
            return_value=True,
        ) as connect, mock.patch(
            "backends.realtime_ws_backend.get_credential"
        ) as credential:
            self.assertTrue(backend.initialize())

        credential.assert_not_called()
        connect.assert_called_once_with(
            "ws://127.0.0.1:8010/ws/transcribe",
            None,
            "server",
            None,
        )
        self.assertTrue(manager.ready)
        self.assertIsNotNone(backend._realtime_client.partial_transcript_callback)

    def test_recording_start_uses_protocol_lifecycle_hook(self):
        manager = FakeManager(FakeConfig({
            "transcription_backend": "realtime-ws",
            "websocket_provider": "realtimestt",
        }))
        backend = RealtimeWsBackend(manager)
        client = mock.Mock()
        client.connected = True
        backend._realtime_client = client
        backend._realtime_streaming_callback = object()

        callback = backend.get_streaming_callback()

        self.assertIs(callback, backend._realtime_streaming_callback)
        client.start_audio_stream.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
