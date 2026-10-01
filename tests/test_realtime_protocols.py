import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))
sys.path.insert(0, str(ROOT / "lib"))
sys.modules.setdefault("websocket", types.SimpleNamespace(WebSocketApp=object))

from backends import RealtimeWsBackend  # noqa: E402
import backends.realtime_ws_backend as realtime_ws_backend  # noqa: E402
from dependency_plan import plan_key  # noqa: E402
from processing_trace import classify_vad_mode  # noqa: E402
from realtime_client import RealtimeClient  # noqa: E402
from realtime_protocols import PROTOCOLS, live_text_mode, load_client_class, resolve_protocol  # noqa: E402


class FakeConfig:
    def __init__(self, values):
        self.values = values

    def get_setting(self, key, default=None):
        return self.values.get(key, default)


def _backend(values):
    manager = types.SimpleNamespace(
        config=FakeConfig(values), temp_dir="/tmp", ready=False, current_model=None,
        _last_use_time=0, _realtime_partial_callback=None,
    )
    return RealtimeWsBackend(manager)


CUSTOM = {
    "transcription_backend": "realtime-ws",
    "websocket_provider": "custom",
    "websocket_model": "local-model",
    "websocket_url": "ws://localhost:8080/v1/realtime",
}


class ResolveProtocolTests(unittest.TestCase):
    def test_builtin_providers_name_their_protocol(self):
        self.assertEqual(resolve_protocol("openai").id, "openai-realtime")
        self.assertEqual(resolve_protocol("google").id, "gemini-live")
        self.assertEqual(resolve_protocol("elevenlabs").id, "elevenlabs")

    def test_custom_defaults_to_openai_realtime(self):
        self.assertEqual(resolve_protocol("custom").id, "openai-realtime")
        self.assertEqual(resolve_protocol("custom", None).id, "openai-realtime")

    def test_configured_protocol_applies_to_custom_only(self):
        self.assertEqual(resolve_protocol("custom", " Gemini-Live ").id, "gemini-live")
        self.assertEqual(resolve_protocol("openai", "gemini-live").id, "openai-realtime")

    def test_unknown_protocol_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_protocol("custom", "carrier-pigeon")

    def test_every_client_class_imports(self):
        self.assertIs(load_client_class(PROTOCOLS["openai-realtime"]), RealtimeClient)
        for protocol in PROTOCOLS.values():
            with self.subTest(protocol=protocol.id):
                self.assertTrue(callable(load_client_class(protocol)))

    def test_dependency_plan_follows_protocol(self):
        self.assertEqual(plan_key("realtime-ws", "elevenlabs", None, ValueError), "elevenlabs")
        self.assertEqual(plan_key("realtime-ws", "google", None, ValueError), "realtime")
        self.assertEqual(plan_key("realtime-ws", "custom", None, ValueError), "realtime")

    def test_trace_reads_protocol_vad(self):
        base = {"transcription_backend": "realtime-ws"}
        self.assertEqual(classify_vad_mode(FakeConfig({**base, "websocket_provider": "elevenlabs"})),
                         "provider_managed")
        self.assertEqual(classify_vad_mode(FakeConfig({**base, "websocket_provider": "google"})),
                         "server_vad")


class KeylessCustomTests(unittest.TestCase):
    def test_custom_initializes_without_a_key(self):
        backend = _backend(CUSTOM)
        with mock.patch.object(realtime_ws_backend, "get_credential", return_value=None), \
             mock.patch.object(RealtimeClient, "connect", return_value=True) as connect:
            self.assertTrue(backend.initialize())
        connect.assert_called_once_with(CUSTOM["websocket_url"], None, "local-model", None)

    def test_keyless_client_sends_no_auth_header(self):
        client = RealtimeClient()
        client.url, client.api_key = "ws://localhost/v1/realtime", None
        self.assertEqual(client._ws_connect_params(), ("ws://localhost/v1/realtime", None))
        client.api_key = "secret"
        self.assertEqual(client._ws_connect_params()[1], {"Authorization": "Bearer secret"})

    def test_keyless_custom_reconnects(self):
        backend = _backend(CUSTOM)
        client = mock.Mock(connected=False, connecting=False)
        client.connect.return_value = True
        backend._realtime_client = client
        backend._realtime_connect_params = {
            "websocket_url": CUSTOM["websocket_url"], "api_key": None,
            "model_id": "local-model", "instructions": None,
        }
        self.assertTrue(backend._reconnect_realtime_client())
        client.connect.assert_called_once_with(CUSTOM["websocket_url"], None, "local-model", None)

    def test_known_provider_still_requires_a_key(self):
        backend = _backend({"websocket_provider": "openai", "websocket_model": "gpt-transcribe"})
        with mock.patch.object(realtime_ws_backend, "get_credential", return_value=None), \
             mock.patch.object(RealtimeClient, "connect", return_value=True) as connect:
            self.assertFalse(backend.initialize())
        connect.assert_not_called()

    def test_unknown_protocol_fails_init(self):
        backend = _backend({**CUSTOM, "websocket_protocol": "carrier-pigeon"})
        with mock.patch.object(realtime_ws_backend, "get_credential", return_value=None):
            self.assertFalse(backend.initialize())
        self.assertIsNone(backend._realtime_client)


class SingleInitPathTests(unittest.TestCase):
    def test_transcribe_only_protocol_ignores_converse(self):
        cls = load_client_class(PROTOCOLS["elevenlabs"])
        backend = _backend({
            "websocket_provider": "elevenlabs",
            "websocket_model": "scribe_v2_realtime",
            "realtime_mode": "converse",
        })
        with mock.patch.object(realtime_ws_backend, "get_credential", return_value="key-1234567890"), \
             mock.patch.object(cls, "connect", return_value=True) as connect:
            self.assertTrue(backend.initialize())
        url, key, model, instructions = connect.call_args.args
        self.assertEqual(url, "wss://api.elevenlabs.io/v1/speech-to-text/realtime")
        self.assertIsNone(instructions)

    def test_gemini_gets_registry_endpoint_and_instructions(self):
        cls = load_client_class(PROTOCOLS["gemini-live"])
        backend = _backend({
            "websocket_provider": "google",
            "websocket_model": "gemini-3.1-flash-live-preview",
            "language": "de",
        })
        with mock.patch.object(realtime_ws_backend, "get_credential", return_value="key-1234567890"), \
             mock.patch.object(cls, "connect", return_value=True) as connect:
            self.assertTrue(backend.initialize())
        url, _key, _model, instructions = connect.call_args.args
        self.assertTrue(url.startswith("wss://generativelanguage.googleapis.com/"))
        self.assertEqual(instructions, "Transcribe in de language.")
        self.assertEqual(backend._realtime_client.language, "de")

    def test_streaming_callback_feeds_the_client(self):
        backend = _backend(CUSTOM)
        with mock.patch.object(realtime_ws_backend, "get_credential", return_value=None), \
             mock.patch.object(RealtimeClient, "connect", return_value=True), \
             mock.patch.object(RealtimeClient, "append_audio") as append:
            self.assertTrue(backend.initialize())
            backend._realtime_streaming_callback("chunk")
        append.assert_called_once_with("chunk")



class LiveTextModeTests(unittest.TestCase):
    def test_openai_continuous_models_are_revisable(self):
        self.assertEqual(live_text_mode("openai", "gpt-live-transcribe"), "revisable")
        self.assertEqual(live_text_mode("openai", "gpt-transcribe"), "none")

    def test_other_builtin_providers_show_nothing_mid_turn(self):
        self.assertEqual(live_text_mode("elevenlabs", "scribe_v2_realtime"), "none")
        self.assertEqual(live_text_mode("google", "gemini-3.1-flash-live-preview"), "none")

    def test_custom_takes_the_configured_promise(self):
        self.assertEqual(live_text_mode("custom", "m"), "none")
        self.assertEqual(live_text_mode("custom", "m", "append_only"), "append_only")
        self.assertEqual(live_text_mode("custom", "m", "bogus"), "none")
        self.assertEqual(live_text_mode("openai", "gpt-transcribe", "append_only"), "none")

    def test_waveform_preview_follows_live_text(self):
        for live_text, expected in (("none", False), ("revisable", True), ("append_only", True)):
            with self.subTest(live_text=live_text):
                backend = _backend({**CUSTOM, "websocket_live_text": live_text, "mic_osd_style": "waveform"})
                backend._manager._realtime_partial_callback = lambda _text: None
                self.assertEqual(
                    backend._is_partial_preview_enabled("custom", "local-model", "transcribe"), expected)


if __name__ == "__main__":
    unittest.main()
