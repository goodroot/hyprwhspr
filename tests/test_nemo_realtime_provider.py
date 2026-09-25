"""
Regression tests for the self-hosted NeMo-Speech.cpp realtime provider
(lib/src/nemo_realtime_client.py, and the 'nemo' wiring in
provider_registry.py / backends/realtime_ws_backend.py / cli/setup.py /
diagnostics.py).
"""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))
sys.modules.setdefault("websocket", types.SimpleNamespace(WebSocketApp=object))

from nemo_realtime_client import NemoRealtimeClient
from realtime_client import RealtimeClient
from provider_registry import provider_requires_api_key, get_models_for_backend, get_realtime_mode
from backends import RealtimeWsBackend
import backends.realtime_ws_backend as realtime_ws_backend
from cli import setup as setup_cli
import diagnostics


class FakeWebSocket:
    def __init__(self):
        self.sent = []

    def send(self, payload):
        self.sent.append(json.loads(payload))


class FakeConfig:
    def __init__(self, values):
        self.values = values

    def get_setting(self, key, default=None):
        return self.values.get(key, default)


# ---------------------------------------------------------------------------
# 1. NemoRealtimeClient
# ---------------------------------------------------------------------------

class NemoRealtimeClientTests(unittest.TestCase):
    def _client_with_ws(self):
        client = NemoRealtimeClient(mode="transcribe")
        client.connected = True
        client.ws = FakeWebSocket()
        return client

    def test_sample_rate_is_16khz(self):
        client = NemoRealtimeClient(mode="transcribe")
        self.assertEqual(client.sample_rate, 16000)

    def test_session_update_is_flat_with_no_language_or_prompt(self):
        client = self._client_with_ws()
        client._send_session_update()

        payload = client.ws.sent[-1]
        self.assertEqual(payload, {"type": "session.update", "session": {"sample_rate": 16000}})
        self.assertNotIn("audio", payload["session"])

    def test_session_update_includes_language_when_set(self):
        client = self._client_with_ws()
        client.language = "en"
        client._send_session_update()

        session = client.ws.sent[-1]["session"]
        self.assertEqual(session["language"], "en")
        self.assertNotIn("audio", session)

    def test_session_update_includes_prompt_only_in_transcribe_mode_when_set(self):
        client = self._client_with_ws()
        client.transcription_prompt = "Linux dictation."
        client._send_session_update()

        session = client.ws.sent[-1]["session"]
        self.assertEqual(session["prompt"], "Linux dictation.")

    def test_session_update_omits_prompt_when_unset(self):
        client = self._client_with_ws()
        client._send_session_update()
        self.assertNotIn("prompt", client.ws.sent[-1]["session"])

    def test_ws_connect_params_no_api_key(self):
        client = NemoRealtimeClient(mode="transcribe")
        client.url = "ws://127.0.0.1:8080/v1/realtime"
        client.api_key = None
        url, headers = client._ws_connect_params()
        self.assertEqual(url, "ws://127.0.0.1:8080/v1/realtime")
        self.assertIsNone(headers)

    def test_ws_connect_params_with_api_key(self):
        client = NemoRealtimeClient(mode="transcribe")
        client.url = "ws://127.0.0.1:8080/v1/realtime"
        client.api_key = "secret"
        url, headers = client._ws_connect_params()
        self.assertEqual(url, "ws://127.0.0.1:8080/v1/realtime")
        self.assertEqual(headers, {"Authorization": "Bearer secret"})


# ---------------------------------------------------------------------------
# 2. Provider registry
# ---------------------------------------------------------------------------

class ProviderRegistryTests(unittest.TestCase):
    def test_nemo_does_not_require_api_key(self):
        self.assertFalse(provider_requires_api_key("nemo"))

    def test_known_cloud_providers_require_api_key(self):
        for provider_id in ("openai", "google", "elevenlabs", "custom"):
            self.assertTrue(provider_requires_api_key(provider_id))

    def test_unknown_provider_requires_api_key(self):
        self.assertTrue(provider_requires_api_key("totally-unknown-provider"))

    def test_nemo_model_listed_for_realtime_ws(self):
        models = get_models_for_backend("nemo", "realtime-ws")
        self.assertIn("nemotron-speech-streaming-en-0.6b", models)

    def test_nemo_mode_is_transcribe(self):
        self.assertEqual(
            get_realtime_mode("nemo", "nemotron-speech-streaming-en-0.6b"), "transcribe"
        )


# ---------------------------------------------------------------------------
# 3. Backend initialize()
# ---------------------------------------------------------------------------

class NemoBackendInitializeTests(unittest.TestCase):
    def _manager(self, extra=None):
        values = {
            "transcription_backend": "realtime-ws",
            "websocket_provider": "nemo",
            "websocket_model": "nemotron-speech-streaming-en-0.6b",
            "realtime_mode": "transcribe",
        }
        if extra:
            values.update(extra)
        config = FakeConfig(values)
        return types.SimpleNamespace(
            config=config, temp_dir="/tmp", ready=False, current_model=None,
            _last_use_time=0, _realtime_partial_callback=None,
        )

    def test_initialize_succeeds_with_no_stored_credential(self):
        manager = self._manager()
        backend = RealtimeWsBackend(manager)

        with mock.patch.object(realtime_ws_backend, "get_credential", return_value=None), \
             mock.patch.object(NemoRealtimeClient, "connect", return_value=True):
            self.assertTrue(backend.initialize())

        self.assertIsInstance(backend._realtime_client, NemoRealtimeClient)

    def test_default_url_matches_registry_endpoint_with_no_query(self):
        manager = self._manager()
        backend = RealtimeWsBackend(manager)

        captured = {}

        def fake_connect(self, url, api_key, model_id, instructions):
            captured["url"] = url
            return True

        with mock.patch.object(realtime_ws_backend, "get_credential", return_value=None), \
             mock.patch.object(NemoRealtimeClient, "connect", fake_connect):
            self.assertTrue(backend.initialize())

        self.assertEqual(captured["url"], "ws://127.0.0.1:8080/v1/realtime")
        self.assertNotIn("?", captured["url"])

    def test_explicit_websocket_url_overrides_default(self):
        manager = self._manager({"websocket_url": "ws://10.0.0.5:9000/v1/realtime"})
        backend = RealtimeWsBackend(manager)

        captured = {}

        def fake_connect(self, url, api_key, model_id, instructions):
            captured["url"] = url
            return True

        with mock.patch.object(realtime_ws_backend, "get_credential", return_value=None), \
             mock.patch.object(NemoRealtimeClient, "connect", fake_connect):
            self.assertTrue(backend.initialize())

        self.assertEqual(captured["url"], "ws://10.0.0.5:9000/v1/realtime")

    def test_openai_without_credential_still_fails(self):
        values = {
            "transcription_backend": "realtime-ws",
            "websocket_provider": "openai",
            "websocket_model": "gpt-realtime-whisper",
            "realtime_mode": "transcribe",
        }
        manager = types.SimpleNamespace(
            config=FakeConfig(values), temp_dir="/tmp", ready=False, current_model=None,
            _last_use_time=0, _realtime_partial_callback=None,
        )
        backend = RealtimeWsBackend(manager)

        with mock.patch.object(realtime_ws_backend, "get_credential", return_value=None):
            self.assertFalse(backend.initialize())


# ---------------------------------------------------------------------------
# 4. Reconnect
# ---------------------------------------------------------------------------

class NemoReconnectTests(unittest.TestCase):
    def _backend_with_params(self, provider_id, api_key):
        values = {"websocket_provider": provider_id}
        manager = types.SimpleNamespace(
            config=FakeConfig(values), temp_dir="/tmp", ready=False, current_model=None,
            _last_use_time=0, _realtime_partial_callback=None,
        )
        backend = RealtimeWsBackend(manager)
        backend._realtime_client = mock.Mock()
        backend._realtime_client.connecting = False
        backend._realtime_client.connected = False
        backend._realtime_connect_params = {
            "websocket_url": "ws://127.0.0.1:8080/v1/realtime",
            "api_key": api_key,
            "model_id": "some-model",
            "instructions": None,
        }
        return backend

    def test_nemo_reconnect_proceeds_without_api_key(self):
        backend = self._backend_with_params("nemo", api_key=None)
        backend._realtime_client.connect.return_value = True

        self.assertTrue(backend._reconnect_realtime_client())
        backend._realtime_client.connect.assert_called_once()

    def test_openai_reconnect_reports_missing_parameters_without_api_key(self):
        backend = self._backend_with_params("openai", api_key=None)

        self.assertFalse(backend._reconnect_realtime_client())
        backend._realtime_client.connect.assert_not_called()
        self.assertEqual(backend.last_connect_failure, "failed")


# ---------------------------------------------------------------------------
# 5. Preview / waveform partial-transcript enablement
# ---------------------------------------------------------------------------

class NemoPreviewTests(unittest.TestCase):
    def test_nemo_partial_preview_enabled_in_waveform_style(self):
        values = {
            "mic_osd_enabled": True,
            "mic_osd_style": "waveform",
        }
        manager = types.SimpleNamespace(
            config=FakeConfig(values), temp_dir="/tmp", ready=False, current_model=None,
            _last_use_time=0, _realtime_partial_callback=lambda text: None,
        )
        backend = RealtimeWsBackend(manager)

        self.assertTrue(
            backend._is_partial_preview_enabled(
                "nemo", "nemotron-speech-streaming-en-0.6b", "transcribe"
            )
        )

    def test_nemo_partial_preview_disabled_without_osd(self):
        values = {
            "mic_osd_enabled": False,
            "mic_osd_style": "waveform",
        }
        manager = types.SimpleNamespace(
            config=FakeConfig(values), temp_dir="/tmp", ready=False, current_model=None,
            _last_use_time=0, _realtime_partial_callback=lambda text: None,
        )
        backend = RealtimeWsBackend(manager)

        self.assertFalse(
            backend._is_partial_preview_enabled(
                "nemo", "nemotron-speech-streaming-en-0.6b", "transcribe"
            )
        )


# ---------------------------------------------------------------------------
# 6. _generate_remote_config
# ---------------------------------------------------------------------------

class GenerateRemoteConfigNemoTests(unittest.TestCase):
    def test_includes_websocket_url_when_custom_config_has_one(self):
        config = setup_cli._generate_remote_config(
            "nemo",
            "nemotron-speech-streaming-en-0.6b",
            None,
            {"websocket_url": "ws://127.0.0.1:8099/v1/realtime"},
            backend_type="realtime-ws",
        )
        self.assertEqual(config["websocket_url"], "ws://127.0.0.1:8099/v1/realtime")
        self.assertEqual(config["websocket_provider"], "nemo")
        self.assertEqual(config["websocket_model"], "nemotron-speech-streaming-en-0.6b")

    def test_omits_websocket_url_when_custom_config_is_none(self):
        config = setup_cli._generate_remote_config(
            "nemo",
            "nemotron-speech-streaming-en-0.6b",
            None,
            None,
            backend_type="realtime-ws",
        )
        self.assertNotIn("websocket_url", config)


# ---------------------------------------------------------------------------
# 7. Diagnostics
# ---------------------------------------------------------------------------

class NemoDiagnosticsTests(unittest.TestCase):
    def test_nemo_realtime_config_produces_no_realtime_model_error(self):
        payload = {
            "transcription_backend": "realtime-ws",
            "websocket_provider": "nemo",
            "websocket_model": "nemotron-speech-streaming-en-0.6b",
            "realtime_mode": "transcribe",
        }
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            # CI sets HYPRWHSPR_ROOT=/nonexistent; validate against this checkout's schema.
            with mock.patch.object(
                diagnostics, "SCHEMA_FILE", ROOT / "share" / "config.schema.json"
            ):
                _, findings, fatal = diagnostics.validate_config(path=config_path)

        self.assertFalse(fatal)
        check_ids = {f["check_id"] for f in findings}
        self.assertNotIn("config.realtime_model", check_ids)
        self.assertNotIn("config.realtime_mode", check_ids)
        self.assertNotIn("config.realtime_continuous", check_ids)


if __name__ == "__main__":
    unittest.main()
