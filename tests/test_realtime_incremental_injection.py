"""Regression tests for opt-in incremental (append-only) injection of
finalized realtime segments (`realtime_incremental_injection`).

Covers three layers:
- NemoRealtimeClient: callback ordering/semantics around
  conversation.item.input_audio_transcription.completed.
- RealtimeWsBackend: wiring rules, transcribe() suppression, preview
  suppression.
- main.hyprwhsprApp: _inject_committed_segment defer/drop/inject branches and
  the _process_audio delivered-incrementally path.
"""

import json
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))
sys.path.insert(0, str(ROOT / "lib"))
sys.modules.setdefault("websocket", types.SimpleNamespace(WebSocketApp=object))

from tests.test_suspend_resume_recovery import _import_main_isolated

from nemo_realtime_client import NemoRealtimeClient
from backends import RealtimeWsBackend
import backends.realtime_ws_backend as realtime_ws_backend


class FakeWebSocket:
    def __init__(self):
        self.sent = []

    def send(self, payload):
        self.sent.append(json.loads(payload))


def _client_with_ws():
    client = NemoRealtimeClient(mode="transcribe")
    client.connected = True
    client.ws = FakeWebSocket()
    client.model = "nemotron-speech-streaming-en-0.6b"
    return client


def _completed_event(transcript, item_id="item_1"):
    return {
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": transcript,
        "item_id": item_id,
    }


class NemoRealtimeClientIncrementalTests(unittest.TestCase):
    """Layer 1: the client's callback wiring in _handle_event."""

    def test_callback_runs_before_response_event_is_set(self):
        client = _client_with_ws()
        observed = {}

        def cb(text):
            observed["event_was_set"] = client.response_event.is_set()
            return True

        client.set_committed_segment_callback(cb)
        client._handle_event(_completed_event("hello there"))

        self.assertFalse(observed["event_was_set"])
        self.assertTrue(client.response_event.is_set())

    def test_truthy_return_sets_incremental_flag(self):
        client = _client_with_ws()
        client.set_committed_segment_callback(lambda text: True)

        client._handle_event(_completed_event("hello there"))

        self.assertTrue(client._incremental_injected_any)

    def test_falsy_return_leaves_flag_false_but_buffers_segment(self):
        client = _client_with_ws()
        client.set_committed_segment_callback(lambda text: False)

        client._handle_event(_completed_event("deferred segment"))

        self.assertFalse(client._incremental_injected_any)
        # Shared handling still ran, so commit_and_get_text's fallback path
        # (the joined transcript) still has the text.
        self.assertEqual(client._committed_segments, ["deferred segment"])
        self.assertTrue(client.response_event.is_set())

    def test_callback_exception_does_not_propagate_and_leaves_flag_false(self):
        client = _client_with_ws()

        def boom(text):
            raise RuntimeError("callback exploded")

        client.set_committed_segment_callback(boom)

        try:
            client._handle_event(_completed_event("hello there"))
        except Exception:
            self.fail("_handle_event must not propagate a callback exception")

        self.assertFalse(client._incremental_injected_any)
        # The base handling must still run despite the callback failing.
        self.assertEqual(client._committed_segments, ["hello there"])
        self.assertTrue(client.response_event.is_set())

    def test_retired_item_is_not_delivered_to_callback(self):
        client = _client_with_ws()
        calls = []
        client.set_committed_segment_callback(lambda text: calls.append(text) or True)

        with client.lock:
            client._retired_item_ids.append("stale_item")

        client._handle_event(_completed_event("late segment", item_id="stale_item"))

        self.assertEqual(calls, [])
        self.assertFalse(client._incremental_injected_any)

    def test_clear_audio_buffer_resets_flag(self):
        client = _client_with_ws()
        client.set_committed_segment_callback(lambda text: True)
        client._handle_event(_completed_event("hello there"))
        self.assertTrue(client._incremental_injected_any)

        client.clear_audio_buffer()

        self.assertFalse(client._incremental_injected_any)

    def test_no_callback_registered_behaves_like_before(self):
        client = _client_with_ws()
        # No set_committed_segment_callback call at all.

        client._handle_event(_completed_event("hello there"))

        self.assertFalse(client._incremental_injected_any)
        self.assertEqual(client._committed_segments, ["hello there"])
        self.assertTrue(client.response_event.is_set())


class FakeConfig:
    def __init__(self, values=None):
        self.values = values or {}

    def get_setting(self, key, default=None):
        return self.values.get(key, default)


class RealtimeWsBackendIncrementalTests(unittest.TestCase):
    """Layer 2: backend wiring, transcribe() suppression, preview suppression."""

    def _manager(self, config):
        return types.SimpleNamespace(
            config=config,
            temp_dir="/tmp",
            ready=False,
            current_model=None,
            _last_use_time=0,
            _realtime_partial_callback=None,
            _realtime_committed_callback=None,
        )

    def _init_backend(self, provider_id, extra_config=None, connect_ok=True):
        values = {
            "websocket_provider": provider_id,
            "websocket_model": (
                "nemotron-speech-streaming-en-0.6b" if provider_id == "nemo" else "gpt-realtime-whisper"
            ),
            "realtime_mode": "transcribe",
        }
        if extra_config:
            values.update(extra_config)
        config = FakeConfig(values)
        manager = self._manager(config)
        callback = mock.Mock(return_value=True)
        manager._realtime_committed_callback = callback
        backend = RealtimeWsBackend(manager)

        with mock.patch.object(realtime_ws_backend, "get_credential", return_value="test-key"), \
             mock.patch("nemo_realtime_client.NemoRealtimeClient.connect", return_value=connect_ok), \
             mock.patch("realtime_client.RealtimeClient.connect", return_value=connect_ok):
            ok = backend.initialize()

        self.assertTrue(ok)
        return backend, callback

    def test_callback_wired_for_nemo_with_flag_true(self):
        backend, callback = self._init_backend(
            "nemo", {"realtime_incremental_injection": True}
        )
        self.assertIs(backend._realtime_client._committed_segment_callback, callback)

    def test_callback_not_wired_for_openai_with_flag_true(self):
        backend, callback = self._init_backend(
            "openai", {"realtime_incremental_injection": True}
        )
        # Base RealtimeClient has no such attribute/slot at all - it was never wired.
        self.assertFalse(hasattr(backend._realtime_client, "_committed_segment_callback"))

    def test_callback_not_wired_for_nemo_with_flag_false(self):
        backend, callback = self._init_backend(
            "nemo", {"realtime_incremental_injection": False}
        )
        self.assertIsNone(backend._realtime_client._committed_segment_callback)

    def test_transcribe_returns_empty_when_delivered_incrementally(self):
        config = FakeConfig({})
        backend = RealtimeWsBackend(self._manager(config))
        backend._realtime_client = types.SimpleNamespace(
            connected=True,
            _incremental_injected_any=True,
            commit_and_get_text=mock.Mock(return_value="leftover joined text"),
        )

        result = backend.transcribe(_audio_data=None)

        self.assertEqual(result, "")

    def test_transcribe_returns_joined_text_when_not_delivered_incrementally(self):
        config = FakeConfig({})
        backend = RealtimeWsBackend(self._manager(config))
        backend._realtime_client = types.SimpleNamespace(
            connected=True,
            _incremental_injected_any=False,
            commit_and_get_text=mock.Mock(return_value="the whole transcript "),
        )

        result = backend.transcribe(_audio_data=None)

        self.assertEqual(result, "the whole transcript")

    def test_preview_disabled_when_incremental_injection_enabled(self):
        config = FakeConfig(
            {"mic_osd_enabled": True, "realtime_incremental_injection": True}
        )
        manager = self._manager(config)
        manager._realtime_partial_callback = mock.Mock()
        backend = RealtimeWsBackend(manager)

        self.assertFalse(
            backend._is_partial_preview_enabled("nemo", "nemotron-speech-streaming-en-0.6b", "transcribe")
        )

    def test_preview_enabled_for_nemo_when_flag_false(self):
        config = FakeConfig(
            {"mic_osd_enabled": True, "realtime_incremental_injection": False}
        )
        manager = self._manager(config)
        manager._realtime_partial_callback = mock.Mock()
        backend = RealtimeWsBackend(manager)

        self.assertTrue(
            backend._is_partial_preview_enabled("nemo", "nemotron-speech-streaming-en-0.6b", "transcribe")
        )


class MainIncrementalInjectionTests(unittest.TestCase):
    """Layer 3: main.hyprwhsprApp._inject_committed_segment / _process_audio."""

    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def _app(self, transcribe_return=' ', delivered_incrementally=False):
        main = self.main
        app = main.hyprwhsprApp.__new__(main.hyprwhsprApp)
        app.config = types.SimpleNamespace(
            get_setting=lambda key, default=None: default,
            get_hallucination_markers=lambda: None,
        )
        app.text_injector = mock.Mock()
        app._recording_lock = threading.Lock()
        app._continuous_delivery_failure_notified = False
        app._recording_finalizing = threading.Event()
        app.is_recording = False
        app.is_processing = False
        app._incremental_outcome = None
        app._current_language_override = None
        app.audio_capture = types.SimpleNamespace(
            sample_rate=16000, abort_recovery=mock.Mock()
        )
        app._background_recovery_needed = threading.Event()
        app.audio_manager = mock.Mock()
        app._recording_control_server = mock.Mock()
        app._recording_control_server.has_capture_subscriber.return_value = False
        app._recording_control_server.is_trace_capture.return_value = False
        app._notify_user = mock.Mock()
        app._show_result_and_hide = mock.Mock()
        app.whisper_manager = types.SimpleNamespace(
            transcribe_audio=mock.Mock(return_value=transcribe_return),
            realtime_delivered_incrementally=mock.Mock(return_value=delivered_incrementally),
        )
        return app

    # -- _inject_committed_segment -------------------------------------

    def test_empty_text_defers(self):
        app = self._app()
        self.assertFalse(app._inject_committed_segment("   "))
        app.text_injector.inject_text.assert_not_called()

    def test_capture_subscriber_defers(self):
        app = self._app()
        app.is_recording = True
        app._recording_control_server.has_capture_subscriber.return_value = True
        self.assertFalse(app._inject_committed_segment("hello"))
        app.text_injector.inject_text.assert_not_called()

    def test_dropped_after_cancel(self):
        app = self._app()
        # Not recording, not processing, not finalizing -> a cancel already happened.
        self.assertFalse(app._inject_committed_segment("hello"))
        app.text_injector.inject_text.assert_not_called()

    def test_proceeds_while_recording(self):
        app = self._app()
        app.is_recording = True
        app.text_injector.inject_text.return_value = self.main.InjectionOutcome.INJECTED
        self.assertTrue(app._inject_committed_segment("hello there"))
        app.text_injector.inject_text.assert_called_once_with("hello there")

    def test_proceeds_while_processing(self):
        app = self._app()
        app.is_processing = True
        app.text_injector.inject_text.return_value = self.main.InjectionOutcome.INJECTED
        self.assertTrue(app._inject_committed_segment("hello there"))

    def test_proceeds_while_finalizing(self):
        app = self._app()
        app._recording_finalizing.set()
        app.text_injector.inject_text.return_value = self.main.InjectionOutcome.INJECTED
        self.assertTrue(app._inject_committed_segment("hello there"))

    def test_hallucination_filtered_without_injecting(self):
        app = self._app()
        app.is_recording = True
        self.assertTrue(app._inject_committed_segment("thanks for watching"))
        app.text_injector.inject_text.assert_not_called()
        self.assertIsNone(app._incremental_outcome)

    def test_failed_outcome_is_sticky_across_later_injected_segments(self):
        app = self._app()
        app.is_recording = True
        app.text_injector.inject_text.return_value = self.main.InjectionOutcome.FAILED
        app._notify_user = mock.Mock()
        self.assertTrue(app._inject_committed_segment("first segment"))
        self.assertEqual(app._incremental_outcome, self.main.InjectionOutcome.FAILED)

        app.text_injector.inject_text.return_value = self.main.InjectionOutcome.INJECTED
        self.assertTrue(app._inject_committed_segment("second segment"))
        self.assertEqual(app._incremental_outcome, self.main.InjectionOutcome.FAILED)

    # -- _process_audio ---------------------------------------------------

    def test_process_audio_delivered_incrementally_no_error_sound_success(self):
        app = self._app(transcribe_return='', delivered_incrementally=True)
        app._incremental_outcome = self.main.InjectionOutcome.INJECTED

        app._process_audio(audio_data=[0.0])

        app.audio_manager.play_error_sound.assert_not_called()
        app._show_result_and_hide.assert_called_once_with(True)

    def test_process_audio_delivered_incrementally_failed_outcome_no_error_sound(self):
        app = self._app(transcribe_return='', delivered_incrementally=True)
        app._incremental_outcome = self.main.InjectionOutcome.FAILED

        app._process_audio(audio_data=[0.0])

        app.audio_manager.play_error_sound.assert_not_called()
        app._show_result_and_hide.assert_called_once_with(False)

    def test_process_audio_non_delivered_empty_transcription_still_errors(self):
        # Regression guard: the ordinary silent-recording path must be untouched.
        app = self._app(transcribe_return='', delivered_incrementally=False)

        app._process_audio(audio_data=[0.0])

        app.audio_manager.play_error_sound.assert_called_once()
        app._show_result_and_hide.assert_called_once_with(False)


if __name__ == "__main__":
    unittest.main()
