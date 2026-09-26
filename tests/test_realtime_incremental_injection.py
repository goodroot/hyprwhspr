"""Regression tests for opt-in incremental (append-only) injection of
streamed realtime text (`realtime_incremental_injection`).

Covers three layers:
- NemoRealtimeClient: delta/completed callback semantics
  (set_stream_text_callback: cb(text, final, whole) -> bool).
- RealtimeWsBackend: wiring rules, transcribe() suppression, preview
  suppression.
- main.hyprwhsprApp: _inject_stream_text defer/drop/inject branches, the
  held hallucination phantom, and finishing the stream in _process_audio /
  on cancel.
- text_injector.TextInjector: the stream session (chunk joining, phrase
  hold-back, once-per-dictation clipboard restore / auto_submit / retention).
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
from tests.text_injector_helpers import ConfigStub, make_injector

from nemo_realtime_client import NemoRealtimeClient
from backends import RealtimeWsBackend
import backends.realtime_ws_backend as realtime_ws_backend
from text_injector import InjectionOutcome


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


def _delta_event(delta, item_id="item_1"):
    return {
        "type": "conversation.item.input_audio_transcription.delta",
        "delta": delta,
        "item_id": item_id,
    }


def _completed_event(transcript, item_id="item_1"):
    return {
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": transcript,
        "item_id": item_id,
    }


class NemoRealtimeClientIncrementalTests(unittest.TestCase):
    """Layer 1: the client's stream-text callback wiring in _handle_event."""

    def test_deltas_deliver_completed_words_only(self):
        client = _client_with_ws()
        calls = []
        client.set_stream_text_callback(
            lambda text, final, whole: calls.append((text, final, whole)) or True
        )

        for delta in ["Going", " along", " slu", "shy", " country"]:
            client._handle_event(_delta_event(delta))

        # Each delta delivers the words completed (whitespace-terminated)
        # since the last delivery: "Going" completes on " along"; "along"
        # completes on " slu" (a real word boundary, the leading space of the
        # next word); "slu"+"shy" never get delivered separately - "slu" has
        # no trailing whitespace yet, so it stays pending until "shy" joins it
        # into "slushy", which then completes on " country". No partial or
        # split word is ever delivered.
        self.assertEqual(
            calls,
            [
                ("Going", False, False),
                ("along", False, False),
                ("slushy", False, False),
            ],
        )

    def test_flush_stream_delivers_pending_words_including_a_partial_one(self):
        # The socket dropped before the segment finalized: the words already
        # streamed must not vanish behind the empty "delivered" transcript.
        client = _client_with_ws()
        calls = []
        client.set_stream_text_callback(
            lambda text, final, whole: calls.append((text, final, whole)) or True
        )
        for delta in ["Going", " along", " slu"]:
            client._handle_event(_delta_event(delta))

        client.flush_stream()

        self.assertEqual(calls[-1], ("slu", True, False))
        self.assertEqual(client._stream_text, "")

    def test_falsy_return_keeps_text_pending_for_retry(self):
        client = _client_with_ws()
        calls = []

        def cb(text, final, whole):
            calls.append(text)
            return False

        client.set_stream_text_callback(cb)

        client._handle_event(_delta_event("Going"))
        client._handle_event(_delta_event(" along"))
        client._handle_event(_delta_event(" country"))

        # Nothing was ever consumed (delivery never advances on a falsy
        # return), so each retry includes the full accumulated completed
        # words, growing each time.
        self.assertEqual(calls, ["Going", "Going along"])
        self.assertFalse(client._incremental_injected_any)

    def test_completed_delivers_tail_from_final_transcript(self):
        client = _client_with_ws()
        calls = []
        client.set_stream_text_callback(
            lambda text, final, whole: calls.append((text, final, whole)) or True
        )

        client._handle_event(_delta_event("Going"))
        client._handle_event(_delta_event(" along"))
        # "Going" delivered; "along" still pending (no trailing whitespace yet).
        self.assertEqual(calls, [("Going", False, False)])

        client._handle_event(_completed_event("Going along slushy."))

        # already-delivered word count = 1 ("Going"); tail = final_words[1:]
        # = "along slushy." -- taken from the final transcript so the
        # trailing period lands, and no duplicate of the recased word.
        self.assertEqual(calls[-1], ("along slushy.", True, False))

    def test_completed_with_nothing_delivered_is_whole(self):
        client = _client_with_ws()
        calls = []
        client.set_stream_text_callback(
            lambda text, final, whole: calls.append((text, final, whole)) or True
        )

        client._handle_event(_completed_event("hello there"))

        self.assertEqual(calls, [("hello there", True, True)])
        self.assertTrue(client._incremental_injected_any)

    def test_completed_ordering_guard_response_event_not_yet_set(self):
        client = _client_with_ws()
        observed = {}

        def cb(text, final, whole):
            observed["event_was_set"] = client.response_event.is_set()
            return True

        client.set_stream_text_callback(cb)
        client._handle_event(_completed_event("hello there"))

        self.assertFalse(observed["event_was_set"])
        self.assertTrue(client.response_event.is_set())

    def test_stream_resets_after_completed_for_next_segment(self):
        client = _client_with_ws()
        calls = []
        client.set_stream_text_callback(
            lambda text, final, whole: calls.append((text, final, whole)) or True
        )

        client._handle_event(_delta_event("Going along"))
        client._handle_event(_completed_event("Going along."))
        calls.clear()

        # New segment: word-count alignment must start fresh, not continue
        # counting from the previous segment.
        client._handle_event(_delta_event("Second"))
        client._handle_event(_completed_event("Second segment."))

        self.assertEqual(calls, [("Second segment.", True, True)])

    def test_tail_aligned_by_words_when_final_adds_punctuation_to_typed_words(self):
        # Finalization can add characters to words already typed (here a
        # comma after "morning"). A character-offset tail would then start
        # mid-token (", and ..."); word-count alignment must not.
        client = _client_with_ws()
        calls = []
        client.set_stream_text_callback(
            lambda text, final, whole: calls.append((text, final, whole)) or True
        )

        client._handle_event(_delta_event("Sunday morning and"))
        self.assertEqual(calls, [("Sunday morning", False, False)])
        calls.clear()

        client._handle_event(_completed_event("Sunday morning, and he can come."))

        self.assertEqual(calls, [("and he can come.", True, False)])

    def test_completed_with_final_shorter_than_delivered_is_skipped(self):
        client = _client_with_ws()
        calls = []
        client.set_stream_text_callback(
            lambda text, final, whole: calls.append((text, final, whole)) or True
        )

        client._handle_event(_delta_event("Going along slushy"))
        calls.clear()
        # Final transcript has fewer words than what was already delivered.
        client._handle_event(_completed_event("Going"))

        self.assertEqual(calls, [])

    def test_callback_exception_does_not_propagate_and_leaves_flag_false(self):
        client = _client_with_ws()

        def boom(text, final, whole):
            raise RuntimeError("callback exploded")

        client.set_stream_text_callback(boom)

        try:
            client._handle_event(_delta_event("hello "))
            client._handle_event(_completed_event("hello there"))
        except Exception:
            self.fail("_handle_event must not propagate a callback exception")

        self.assertFalse(client._incremental_injected_any)
        # The base handling must still run despite the callback failing.
        self.assertEqual(client._committed_segments, ["hello there"])
        self.assertTrue(client.response_event.is_set())

    def test_retired_item_is_not_delivered_to_callback_for_delta_or_completed(self):
        client = _client_with_ws()
        calls = []
        client.set_stream_text_callback(lambda text, final, whole: calls.append(text) or True)

        with client.lock:
            client._retired_item_ids.append("stale_item")

        client._handle_event(_delta_event("hello", item_id="stale_item"))
        client._handle_event(_completed_event("late segment", item_id="stale_item"))

        self.assertEqual(calls, [])
        self.assertFalse(client._incremental_injected_any)

    def test_clear_audio_buffer_resets_stream_and_flag(self):
        client = _client_with_ws()
        client.set_stream_text_callback(lambda text, final, whole: True)
        client._handle_event(_delta_event("Going along"))
        client._handle_event(_completed_event("Going along."))
        self.assertTrue(client._incremental_injected_any)

        client.clear_audio_buffer()

        self.assertFalse(client._incremental_injected_any)
        self.assertEqual(client._stream_text, "")
        self.assertEqual(client._stream_delivered, 0)

    def test_no_callback_registered_behaves_like_before(self):
        client = _client_with_ws()
        # No set_stream_text_callback call at all.

        client._handle_event(_delta_event("hello "))
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
            _realtime_stream_callback=None,
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
        manager._realtime_stream_callback = callback
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
        self.assertIs(backend._realtime_client._stream_text_callback, callback)

    def test_callback_not_wired_for_openai_with_flag_true(self):
        backend, callback = self._init_backend(
            "openai", {"realtime_incremental_injection": True}
        )
        # Base RealtimeClient has no such attribute/slot at all - it was never wired.
        self.assertFalse(hasattr(backend._realtime_client, "_stream_text_callback"))

    def test_callback_not_wired_for_nemo_with_flag_false(self):
        backend, callback = self._init_backend(
            "nemo", {"realtime_incremental_injection": False}
        )
        self.assertIsNone(backend._realtime_client._stream_text_callback)

    def test_callback_not_wired_in_converse_mode(self):
        # Converse returns the model's reply; typing the user's own speech
        # instead, and then discarding the reply, is wrong.
        backend, callback = self._init_backend(
            "nemo", {"realtime_incremental_injection": True, "realtime_mode": "converse"}
        )
        self.assertIsNone(backend._realtime_client._stream_text_callback)

    def test_callback_not_wired_while_a_post_transcription_hook_is_set(self):
        backend, callback = self._init_backend(
            "nemo", {"realtime_incremental_injection": True, "post_transcription_hook": "cat"}
        )
        self.assertIsNone(backend._realtime_client._stream_text_callback)

    def test_rewiring_rechecks_the_opt_in(self):
        # set_realtime_stream_callback can reach a live backend after init;
        # it must not switch streaming on for users who left it off.
        backend, callback = self._init_backend(
            "nemo", {"realtime_incremental_injection": False}
        )
        backend.apply_stream_callback(callback)
        self.assertIsNone(backend._realtime_client._stream_text_callback)

    def test_transcribe_flushes_streamed_words_when_not_connected(self):
        backend = RealtimeWsBackend(self._manager(FakeConfig({})))
        backend._realtime_client = types.SimpleNamespace(
            connected=False, _incremental_injected_any=True, flush_stream=mock.Mock(),
        )

        self.assertEqual(backend.transcribe(_audio_data=None), "")
        backend._realtime_client.flush_stream.assert_called_once()

    def test_transcribe_flushes_streamed_words_when_commit_fails(self):
        backend = RealtimeWsBackend(self._manager(FakeConfig({})))
        backend._realtime_client = types.SimpleNamespace(
            connected=True,
            _incremental_injected_any=True,
            commit_and_get_text=mock.Mock(side_effect=TimeoutError("no final")),
            flush_stream=mock.Mock(),
        )

        self.assertEqual(backend.transcribe(_audio_data=None), "")
        backend._realtime_client.flush_stream.assert_called_once()

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
    """Layer 3: main.hyprwhsprApp._inject_stream_text / _process_audio."""

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
        app._stream_phantom = None
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

    # -- _inject_stream_text -------------------------------------

    def test_empty_text_defers(self):
        app = self._app()
        self.assertFalse(app._inject_stream_text("   "))
        app.text_injector.inject_text.assert_not_called()

    def test_capture_subscriber_defers(self):
        app = self._app()
        app.is_recording = True
        app._recording_control_server.has_capture_subscriber.return_value = True
        self.assertFalse(app._inject_stream_text("hello"))
        app.text_injector.inject_text.assert_not_called()

    def test_dropped_after_cancel(self):
        app = self._app()
        # Not recording, not processing, not finalizing -> a cancel already happened.
        self.assertFalse(app._inject_stream_text("hello"))
        app.text_injector.inject_text.assert_not_called()

    def test_chunk_goes_to_the_stream_session_with_its_final_flag(self):
        app = self._app()
        app.is_recording = True
        app.text_injector.inject_stream_chunk.return_value = self.main.InjectionOutcome.INJECTED
        self.assertTrue(app._inject_stream_text("hello", final=False, whole=False))
        self.assertTrue(app._inject_stream_text("there.", final=True, whole=False))
        self.assertEqual(
            app.text_injector.inject_stream_chunk.call_args_list,
            [mock.call("hello", final=False), mock.call("there.", final=True)],
        )
        # Per-dictation delivery (hook, retention, auto_submit) is not per chunk.
        app.text_injector.inject_text.assert_not_called()
        self.assertEqual(app._incremental_outcome, self.main.InjectionOutcome.INJECTED)

    def test_proceeds_while_processing(self):
        app = self._app()
        app.is_processing = True
        app.text_injector.inject_stream_chunk.return_value = self.main.InjectionOutcome.INJECTED
        self.assertTrue(app._inject_stream_text("hello there"))
        app.text_injector.inject_stream_chunk.assert_called_once()

    def test_proceeds_while_finalizing(self):
        app = self._app()
        app._recording_finalizing.set()
        app.text_injector.inject_stream_chunk.return_value = self.main.InjectionOutcome.INJECTED
        self.assertTrue(app._inject_stream_text("hello there"))
        app.text_injector.inject_stream_chunk.assert_called_once()

    def test_first_whole_segment_phantom_is_held_not_typed(self):
        app = self._app()
        app.is_recording = True
        self.assertTrue(app._inject_stream_text("thanks for watching", final=True, whole=True))
        app.text_injector.inject_stream_chunk.assert_not_called()
        self.assertEqual(app._stream_phantom, "thanks for watching")

    def test_held_phantom_goes_out_with_the_speech_that_follows_it(self):
        # "Thank you." opening a longer dictation is real speech.
        app = self._app()
        app.is_recording = True
        app.text_injector.inject_stream_chunk.return_value = self.main.InjectionOutcome.INJECTED
        app._inject_stream_text("Thank you.", final=True, whole=True)
        app._inject_stream_text("for the help", final=False, whole=False)
        app.text_injector.inject_stream_chunk.assert_called_once_with(
            "Thank you. for the help", final=False
        )
        self.assertIsNone(app._stream_phantom)

    def test_phantom_matching_segment_after_real_text_is_typed(self):
        app = self._app()
        app.is_recording = True
        app.text_injector.inject_stream_chunk.return_value = self.main.InjectionOutcome.INJECTED
        app._inject_stream_text("Dear Sam, the report is done.", final=True, whole=True)
        app._inject_stream_text("Thank you.", final=True, whole=True)
        self.assertEqual(app.text_injector.inject_stream_chunk.call_count, 2)
        self.assertIsNone(app._stream_phantom)

    def test_mid_segment_chunk_matching_a_marker_is_typed(self):
        app = self._app()
        app.is_recording = True
        app.text_injector.inject_stream_chunk.return_value = self.main.InjectionOutcome.INJECTED
        self.assertTrue(
            app._inject_stream_text("thanks for watching", final=False, whole=False)
        )
        app.text_injector.inject_stream_chunk.assert_called_once_with(
            "thanks for watching", final=False
        )

    def test_injector_exception_is_a_failed_chunk_not_a_deferral(self):
        # Returning False here would leave the text for a joined transcript
        # that transcribe() discards once anything was delivered.
        app = self._app()
        app.is_recording = True
        app.text_injector.inject_stream_chunk.side_effect = RuntimeError("boom")
        self.assertTrue(app._inject_stream_text("hello"))
        self.assertEqual(app._incremental_outcome, self.main.InjectionOutcome.FAILED)

    def test_failed_outcome_is_sticky_across_later_injected_chunks(self):
        app = self._app()
        app.is_recording = True
        app.text_injector.inject_stream_chunk.return_value = self.main.InjectionOutcome.FAILED
        self.assertTrue(app._inject_stream_text("first segment"))
        self.assertEqual(app._incremental_outcome, self.main.InjectionOutcome.FAILED)

        app.text_injector.inject_stream_chunk.return_value = self.main.InjectionOutcome.INJECTED
        self.assertTrue(app._inject_stream_text("second segment"))
        self.assertEqual(app._incremental_outcome, self.main.InjectionOutcome.FAILED)
        # One notification for the dictation, at the end - not one per chunk.
        app._notify_user.assert_not_called()

    # -- _process_audio ---------------------------------------------------

    def test_process_audio_delivered_incrementally_finishes_the_stream(self):
        app = self._app(transcribe_return='', delivered_incrementally=True)
        app._incremental_outcome = self.main.InjectionOutcome.INJECTED
        app.text_injector.end_stream.return_value = self.main.InjectionOutcome.INJECTED

        app._process_audio(audio_data=[0.0])

        self.assertEqual(app.text_injector.end_stream.call_args_list[0], mock.call(submit=True))
        app.audio_manager.play_error_sound.assert_not_called()
        app._notify_user.assert_not_called()
        app._show_result_and_hide.assert_called_once_with(True)

    def test_process_audio_failed_chunk_reports_failure_and_notifies_once(self):
        app = self._app(transcribe_return='', delivered_incrementally=True)
        app._incremental_outcome = self.main.InjectionOutcome.FAILED
        app.text_injector.end_stream.return_value = self.main.InjectionOutcome.INJECTED

        app._process_audio(audio_data=[0.0])

        app.audio_manager.play_error_sound.assert_not_called()
        app._notify_user.assert_called_once()
        app._show_result_and_hide.assert_called_once_with(False)

    def test_process_audio_failed_finish_reports_failure(self):
        app = self._app(transcribe_return='', delivered_incrementally=True)
        app._incremental_outcome = self.main.InjectionOutcome.INJECTED
        app.text_injector.end_stream.return_value = self.main.InjectionOutcome.FAILED

        app._process_audio(audio_data=[0.0])

        app._notify_user.assert_called_once()
        app._show_result_and_hide.assert_called_once_with(False)

    def test_process_audio_phantom_only_recording_is_dropped(self):
        app = self._app(transcribe_return='', delivered_incrementally=True)
        app._stream_phantom = "thanks for watching"

        app._process_audio(audio_data=[0.0])

        app.text_injector.inject_stream_chunk.assert_not_called()
        self.assertEqual(app.text_injector.end_stream.call_args_list[0], mock.call(submit=False))
        app.audio_manager.play_error_sound.assert_called_once()
        app._show_result_and_hide.assert_called_once_with(False)
        self.assertIsNone(app._stream_phantom)

    def test_process_audio_non_delivered_empty_transcription_still_errors(self):
        # Regression guard: the ordinary silent-recording path must be untouched.
        app = self._app(transcribe_return='', delivered_incrementally=False)

        app._process_audio(audio_data=[0.0])

        app.audio_manager.play_error_sound.assert_called_once()
        app._show_result_and_hide.assert_called_once_with(False)

    def test_cancel_ends_the_stream_without_submitting(self):
        app = self._app()
        app.is_recording = True
        app._current_language_override = None
        app._cleanup_recording_state = mock.Mock()
        app.audio_capture.stop_recording = mock.Mock()
        app.whisper_manager.discard_realtime_audio = mock.Mock()

        app._cancel_recording()

        app.text_injector.end_stream.assert_called_once_with(submit=False)


class TextInjectorStreamSessionTests(unittest.TestCase):
    """Layer 4: TextInjector.inject_stream_chunk / end_stream."""

    def _injector(self, settings=None):
        injector = make_injector()
        injector.config_manager = ConfigStub(settings or {})
        self.pasted = []

        def paste(text, auto_submit=True, retain=False, stream=None):
            self.assertFalse(auto_submit)
            self.assertFalse(retain)
            self.assertIsNotNone(stream)
            self.pasted.append(text)
            return True

        patcher = mock.patch.object(injector, "_inject_via_clipboard_and_hotkey", side_effect=paste)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.enter = mock.patch.object(injector, "_send_enter_if_auto_submit").start()
        self.restore = mock.patch.object(injector, "_restore_clipboard").start()
        self.addCleanup(mock.patch.stopall)
        return injector

    def test_chunks_are_joined_with_a_leading_space(self):
        injector = self._injector({"append_trailing_space": False})
        injector.inject_stream_chunk("can you", final=False)
        injector.inject_stream_chunk("check the logs", final=False)
        self.assertEqual(self.pasted, ["can you", " check the logs"])

    def test_auto_submit_enter_is_sent_once_at_the_end(self):
        injector = self._injector({"auto_submit": True})
        injector.inject_stream_chunk("can you", final=False)
        injector.inject_stream_chunk("check the logs", final=True)
        self.enter.assert_not_called()
        self.assertEqual(injector.end_stream(), InjectionOutcome.INJECTED)
        self.enter.assert_called_once()

    def test_segments_stay_separated_when_trailing_space_is_off(self):
        injector = self._injector({"append_trailing_space": False})
        injector.inject_stream_chunk("Hello.", final=True)
        injector.inject_stream_chunk("How are you?", final=True)
        injector.end_stream()
        self.assertEqual("".join(self.pasted), "Hello. How are you?")

    def test_trailing_space_setting_applies_once_at_the_end(self):
        injector = self._injector({"append_trailing_space": True})
        injector.inject_stream_chunk("Hello there.", final=True)
        injector.end_stream()
        self.assertEqual(self.pasted, ["Hello there.", " "])

    def test_spoken_punctuation_attaches_to_the_previous_chunk(self):
        injector = self._injector({"append_trailing_space": False})
        injector.inject_stream_chunk("hello", final=False)
        injector.inject_stream_chunk("comma world", final=False)
        self.assertEqual("".join(self.pasted), "hello, world")

    def test_multi_word_command_split_across_chunks_is_held_together(self):
        injector = self._injector({"append_trailing_space": False})
        injector.inject_stream_chunk("is it done question", final=False)
        injector.inject_stream_chunk("mark", final=True)
        self.assertEqual("".join(self.pasted), "is it done?")

    def test_new_line_split_across_chunks_survives(self):
        injector = self._injector({"append_trailing_space": False})
        injector.inject_stream_chunk("first item new", final=False)
        injector.inject_stream_chunk("line second item", final=True)
        self.assertEqual("".join(self.pasted), "first item\nsecond item")

    def test_new_line_at_a_segment_end_waits_for_the_next_word(self):
        injector = self._injector({"append_trailing_space": False})
        injector.inject_stream_chunk("first item new line", final=True)
        injector.inject_stream_chunk("second item", final=True)
        injector.end_stream()
        self.assertEqual("".join(self.pasted), "first item\nsecond item")

    def test_multi_word_override_split_across_chunks_matches(self):
        injector = self._injector({
            "append_trailing_space": False,
            "word_overrides": {"hyper whisper": "hyprwhspr"},
        })
        injector.inject_stream_chunk("I use hyper", final=False)
        injector.inject_stream_chunk("whisper daily", final=False)
        self.assertEqual("".join(self.pasted), "I use hyprwhspr daily")

    def test_held_words_are_delivered_at_the_end(self):
        injector = self._injector({"append_trailing_space": False})
        injector.inject_stream_chunk("open the question", final=False)
        injector.end_stream()
        self.assertEqual("".join(self.pasted), "open the question")

    def test_whole_dictation_is_retained_for_recovery(self):
        injector = self._injector({"append_trailing_space": False})
        injector.inject_stream_chunk("hello", final=False)
        injector.inject_stream_chunk("there friend.", final=True)
        injector.end_stream()
        self.assertEqual(injector._last_text, "hello there friend.")

    def test_cancel_skips_held_words_and_enter(self):
        injector = self._injector({"auto_submit": True, "append_trailing_space": True})
        injector.inject_stream_chunk("open the question", final=False)
        injector.end_stream(submit=False)
        self.assertEqual(self.pasted, ["open the"])
        self.enter.assert_not_called()

    def test_failed_chunk_makes_the_dictation_fail_without_enter(self):
        injector = self._injector({"auto_submit": True})
        with mock.patch.object(injector, "_inject_via_clipboard_and_hotkey", return_value=False):
            self.assertEqual(
                injector.inject_stream_chunk("hello", final=True), InjectionOutcome.FAILED
            )
        self.assertEqual(injector.end_stream(), InjectionOutcome.FAILED)
        self.enter.assert_not_called()

    def test_end_without_a_stream_is_a_noop(self):
        injector = self._injector()
        self.assertIsNone(injector.end_stream())
        self.enter.assert_not_called()
        self.restore.assert_not_called()


class TextInjectorStreamClipboardTests(unittest.TestCase):
    """The clipboard is saved before the first chunk and restored once."""

    def test_clipboard_saved_once_and_restored_once_to_the_original(self):
        injector = make_injector()
        injector.config_manager = ConfigStub({"append_trailing_space": False})
        clipboard = {"value": b"https://original.example"}

        def copy(text):
            clipboard["value"] = text.encode("utf-8")
            return True

        with (
            mock.patch.object(injector, "_save_clipboard", side_effect=lambda: clipboard["value"]) as save,
            mock.patch.object(injector, "_copy_text_to_clipboard", side_effect=copy),
            mock.patch.object(injector, "_is_x11_session", return_value=False),
            mock.patch.object(injector, "_is_hyprland_session", return_value=True),
            mock.patch.object(injector, "_is_gnome_wayland_session", return_value=False),
            mock.patch.object(injector, "_active_window_lookup_needed", return_value=False),
            mock.patch.object(injector, "_resolve_paste_chord", return_value=("ctrl+v", None)),
            mock.patch.object(injector, "_send_shortcut_hyprland", return_value=True),
            mock.patch.object(injector, "_restore_clipboard") as restore,
            mock.patch.object(injector, "_send_enter_if_auto_submit"),
            mock.patch("text_injector.time.sleep"),
        ):
            injector.inject_stream_chunk("hello there", final=False)
            injector.inject_stream_chunk("friend", final=True)
            restore.assert_not_called()
            injector.end_stream()

        save.assert_called_once()
        restore.assert_called_once()
        self.assertEqual(restore.call_args.args[0], b"https://original.example")
        self.assertEqual(restore.call_args.kwargs["injected"], b" friend")


if __name__ == "__main__":
    unittest.main()
