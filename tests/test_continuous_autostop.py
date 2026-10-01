"""silence_timeout ends a continuous session without touching its chunk pastes (#273)."""
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

from tests.test_suspend_resume_recovery import _import_main_isolated, patch_app_global


class LevelFeed:
    """rolling_avg_level replays a script, then stays at its last value."""

    def __init__(self, levels):
        self._levels = list(levels)

    @property
    def rolling_avg_level(self):
        return self._levels.pop(0) if len(self._levels) > 1 else self._levels[0]


class ContinuousAutostopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def _app(self, levels, silence_timeout, chunk_seconds=0.001):
        settings = {
            'continuous_silence_seconds': chunk_seconds,
            'continuous_silence_threshold': 0.5,   # skip noise-floor calibration
            'silence_timeout': silence_timeout,
        }
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app.config = types.SimpleNamespace(get_setting=lambda key, default=None: settings.get(key, default))
        app._POLL_INTERVAL = 0.001
        app._recording_lock = threading.Lock()
        app._recording_session = object()
        app._continuous_silence_stop = threading.Event()
        app._continuous_silence_thread = None
        app._continuous_transcription_done = threading.Event()
        app._continuous_transcription_done.set()
        app.audio_capture = LevelFeed(levels)
        app.is_recording = True
        app._continuous_flush_audio = mock.Mock()

        def stop(expected_session=None):
            if expected_session is not None and app._recording_session is not expected_session:
                return
            app.is_recording = False
        app._stop_recording = mock.Mock(side_effect=stop)
        return app

    def _flush_app(self, audio):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app._recording_lock = threading.Lock()
        app._recording_session = object()
        app._current_language_override = 'en'
        app.is_recording = True
        app._continuous_flush_lock = threading.Lock()
        app._continuous_transcription_done = threading.Event()
        app._continuous_transcription_done.set()
        app._continuous_cancelled = False
        app.audio_capture = mock.Mock()
        app.audio_capture.sample_rate = 10
        app.audio_capture.flush_buffer.return_value = audio
        app.config = types.SimpleNamespace(get_hallucination_markers=lambda: [])
        app.whisper_manager = mock.Mock()
        app._is_zero_volume = mock.Mock(return_value=False)
        app._save_debug_recording = mock.Mock()
        app._inject_text = mock.Mock()
        app._notify_capture = mock.Mock()
        return app

    def _run(self, app, ticks=None):
        app._continuous_start_silence_monitor()
        thread = app._continuous_silence_thread
        if ticks is not None:   # session that should not end on its own
            threading.Event().wait(ticks * app._POLL_INTERVAL)
            app._continuous_silence_stop.set()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

    # Chunks paste after 2 silent ticks; the session stops after 5.
    def test_stops_after_timeout_from_last_speech_while_chunks_still_paste(self):
        app = self._app([1.0, 1.0, 0.0], silence_timeout=0.005, chunk_seconds=0.002)
        with patch_app_global(self.main.hyprwhsprApp, 'log', mock.Mock()):
            self._run(app)
        app._stop_recording.assert_called_once_with(expected_session=app._recording_session)
        app._continuous_flush_audio.assert_called_once_with(expected_session=app._recording_session)  # silence after the pause did not flush

    def test_speech_resets_the_stop_timer(self):
        app = self._app([1.0] + [0.0] * 4 + [1.0] + [0.0] * 5, silence_timeout=0.005, chunk_seconds=0.002)
        with patch_app_global(self.main.hyprwhsprApp, 'log', mock.Mock()):
            self._run(app)
        app._stop_recording.assert_called_once_with(expected_session=app._recording_session)
        self.assertEqual(app._continuous_flush_audio.call_count, 2)

    def test_silence_timeout_uses_poll_intervals_instead_of_spinning(self):
        app = self._app([1.0, 0.0], silence_timeout=0.085, chunk_seconds=1)
        app._POLL_INTERVAL = 0.02
        stopped_at = []
        app._stop_recording.side_effect = lambda expected_session=None: stopped_at.append(time.monotonic())
        started_at = time.monotonic()
        with patch_app_global(self.main.hyprwhsprApp, 'log', mock.Mock()):
            self._run(app)
        self.assertEqual(len(stopped_at), 1)
        self.assertGreaterEqual(stopped_at[0] - started_at, 0.09)

    def test_quiet_room_never_pastes_or_stops(self):
        for levels, timeout, chunks in (([0.0], 0.005, 0), ([1.0, 0.0], 0, 1)):
            with self.subTest(levels=levels, timeout=timeout):
                app = self._app(levels, silence_timeout=timeout)
                self._run(app, ticks=200)
                app._stop_recording.assert_not_called()
                self.assertEqual(app._continuous_flush_audio.call_count, chunks)

    def test_busy_flush_keeps_the_chunk_pending_for_a_later_pause(self):
        app = self._app([1.0, 0.0], silence_timeout=0, chunk_seconds=0.002)
        app._continuous_flush_audio = mock.Mock(side_effect=[False, True])
        with patch_app_global(self.main.hyprwhsprApp, 'log', mock.Mock()):
            self._run(app, ticks=100)
        self.assertEqual(app._continuous_flush_audio.call_count, 2)
        app._stop_recording.assert_not_called()

    def test_flush_returns_busy_empty_and_error_outcomes(self):
        busy = self._flush_app(None)
        busy._continuous_flush_lock.acquire()
        self.assertFalse(busy._continuous_flush_audio())
        busy.audio_capture.flush_buffer.assert_not_called()
        busy._continuous_flush_lock.release()

        empty = self._flush_app(None)
        self.assertTrue(empty._continuous_flush_audio())
        self.assertTrue(empty._continuous_transcription_done.is_set())
        self.assertFalse(empty._continuous_flush_lock.locked())

        failed = self._flush_app(None)
        failed.audio_capture.flush_buffer.side_effect = RuntimeError('capture failed')
        with patch_app_global(self.main.hyprwhsprApp, 'log', mock.Mock()):
            self.assertFalse(failed._continuous_flush_audio())
        self.assertTrue(failed._continuous_transcription_done.is_set())
        self.assertFalse(failed._continuous_flush_lock.locked())

    def test_stale_monitor_flush_does_not_consume_replacement_audio(self):
        app = self._flush_app(None)
        old_session = app._recording_session
        with app._recording_lock:
            app._recording_session = object()
        self.assertFalse(app._continuous_flush_audio(expected_session=old_session))
        app.audio_capture.flush_buffer.assert_not_called()
        self.assertFalse(app._continuous_flush_lock.locked())

    def test_stale_worker_discards_result_without_finalizing_new_capture(self):
        app = self._flush_app(np.full(10, 0.25, dtype=np.float32))
        entered = threading.Event()
        release = threading.Event()

        def transcribe(*args, **kwargs):
            entered.set()
            release.wait(timeout=2)
            return 'old session text'

        app.whisper_manager.transcribe_audio.side_effect = transcribe
        with patch_app_global(self.main.hyprwhsprApp, 'log', mock.Mock()):
            app._continuous_flush_audio()
            self.assertTrue(entered.wait(timeout=1))
            with app._recording_lock:
                app._recording_session = object()
            release.set()
            self.assertTrue(app._continuous_transcription_done.wait(timeout=1))
        app._inject_text.assert_not_called()
        app._notify_capture.assert_not_called()

    def test_each_monitor_generation_owns_its_stop_event(self):
        app = self._app([1.0], silence_timeout=0)
        app._continuous_start_silence_monitor()
        first_event = app._continuous_silence_stop
        first_thread = app._continuous_silence_thread
        app._continuous_start_silence_monitor()
        second_event = app._continuous_silence_stop
        second_thread = app._continuous_silence_thread
        try:
            self.assertIsNot(first_event, second_event)
            self.assertTrue(first_event.is_set())
            self.assertFalse(second_event.is_set())
        finally:
            app._continuous_stop_silence_monitor()
            first_thread.join(timeout=1)
            second_thread.join(timeout=1)

    def test_capture_flush_and_final_stop_deliver_disjoint_audio(self):
        capture = object.__new__(self.main.AudioCapture)
        capture.lock = threading.Lock()
        capture._record_stop_event = threading.Event()
        capture.record_thread = None
        capture.stream = None
        capture.is_recording = True
        capture.sample_rate = 10
        first_chunk = np.full(10, 0.25, dtype=np.float32)
        final_tail = np.full(10, 0.75, dtype=np.float32)
        capture.audio_data = [first_chunk]
        capture._buffered_samples = len(first_chunk)
        capture._buffer_capped = False

        flushed = capture.flush_buffer()
        with capture.lock:
            capture.audio_data.append(final_tail)
            capture._buffered_samples = len(final_tail)
        stopped = capture.stop_recording()

        np.testing.assert_array_equal(flushed, first_chunk)
        np.testing.assert_array_equal(stopped, final_tail)

    def test_waits_for_an_in_flight_chunk_and_skips_a_replaced_session(self):
        app = self._app([0.0], silence_timeout=1)
        app._continuous_transcription_done.clear()
        waited = threading.Thread(target=app._continuous_autostop, args=(app._recording_session, 1))
        with patch_app_global(self.main.hyprwhsprApp, 'log', mock.Mock()):
            waited.start()
            threading.Event().wait(0.05)
            app._stop_recording.assert_not_called()   # chunk still pasting
            app._continuous_transcription_done.set()
            waited.join(timeout=2)
            app._stop_recording.assert_called_once_with(expected_session=app._recording_session)

            app._continuous_autostop(object(), 1)    # a session that already ended
        app._stop_recording.assert_called_once_with(expected_session=app._recording_session)

    def test_stale_autostop_cannot_stop_a_replacement_session(self):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app._recording_lock = threading.Lock()
        app.is_recording = True
        app._recording_session = object()
        stale_session = object()

        self.main.hyprwhsprApp._stop_recording(app, expected_session=stale_session)

        self.assertTrue(app.is_recording)

    def test_autostop_wait_exits_when_its_session_is_replaced(self):
        app = self._app([0.0], silence_timeout=1)
        done = threading.Event()
        entered_wait = threading.Event()

        class ObservedEvent:
            def wait(self, timeout=None):
                entered_wait.set()
                return done.wait(timeout)

        app._continuous_transcription_done = ObservedEvent()
        old_session = app._recording_session
        thread = threading.Thread(target=app._continuous_autostop, args=(old_session, 1))
        with patch_app_global(self.main.hyprwhsprApp, 'log', mock.Mock()):
            thread.start()
            self.assertTrue(entered_wait.wait(timeout=1))
            with app._recording_lock:
                app._recording_session = object()
            thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        app._stop_recording.assert_not_called()

    def test_monitor_can_tear_itself_down(self):
        # _stop_recording()'s error cleanup runs on the monitor thread; a self-join
        # raised and skipped the playback restore after it.
        app = self._app([0.0], silence_timeout=0)
        errors = []

        def run():
            try:
                app._continuous_stop_silence_monitor()
            except Exception as e:
                errors.append(e)
        app._continuous_silence_thread = thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=2)
        self.assertEqual(errors, [])
        self.assertIsNone(app._continuous_silence_thread)


if __name__ == '__main__':
    unittest.main()
