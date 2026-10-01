"""silence_timeout ends a continuous session without touching its chunk pastes (#273)."""
import threading
import types
import unittest
from unittest import mock

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

        def stop():
            app.is_recording = False
        app._stop_recording = mock.Mock(side_effect=stop)
        return app

    def _run(self, app, ticks=None):
        app._continuous_start_silence_monitor()
        thread = app._continuous_silence_thread
        if ticks is not None:   # session that should not end on its own
            threading.Event().wait(ticks * app._POLL_INTERVAL)
            app._continuous_silence_stop.set()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

    def test_stops_after_timeout_from_last_speech_while_chunks_still_paste(self):
        # Chunks flush every silent tick; the stop timer keeps counting through them.
        app = self._app([1.0, 1.0, 0.0], silence_timeout=0.005)
        with patch_app_global(self.main.hyprwhsprApp, 'log', mock.Mock()):
            self._run(app)
        app._stop_recording.assert_called_once_with()
        self.assertEqual(app._continuous_flush_audio.call_count, 5)

    def test_speech_resets_the_stop_timer(self):
        app = self._app([1.0] + [0.0] * 4 + [1.0] + [0.0] * 4 + [0.0], silence_timeout=0.005)
        with patch_app_global(self.main.hyprwhsprApp, 'log', mock.Mock()):
            self._run(app)
        app._stop_recording.assert_called_once_with()
        self.assertEqual(app._continuous_flush_audio.call_count, 4 + 5)

    def test_never_stops_before_speech_or_when_disabled(self):
        for levels, timeout in (([0.0], 0.005), ([1.0, 0.0], 0)):
            with self.subTest(levels=levels, timeout=timeout):
                app = self._app(levels, silence_timeout=timeout)
                self._run(app, ticks=200)
                app._stop_recording.assert_not_called()
                app._continuous_flush_audio.assert_called()

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
            app._stop_recording.assert_called_once_with()

            app._continuous_autostop(object(), 1)    # a session that already ended
        app._stop_recording.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
