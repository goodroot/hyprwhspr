"""Push-to-talk hold-to-lock: a long enough hold latches the recording on release.

`push_to_talk_lock_seconds` is read only in push_to_talk mode. With it unset (0,
the default) every release stops the recording exactly as before, so the cases
below cover both the new latch and the untouched default behavior.
"""

import threading
import time
import types
import unittest
from unittest import mock

from tests.test_suspend_resume_recovery import _import_main_isolated

LOCK_KEY = 'push_to_talk_lock_seconds'

# release in mode -> (_stop_recording calls, _notify_user calls)
RELEASE_EXPECTED = {
    'toggle': (0, 0),
    'continuous': (0, 0),
    'auto': (1, 0),
    'long_form': (0, 0),
}
# control 'stop' in mode -> (_stop_recording, long-form pause, continuous wait, notify)
STOP_EXPECTED = {
    'toggle': (1, 0, 0, 0),
    'continuous': (1, 0, 1, 0),
    'auto': (1, 0, 0, 0),
    'long_form': (0, 1, 0, 0),
}


class _FrozenClock:
    """`time` module stand-in with a settable `now`; other attributes proxy through."""

    def __init__(self, now=1000.0):
        self.now = now

    def time(self):
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


class _FakeCapture:
    """audio_capture stand-in: frames_since_start advances on every read, so the
    stream verification in _start_recording succeeds without hardware."""

    def __init__(self):
        self.lock = threading.Lock()
        self.sample_rate = 16000
        self.stop_result = b'captured audio'
        self._frames = 0

    @property
    def frames_since_start(self):
        self._frames += 1
        return self._frames

    def start_recording(self, streaming_callback=None):
        return True

    def stop_recording(self):
        return self.stop_result

    def abort_recovery(self):
        return None


class PushToTalkLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def setUp(self):
        self.clock = _FrozenClock()
        patcher = mock.patch.object(self.main, 'time', self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _app(self, recording_mode='push_to_talk', lock_seconds=0.0, lock_absent=False,
             recording=False, mock_stop=True, mock_start=True, mock_cancel=True):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        config = mock.Mock()

        def get_setting(key, default=None):
            if key == 'recording_mode':
                return recording_mode
            if key == LOCK_KEY and not lock_absent:
                return lock_seconds
            return default

        config.get_setting.side_effect = get_setting
        app.config = config
        app.is_recording = recording
        app._recording_lock = threading.Lock()
        app._recording_finalizing = threading.Event()
        app._current_language_override = None
        app._ptt_lock = threading.Lock()
        app._ptt_press_time = None
        app._ptt_locked = False
        if mock_start:
            app._start_recording = mock.Mock()
        if mock_stop:
            app._stop_recording = mock.Mock()
        if mock_cancel:
            app._cancel_recording = mock.Mock()
        app._notify_user = mock.Mock()
        app._notify_capture = mock.Mock()
        app._notify_zero_volume = mock.Mock()
        app._cleanup_recording_state = mock.Mock()
        app._clear_mic_osd_preview_text = mock.Mock()
        app._clear_zero_volume_signal = mock.Mock()
        app._hide_mic_osd = mock.Mock()
        app._show_mic_osd = mock.Mock()
        app._set_visualizer_state = mock.Mock()
        app._start_audio_level_monitoring = mock.Mock()
        app._stop_audio_level_monitoring = mock.Mock()
        app._write_recording_status = mock.Mock()
        app._release_blocked_capture = mock.Mock()
        app._autostop_stop_silence_monitor = mock.Mock()
        app._autostop_start_silence_monitor = mock.Mock()
        app._continuous_stop_silence_monitor = mock.Mock()
        app._continuous_stop_and_wait = mock.Mock()
        app._continuous_start_silence_monitor = mock.Mock()
        app._continuous_cancelled = False
        app._is_zero_volume = lambda audio_data: False
        app._process_audio = mock.Mock()
        app._background_recovery_needed = threading.Event()
        app._model_initializing = False
        app._backend_init_failed = False
        app._file_transcription_active = False
        app._model_operation_active = False
        app._auto_mode_lock = threading.Lock()
        app._shortcut_press_time = self.clock.now - 5.0
        app._recording_started_this_press = False
        app._tap_threshold = 0.4
        app._longform = types.SimpleNamespace(
            request_start=mock.Mock(),
            request_pause=mock.Mock(),
            request_cancel=mock.Mock(),
        )
        app.audio_capture = _FakeCapture()
        app.audio_manager = types.SimpleNamespace(
            play_start_sound=mock.Mock(),
            play_stop_sound=mock.Mock(),
            play_error_sound=mock.Mock(),
        )
        app.playback_suppressor = types.SimpleNamespace(is_active=False, restore=mock.Mock())
        app.whisper_manager = types.SimpleNamespace(
            realtime_client_missing=lambda: False,
            get_realtime_streaming_callback=lambda: None,
            update_realtime_language=mock.Mock(),
            discard_realtime_audio=mock.Mock(),
        )
        return app

    def _pending_press(self, app, held_seconds):
        """Pretend the accepted press that started this session happened `held_seconds` ago."""
        app._ptt_press_time = self.clock.now - held_seconds

    def test_release_below_threshold_stops(self):
        app = self._app(lock_seconds=3.0, recording=True, mock_stop=False)
        self._pending_press(app, held_seconds=1.5)
        app._on_shortcut_released()
        self.assertFalse(app.is_recording)
        self.assertEqual(app.audio_manager.play_stop_sound.call_count, 1)
        self.assertEqual(app._notify_user.call_count, 0)
        self.assertFalse(app._ptt_locked)

    def test_release_at_or_above_threshold_latches(self):
        for held in (3.0, 4.2):
            with self.subTest(held=held):
                app = self._app(lock_seconds=3.0, recording=True)
                self._pending_press(app, held_seconds=held)
                app._on_shortcut_released()
                self.assertEqual(app._stop_recording.call_count, 0)
                self.assertTrue(app._ptt_locked)
                self.assertEqual(app._notify_user.call_count, 1)

    def test_next_press_ends_latched_session(self):
        app = self._app(lock_seconds=3.0, recording=True, mock_stop=False)
        self._pending_press(app, held_seconds=4.0)
        app._on_shortcut_released()
        self.assertTrue(app.is_recording)
        self.assertEqual(app.audio_manager.play_stop_sound.call_count, 0)

        app._handle_shortcut_triggered()
        self.assertFalse(app.is_recording)
        self.assertEqual(app.audio_manager.play_stop_sound.call_count, 1)
        self.assertFalse(app._ptt_locked)

    def test_lock_disabled_never_latches(self):
        for lock_absent in (False, True):
            with self.subTest(lock_absent=lock_absent):
                app = self._app(lock_seconds=0.0, lock_absent=lock_absent,
                                recording=True, mock_stop=False)
                self._pending_press(app, held_seconds=30.0)
                app._on_shortcut_released()
                self.assertFalse(app.is_recording)
                self.assertEqual(app.audio_manager.play_stop_sound.call_count, 1)
                self.assertEqual(app._notify_user.call_count, 0)
                self.assertFalse(app._ptt_locked)

    def test_control_stop_latches_then_next_stop_ends_session(self):
        app = self._app(lock_seconds=3.0, recording=True)
        self._pending_press(app, held_seconds=4.0)
        app._handle_control_command('stop')
        self.assertEqual(app._stop_recording.call_count, 0)
        self.assertTrue(app._ptt_locked)

        app._handle_control_command('stop')
        self.assertEqual(app._stop_recording.call_count, 1)

    def test_control_start_ends_latched_session(self):
        app = self._app(lock_seconds=3.0, recording=True, mock_stop=False)
        self._pending_press(app, held_seconds=4.0)
        app._on_shortcut_released()
        self.assertTrue(app.is_recording)

        app._handle_control_command('start')
        self.assertFalse(app.is_recording)
        self.assertFalse(app._ptt_locked)

    def test_cancel_clears_lock_state(self):
        app = self._app(lock_seconds=3.0, recording=True, mock_cancel=False)
        self._pending_press(app, held_seconds=4.0)
        app._on_shortcut_released()
        self.assertTrue(app._ptt_locked)

        app._handle_control_command('cancel')
        self.assertFalse(app.is_recording)
        self.assertFalse(app._ptt_locked)

    def test_start_clears_stale_latch_and_stamps_press(self):
        app = self._app(lock_seconds=3.0, recording=False, mock_start=False)
        app._ptt_locked = True

        app._start_recording()

        self.assertTrue(app.is_recording)
        self.assertFalse(app._ptt_locked)
        self.assertEqual(app._ptt_press_time, self.clock.now)

        self.clock.now += 4.0
        app._on_shortcut_released()
        self.assertTrue(app.is_recording)
        self.assertTrue(app._ptt_locked)

    def test_other_modes_unaffected(self):
        for mode in ('toggle', 'continuous', 'auto', 'long_form'):
            with self.subTest(mode=mode):
                outcomes = []
                for lock_seconds in (0.0, 3.0):
                    release_app = self._app(recording_mode=mode, lock_seconds=lock_seconds,
                                            recording=True)
                    self._pending_press(release_app, held_seconds=5.0)
                    release_app._on_shortcut_released()
                    release = (release_app._stop_recording.call_count,
                               release_app._notify_user.call_count)

                    stop_app = self._app(recording_mode=mode, lock_seconds=lock_seconds,
                                         recording=True)
                    self._pending_press(stop_app, held_seconds=5.0)
                    stop_app._handle_control_command('stop')
                    control = (stop_app._stop_recording.call_count,
                               stop_app._longform.request_pause.call_count,
                               stop_app._continuous_stop_and_wait.call_count,
                               stop_app._notify_user.call_count)

                    self.assertFalse(release_app._ptt_locked)
                    self.assertFalse(stop_app._ptt_locked)
                    outcomes.append((release, control))

                self.assertEqual(outcomes[0], outcomes[1], 'lock value changed behavior')
                self.assertEqual(outcomes[0][0], RELEASE_EXPECTED[mode])
                self.assertEqual(outcomes[0][1], STOP_EXPECTED[mode])


if __name__ == '__main__':
    unittest.main()
