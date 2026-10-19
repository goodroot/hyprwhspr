"""Mid-recording capture stall watchdog (#282); audio and desktop are mocked."""

import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from tests.test_suspend_resume_recovery import _import_main_isolated, patch_app_global
from tests.text_injector_helpers import ConfigStub

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib' / 'src'))


class CallbackStalledTests(unittest.TestCase):
    def capture(self, last=100.0, recording=True, recovering=False):
        with mock.patch.dict(sys.modules, {'sounddevice': mock.Mock()}):
            import audio_capture
        self.module = audio_capture
        capture = audio_capture.AudioCapture.__new__(audio_capture.AudioCapture)
        capture.lock = threading.Lock()
        capture.last_callback_monotonic = last
        capture.is_recording = recording
        capture.recovery_in_progress = recovering
        return capture

    def stalled(self, capture, now):
        with mock.patch.object(self.module.time, 'monotonic', return_value=now):
            return capture.callback_stalled()

    def test_quiet_stream_past_threshold_is_stalled(self):
        self.assertTrue(self.stalled(self.capture(), 102.5))

    def test_recent_callback_is_not_stalled(self):
        self.assertFalse(self.stalled(self.capture(), 101.5))

    def test_never_delivered_not_recording_or_recovering_is_not_stalled(self):
        for kwargs in ({'last': 0.0}, {'recording': False}, {'recovering': True}):
            with self.subTest(**kwargs):
                self.assertFalse(self.stalled(self.capture(**kwargs), 500.0))


class StopRecordingStalledTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def app(self, mode='toggle'):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app._recording_lock = threading.Lock()
        app._recording_session = object()
        app.is_recording = True
        app.config = ConfigStub({'recording_mode': mode})
        app._continuous_silence_stop = threading.Event()
        app._continuous_silence_thread = None
        app._continuous_transcription_done = threading.Event()
        app._continuous_transcription_done.set()
        app._notify_user = mock.Mock()
        app._stop_recording = mock.Mock()
        return app

    def test_stops_owned_session_and_notifies(self):
        app = self.app()
        session = app._recording_session
        app._stop_recording_stalled(session)
        app._stop_recording.assert_called_once_with(expected_session=session)
        app._notify_user.assert_called_once()
        self.assertFalse(app._continuous_silence_stop.is_set())

    def test_continuous_drains_before_stopping(self):
        app = self.app('continuous')
        stop_event = app._continuous_silence_stop
        app._stop_recording.side_effect = lambda **kw: self.assertTrue(stop_event.is_set())
        app._stop_recording_stalled(app._recording_session)
        app._stop_recording.assert_called_once()

    def test_restart_during_notification_keeps_new_session_monitor(self):
        app = self.app('continuous')
        old_session, old_stop = app._recording_session, app._continuous_silence_stop
        new_stop, new_thread = threading.Event(), mock.Mock()

        def cancel_and_restart(*args):
            # The user cancels and starts a new continuous session meanwhile
            app._recording_session = object()
            app._continuous_silence_stop = new_stop
            app._continuous_silence_thread = new_thread
        app._notify_user.side_effect = cancel_and_restart

        app._stop_recording_stalled(old_session)

        self.assertTrue(old_stop.is_set())
        self.assertFalse(new_stop.is_set())
        self.assertIs(app._continuous_silence_thread, new_thread)
        app._stop_recording.assert_called_once_with(expected_session=old_session)

    def test_later_session_is_left_alone(self):
        app = self.app()
        app._stop_recording_stalled(object())
        app._stop_recording.assert_not_called()
        app._notify_user.assert_not_called()


class LevelMonitorStallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def test_monitor_hands_its_session_to_the_stall_stop(self):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app.audio_level_thread = None
        app._audio_level_stop = threading.Event()
        app._recording_session = session = object()
        app.is_recording = True
        app.config = ConfigStub({})
        app.audio_capture = mock.Mock()
        app.audio_capture.callback_stalled.return_value = True
        stopped = threading.Event()
        app._stop_recording_stalled = mock.Mock(side_effect=lambda s: stopped.set())

        with tempfile.TemporaryDirectory() as tmp:
            with patch_app_global(self.main.hyprwhsprApp, 'AUDIO_LEVEL_FILE', Path(tmp) / 'audio_level'):
                app._start_audio_level_monitoring()
                self.assertTrue(stopped.wait(2.0))
                app.audio_level_thread.join(2.0)

        self.assertFalse(app.audio_level_thread.is_alive())
        app._stop_recording_stalled.assert_called_once_with(session)
        app.audio_capture.get_audio_level.assert_not_called()


class _FastEvent(threading.Event):
    def wait(self, timeout=None):
        return super().wait(0.001)


class MuteDetectionStallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def run_monitor(self, advancing):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app.audio_level_thread = None
        app._audio_level_stop = _FastEvent()
        app._recording_session = object()
        app.is_recording = True
        app.config = ConfigStub({'mute_detection': True})
        app.audio_capture = mock.Mock(lock=threading.Lock(), frames_since_start=7, current_level=0.0)
        app.audio_capture.get_audio_level.return_value = 0.0
        app._cancel_recording_muted = mock.Mock()
        ticks = [0]

        def tick():
            ticks[0] += 1
            if advancing:
                app.audio_capture.frames_since_start += 1
            if ticks[0] > 40:
                app._audio_level_stop.set()
            return False
        app.audio_capture.callback_stalled.side_effect = tick

        with tempfile.TemporaryDirectory() as tmp:
            with patch_app_global(self.main.hyprwhsprApp, 'AUDIO_LEVEL_FILE', Path(tmp) / 'audio_level'):
                app._start_audio_level_monitoring()
                app.audio_level_thread.join(5.0)
        return app._cancel_recording_muted

    def test_frozen_silent_level_does_not_cancel(self):
        # A stall leaves current_level at the last chunk's value (#282)
        self.run_monitor(advancing=False).assert_not_called()

    def test_fresh_silent_callbacks_still_cancel(self):
        self.run_monitor(advancing=True).assert_called_once()


if __name__ == '__main__':
    unittest.main()
