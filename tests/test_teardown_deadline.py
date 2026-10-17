"""A stalled cancel must eventually restart, without interrupting transcription."""

import threading
import types
import unittest
from unittest import mock

from tests import test_stream_orphan_recovery as capture_tests
from tests.test_suspend_resume_recovery import _import_main_isolated, patch_app_global


class TeardownDeadlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def setUp(self):
        fixture = capture_tests.StreamOrphanRecoveryTests()
        self.addCleanup(fixture.tearDown)
        self.addCleanup(fixture.doCleanups)
        self.sd = capture_tests.FakeSoundDevice()
        self.module = fixture._load_audio_capture(self.sd)
        self.capture = fixture._start_recording(self.module, self.sd)
        self.release = threading.Event()
        self.capture.stream.stop = self.release.wait
        started = fixture._fast_teardowns(self.capture)

        def finish():
            self.release.set()
            fixture._stop_and_join(self.capture)

        self.addCleanup(finish)
        fixture._stop_while_teardown_pending(self.capture, started)
        with self.capture.lock:
            self.started_at = (min(self.capture._pending_teardowns.values())
                               - self.module._TEARDOWN_DEADLINE_SECONDS)

        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        self.app = app
        app.audio_capture = self.capture
        app._recording_lock = threading.Lock()
        app._recording_finalizing = threading.Event()
        for name in ('is_recording', 'is_processing', '_recording_starting',
                     '_file_transcription_active', '_longform_active', '_model_operation_active'):
            setattr(app, name, False)
        app.recovery_attempted = threading.Event()
        app._notify_user = mock.Mock()
        app._write_recovery_result = mock.Mock()
        app._mic_state_lock = threading.Lock()
        app._mic_disconnected = False
        self.capture.on_unrecoverable_stream = app._on_unrecoverable_audio_stream
        self.exit = mock.Mock()

        patches = [
            mock.patch('instance_detection.is_running_under_systemd', return_value=True),
            patch_app_global(type(app), 'os', types.SimpleNamespace(_exit=self.exit)),
            patch_app_global(type(app), 'RECOVERY_REQUESTED_FILE',
                             mock.Mock(exists=lambda: False)),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def tick(self, age):
        with mock.patch.object(self.module.time, 'monotonic', return_value=self.started_at + age):
            self.app._attempt_recovery_if_needed()

    def test_cancel_before_pulse_recovery_restarts_without_waiting_for_deadline(self):
        # The pending teardown began on the dead server; no 30s wait (#209)
        with patch_app_global(type(self.app), 'time', types.SimpleNamespace(sleep=lambda _: None)):
            self.app._on_pulse_server_restarted()
        self.app._write_recovery_result.assert_called_once_with(False, 'pulse_restart')
        self.exit.assert_called_once_with(1)
        self.tick(60.0)
        self.exit.assert_called_once_with(1)
        self.assertFalse(self.sd.terminated)

    def test_stalled_cancel_alone_restarts_at_its_deadline(self):
        self.tick(29.9)
        self.exit.assert_not_called()
        self.tick(30.0)
        self.exit.assert_called_once_with(1)

    def test_deadline_waits_for_processing_and_the_stop_handoff(self):
        for busy in ('is_recording', 'is_processing', '_recording_starting',
                     '_file_transcription_active', '_longform_active', '_model_operation_active'):
            with self.subTest(busy=busy):
                setattr(self.app, busy, True)
                self.tick(60.0)
                self.exit.assert_not_called()
                setattr(self.app, busy, False)
        self.app._recording_finalizing.set()
        self.tick(60.0)
        self.exit.assert_not_called()
        self.app._recording_finalizing.clear()
        self.tick(60.0)
        self.exit.assert_called_once_with(1)

    def test_teardown_finishing_while_processing_cancels_escalation(self):
        self.app.is_processing = True
        self.tick(60.0)
        self.release.set()
        self.capture.record_thread.join(timeout=2.0)
        self.assertFalse(self.capture.record_thread.is_alive())
        self.app.is_processing = False
        self.tick(90.0)
        self.exit.assert_not_called()
        self.assertFalse(self.capture._has_pending_teardowns())
        self.assertTrue(self.capture._cleanup_complete.is_set())

    def test_non_systemd_reports_once_and_late_completion_still_unblocks(self):
        with mock.patch('instance_detection.is_running_under_systemd', return_value=False), \
                patch_app_global(type(self.app), 'log', mock.Mock()) as log:
            self.tick(30.0)
            self.tick(60.0)
            log.assert_called_once()
        self.exit.assert_not_called()
        self.release.set()
        self.capture.record_thread.join(timeout=2.0)
        self.assertFalse(self.capture._has_pending_teardowns())
        self.assertTrue(self.capture._cleanup_complete.is_set())

    def test_pending_start_advice_is_transient(self):
        self.assertEqual(self.app._mic_failure_message('unplug and replug'),
                         'Still releasing the microphone - try again in a moment')



class MidRecordingWedgeTests(unittest.TestCase):
    """A pulse restart mid-recording must not leave the app recording nothing (#209)."""

    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def setUp(self):
        fixture = capture_tests.StreamOrphanRecoveryTests()
        self.addCleanup(fixture.tearDown)
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.sd = capture_tests.FakeSoundDevice()
        self.module = fixture._load_audio_capture(self.sd)
        self.capture = fixture._start_recording(self.module, self.sd)
        fixture._fast_teardowns(self.capture)
        self.release = threading.Event()
        self.addCleanup(lambda: (self.release.set(), fixture._stop_and_join(self.capture)))

        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        self.app = app
        app.audio_capture = self.capture
        app._recording_lock = threading.Lock()
        app._recording_finalizing = threading.Event()
        for name in ('is_processing', '_recording_starting', '_file_transcription_active',
                     '_longform_active', '_model_operation_active'):
            setattr(app, name, False)
        app.is_recording = True
        app._recording_session = object()
        app._current_language_override = None
        app._ptt_reset = mock.Mock()
        app._cleanup_recording_state = mock.Mock()
        app._wait_for_start_settled = mock.Mock()
        app.whisper_manager = mock.Mock()
        app.audio_manager = mock.Mock()
        app._notify_user = mock.Mock()
        app._write_recovery_result = mock.Mock()
        self.capture.on_unrecoverable_stream = app._on_unrecoverable_audio_stream
        self.exit = mock.Mock()

        patches = [
            mock.patch('instance_detection.is_running_under_systemd', return_value=True),
            mock.patch('subprocess.run',
                       side_effect=lambda *a, **k: capture_tests.FakeCompleted("alsa_input.test mic")),
            patch_app_global(type(app), 'os', types.SimpleNamespace(_exit=self.exit)),
            patch_app_global(type(app), 'time', types.SimpleNamespace(sleep=lambda _: None)),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_wedged_recovery_ends_session_and_restarts_without_waiting(self):
        self.capture.stream.abort = self.release.wait
        self.app._on_pulse_server_restarted()

        self.assertFalse(self.app.is_recording)
        self.app._cleanup_recording_state.assert_called_once()
        self.app.audio_manager.play_error_sound.assert_called_once_with()
        self.exit.assert_called_once_with(1)
        self.assertFalse(self.sd.terminated)

    def test_recovered_server_still_ends_the_orphaned_session(self):
        self.app._on_pulse_server_restarted()

        self.app._write_recovery_result.assert_called_once_with(True, 'pulse_restart')
        self.assertFalse(self.app.is_recording)
        self.app.audio_manager.play_error_sound.assert_called_once_with()
        self.exit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
