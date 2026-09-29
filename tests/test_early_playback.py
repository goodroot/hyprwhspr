"""Startup suppression ordering and ownership; external audio is entirely mocked."""
import threading
import types
import unittest
from unittest import mock

from tests.test_suspend_resume_recovery import _import_main_isolated
from tests.text_injector_helpers import ConfigStub


class EarlyPlaybackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def app(self, mode='duck'):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app._recording_lock = threading.Lock()
        app._playback_lock = threading.Lock()
        app._playback_session = None
        app._recording_session = object()
        app._playback_shutdown = False
        app._recording_starting = False
        app._start_settled = threading.Event(); app._start_settled.set(); app._start_owner = None
        app._model_initializing = app._backend_init_failed = False
        app._file_transcription_active = app._model_operation_active = False
        app.is_recording = False
        app.config = ConfigStub({'transcription_backend': 'onnx-asr', 'audio_ducking': True, 'audio_ducking_mode': mode})
        app.whisper_manager = mock.Mock()
        app.whisper_manager._model_manually_unloaded = False
        app.whisper_manager.realtime_client_missing.return_value = False
        app.audio_capture = mock.Mock(lock=threading.Lock(), frames_since_start=1)
        app.audio_capture.start_recording.return_value = True
        app.audio_manager = mock.Mock()
        app.playback_suppressor = mock.Mock(is_active=True)
        app._background_recovery_needed = threading.Event()
        for method in ('_clear_mic_osd_preview_text', '_clear_zero_volume_signal', '_write_recording_status', '_show_mic_osd', '_hide_mic_osd', '_stop_audio_level_monitoring', '_start_audio_level_monitoring', '_release_blocked_capture', '_notify_zero_volume', '_notify_user'):
            setattr(app, method, mock.Mock())
        app._mic_failure_message = lambda message: message
        return app

    def test_suppression_precedes_stability_check_and_unstable_restores(self):
        for mode in ('duck', 'pause'):
            for stable in (False, True):
                with self.subTest(mode=mode, stable=stable):
                    app = self.app(mode)
                    order = []
                    app.audio_manager.play_start_sound.side_effect = lambda: order.append('sound')
                    app.playback_suppressor.suppress.side_effect = lambda **kw: order.append('suppress')
                    def sleep(delay):
                        self.assertEqual(delay, .2)
                        order.append('stability')
                        if stable:
                            app.audio_capture.frames_since_start += 1
                    with mock.patch('time.sleep', side_effect=sleep):
                        app._start_recording()
                    self.assertEqual(order, ['sound', 'suppress', 'stability'])
                    self.assertEqual(app.playback_suppressor.restore.called, not stable)
                    self.assertEqual(app.is_recording, stable)

    def test_cancel_during_suppression_restores_and_old_owner_cannot_restore_new(self):
        for mode in ('duck', 'pause'):
            app = self.app(mode)
            app.is_recording = True
            session = app._recording_session
            entered, release = threading.Event(), threading.Event()
            def suppress(**kwargs):
                # External audio calls must not hold the recording state lock.
                self.assertTrue(app._recording_lock.acquire(blocking=False))
                app._recording_lock.release()
                entered.set()
                release.wait(2)
            app.playback_suppressor.suppress.side_effect = suppress
            worker = threading.Thread(target=app._suppress_recording_playback, args=(session,))
            worker.start()
            self.assertTrue(entered.wait(2))
            with app._recording_lock:
                app.is_recording = False
            release.set()
            worker.join(2)
            self.assertFalse(worker.is_alive())
            app.playback_suppressor.restore.assert_called_once()
            app.is_recording = True
            new_session = app._recording_session = object()
            app.playback_suppressor.suppress.side_effect = None
            self.assertTrue(app._suppress_recording_playback(new_session))
            app._restore_recording_playback(session)
            app._restore_recording_playback(None)
            app.playback_suppressor.restore.assert_called_once()
            app._restore_recording_playback(new_session)
            self.assertEqual(app.playback_suppressor.restore.call_count, 2)

    def test_shutdown_blocks_suppression_and_restores_both_modes(self):
        for mode in ('duck', 'pause'):
            app = self.app(mode)
            app.is_recording = True
            self.assertTrue(app._suppress_recording_playback(app._recording_session))
            app._restore_recording_playback(shutdown=True)
            self.assertFalse(app._suppress_recording_playback(app._recording_session))
            app.playback_suppressor.suppress.assert_called_once()
            app.playback_suppressor.restore.assert_called_once()

    def test_stop_during_capture_start_prevents_late_suppression(self):
        app = self.app()
        def start(**kwargs):
            with app._recording_lock:
                app.is_recording = False
            # A rapid restart is refused until old startup releases capture ownership.
            app._start_recording()
            return True
        app.audio_capture.start_recording.side_effect = start
        app._start_recording()
        app.audio_capture.start_recording.assert_called_once()
        # The stopper owns teardown and the audio; the aborted start leaves capture alone.
        app.audio_capture.stop_recording.assert_not_called()
        app.playback_suppressor.suppress.assert_not_called()
        app._notify_zero_volume.assert_not_called()
        # The refused restart's capture client must not wait forever.
        app._release_blocked_capture.assert_called_once()
        app._notify_user.assert_called_once()
        self.assertFalse(app._recording_starting)

    def test_stop_in_stability_window_is_not_an_unstable_stream(self):
        app = self.app()
        def sleep(delay):
            # Stop lands during the 0.2s check; frames freeze once stop is signalled.
            with app._recording_lock:
                app.is_recording = False
        with mock.patch('time.sleep', side_effect=sleep):
            app._start_recording()
        app._notify_zero_volume.assert_not_called()
        app.audio_capture.stop_recording.assert_not_called()
        self.assertTrue(app._start_settled.is_set())

    def test_stop_waits_for_start_and_alone_stops_capture(self):
        app = self.app()
        entered, release = threading.Event(), threading.Event()
        def suppress(**kwargs):
            entered.set()
            release.wait(2)
        app.playback_suppressor.suppress.side_effect = suppress
        starter = threading.Thread(target=app._start_recording)
        starter.start()
        self.assertTrue(entered.wait(2))
        with app._recording_lock:
            app.is_recording = False
        stopped = threading.Event()
        def stop():
            app._wait_for_start_settled()
            app.audio_capture.stop_recording()
            stopped.set()
        stopper = threading.Thread(target=stop)
        stopper.start()
        self.assertFalse(stopped.wait(0.2))
        release.set()
        starter.join(2)
        stopper.join(2)
        self.assertTrue(stopped.is_set())
        app.audio_capture.stop_recording.assert_called_once()
        app.playback_suppressor.restore.assert_called_once()

    def test_shutdown_during_start_releases_capture(self):
        app = self.app()
        def start(**kwargs):
            app._playback_shutdown = True
            return True
        app.audio_capture.start_recording.side_effect = start
        app._start_recording()
        app.audio_capture.stop_recording.assert_called_once()
