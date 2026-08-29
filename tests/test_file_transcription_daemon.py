import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests.test_suspend_resume_recovery import _import_main_isolated


class FakeConfig:
    def __init__(self, backend='pywhispercpp'):
        self.backend = backend

    def get_setting(self, key, default=None):
        if key == 'transcription_backend':
            return self.backend
        return default


class FileTranscriptionDaemonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def _app(self, backend='pywhispercpp'):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app.config = FakeConfig(backend)
        app._recording_lock = threading.Lock()
        app._recording_finalizing = threading.Event()
        app.is_recording = False
        app.is_processing = False
        app._file_transcription_active = False
        app._longform_active = False
        app._model_operation_active = False
        app._model_initializing = False
        app.whisper_manager = types.SimpleNamespace(
            is_ready=lambda: True,
            transcribe_audio=mock.Mock(return_value=' raw transcript '),
        )
        return app

    def test_idle_daemon_transcribes_and_optionally_cleans(self):
        app = self._app()
        with tempfile.TemporaryDirectory() as tempdir, \
                mock.patch.object(self.main, 'MODEL_UNLOADED_FILE', Path(tempdir) / 'unloaded'), \
                mock.patch.object(self.main, 'decode_audio_file', return_value=(
                    np.array([0.1, 0.2], dtype=np.float32), 44100
                )), mock.patch.object(
                    self.main, 'preprocess_text', return_value='clean transcript'
                ) as preprocess:
            self.assertEqual(
                app._handle_file_transcribe('/tmp/audio.wav', 'fr', True),
                (True, 'clean transcript'),
            )
        app.whisper_manager.transcribe_audio.assert_called_once()
        self.assertEqual(
            app.whisper_manager.transcribe_audio.call_args.kwargs,
            {'sample_rate': 44100, 'language_override': 'fr'},
        )
        preprocess.assert_called_once_with('raw transcript', app.config)
        self.assertFalse(app.is_processing)
        self.assertFalse(app._file_transcription_active)

    def test_file_cleanup_does_not_clear_longform_processing_owner(self):
        app = self._app()
        app.whisper_manager.transcribe_audio.side_effect = lambda *_args, **_kwargs: (
            setattr(app, 'is_processing', True) or 'transcript'
        )
        with tempfile.TemporaryDirectory() as tempdir, \
                mock.patch.object(self.main, 'MODEL_UNLOADED_FILE', Path(tempdir) / 'unloaded'), \
                mock.patch.object(self.main, 'decode_audio_file', return_value=(
                    np.array([0.1], dtype=np.float32), 16000
                )):
            self.assertEqual(
                app._handle_file_transcribe('/tmp/audio.wav'),
                (True, 'transcript'),
            )
        self.assertTrue(app.is_processing)
        self.assertFalse(app._file_transcription_active)

    def test_decoder_dependency_error_returns_a_response_and_releases_owner(self):
        app = self._app()
        with tempfile.TemporaryDirectory() as tempdir, \
                mock.patch.object(self.main, 'MODEL_UNLOADED_FILE', Path(tempdir) / 'unloaded'), \
                mock.patch.object(
                    self.main, 'decode_audio_file',
                    side_effect=self.main.AudioFileError(
                        'Audio file decoding requires soundfile; run: hyprwhspr setup'
                    ),
                ):
            ok, error = app._handle_file_transcribe('/tmp/audio.wav')
        self.assertFalse(ok)
        self.assertIn('hyprwhspr setup', error)
        self.assertFalse(app._file_transcription_active)

    def test_model_lifecycle_is_blocked_during_file_transcription(self):
        app = self._app()
        app._file_transcription_active = True
        app.whisper_manager.unload_model = mock.Mock()
        app.whisper_manager.reload_model = mock.Mock()
        app._notify_user = mock.Mock()
        app._handle_control_command('model_unload')
        app._handle_control_command('model_reload')
        app.whisper_manager.unload_model.assert_not_called()
        app.whisper_manager.reload_model.assert_not_called()

    def test_blocked_start_releases_a_waiting_capture_client(self):
        app = self._app()
        app._backend_init_failed = False
        app._file_transcription_active = True
        app._notify_user = mock.Mock()
        app._notify_capture = mock.Mock()
        app._recording_control_server = types.SimpleNamespace(
            has_capture_subscriber=lambda: True
        )

        app._start_recording()

        app._notify_capture.assert_called_once_with("", final=True)

    def test_longform_and_model_operations_block_file_requests(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with mock.patch.object(self.main, 'MODEL_UNLOADED_FILE', Path(tempdir) / 'unloaded'):
                app = self._app()
                app._longform_active = True
                self.assertIn('long-form', app._handle_file_transcribe('x.wav')[1])
                app._longform_active = False
                app._model_operation_active = True
                self.assertIn('loading', app._handle_file_transcribe('x.wav')[1])

    def test_a_second_concurrent_file_request_is_rejected(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with mock.patch.object(self.main, 'MODEL_UNLOADED_FILE', Path(tempdir) / 'unloaded'):
                app = self._app()
                started = threading.Event()
                release = threading.Event()
                second = {}

                def slow_transcribe(*_args, **_kwargs):
                    started.set()
                    release.wait(5)
                    return 'first transcript'

                app.whisper_manager.transcribe_audio = slow_transcribe
                with mock.patch.object(self.main, 'decode_audio_file', return_value=(
                        np.array([0.1], dtype=np.float32), 16000)):
                    worker = threading.Thread(
                        target=lambda: second.setdefault(
                            'first', app._handle_file_transcribe('/tmp/a.wav')
                        )
                    )
                    worker.start()
                    self.assertTrue(started.wait(5))
                    ok, error = app._handle_file_transcribe('/tmp/b.wav')
                    release.set()
                    worker.join(5)

        self.assertFalse(ok)
        self.assertIn('already running', error)
        self.assertEqual(second['first'], (True, 'first transcript'))
        self.assertFalse(app._file_transcription_active)

    def test_stop_recording_failure_clears_the_finalizing_gate(self):
        app = self._app()
        app.is_recording = True
        # Raised from the first step after the gate is set; before the fix this
        # ran outside the try and wedged every later file/model request
        app._autostop_stop_silence_monitor = mock.Mock(side_effect=RuntimeError('boom'))
        app._clear_mic_osd_preview_text = mock.Mock()
        app._set_visualizer_state = mock.Mock()
        app._stop_audio_level_monitoring = mock.Mock()
        app._write_recording_status = mock.Mock()
        app._show_result_and_hide = mock.Mock()
        app._continuous_stop_silence_monitor = mock.Mock()
        app.whisper_manager.close_realtime_connection = mock.Mock()
        app.playback_suppressor = types.SimpleNamespace(is_active=False)
        app._notify_capture = mock.Mock()
        app._recording_control_server = types.SimpleNamespace(
            has_capture_subscriber=lambda: False,
            is_trace_capture=lambda: False,
            notify_capture=mock.Mock(),
        )

        app._stop_recording()

        self.assertFalse(app._recording_finalizing.is_set())

    def test_model_operation_frees_the_recording_lock_while_loading(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with mock.patch.object(self.main, 'MODEL_UNLOADED_FILE', Path(tempdir) / 'unloaded'):
                app = self._app()
                app._notify_user = mock.Mock()
                observed = {}

                def slow_reload():
                    observed['lock_free'] = app._recording_lock.acquire(blocking=False)
                    if observed['lock_free']:
                        app._recording_lock.release()
                    observed['claimed'] = app._model_operation_active
                    return True

                app.whisper_manager.reload_model = slow_reload
                app._handle_model_operation('reload')

        self.assertTrue(observed['lock_free'])
        self.assertTrue(observed['claimed'])
        self.assertFalse(app._model_operation_active)

    def test_regular_processing_does_not_trigger_file_only_start_gate(self):
        app = self._app()
        app.is_processing = True
        app._backend_init_failed = False
        app.whisper_manager._model_manually_unloaded = False
        app.whisper_manager.realtime_client_missing = lambda: False
        app.whisper_manager.close_realtime_connection = mock.Mock()
        app._clear_mic_osd_preview_text = mock.Mock()
        app._clear_zero_volume_signal = mock.Mock(
            side_effect=RuntimeError('stop after gate')
        )
        app._hide_mic_osd = mock.Mock()
        app._stop_audio_level_monitoring = mock.Mock()
        app._write_recording_status = mock.Mock()
        app.playback_suppressor = types.SimpleNamespace(is_active=False)
        app._recording_control_server = types.SimpleNamespace(
            has_capture_subscriber=lambda: False
        )
        app._start_recording()
        app._clear_mic_osd_preview_text.assert_called_once()

        app.is_recording = False
        app._file_transcription_active = True
        app._notify_user = mock.Mock()
        app._clear_mic_osd_preview_text.reset_mock()
        app._start_recording()
        app._clear_mic_osd_preview_text.assert_not_called()

    def test_busy_initializing_unloaded_and_realtime_states_are_rejected(self):
        with tempfile.TemporaryDirectory() as tempdir:
            unloaded = Path(tempdir) / 'unloaded'
            app = self._app()
            with mock.patch.object(self.main, 'MODEL_UNLOADED_FILE', unloaded):
                app.is_recording = True
                self.assertIn('while recording', app._handle_file_transcribe('x.wav')[1])
                app.is_recording = False
                app.is_processing = True
                self.assertIn('already processing', app._handle_file_transcribe('x.wav')[1])
                app.is_processing = False
                app._recording_finalizing.set()
                self.assertIn('finalizing', app._handle_file_transcribe('x.wav')[1])
                app._recording_finalizing.clear()
                app._model_initializing = True
                self.assertIn('initializing', app._handle_file_transcribe('x.wav')[1])
                app._model_initializing = False
                unloaded.touch()
                self.assertIn('not loaded', app._handle_file_transcribe('x.wav')[1])

            realtime = self._app('realtime-ws')
            self.assertIn('live capture only', realtime._handle_file_transcribe('x.wav')[1])


if __name__ == '__main__':
    unittest.main()
