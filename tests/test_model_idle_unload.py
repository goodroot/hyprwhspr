"""model_idle_unload_minutes: an idle local model leaves memory and the next use reloads it (#302)."""
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

from tests.test_suspend_resume_recovery import _import_main_isolated, patch_app_global


class ModelIdleUnloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()
        cls.app_cls = cls.main.hyprwhsprApp

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.marker = Path(temp.name) / 'model_unloaded'
        for name, value in (('MODEL_UNLOADED_FILE', self.marker), ('log', mock.Mock())):
            patcher = patch_app_global(self.app_cls, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _app(self, minutes=10, local=True):
        settings = {'model_idle_unload_minutes': minutes}
        app = self.app_cls.__new__(self.app_cls)
        app.config = types.SimpleNamespace(get_setting=lambda key, default=None: settings.get(key, default))
        app._recording_lock = threading.Lock()
        app._recording_finalizing = threading.Event()
        app.is_recording = app.is_processing = False
        app._file_transcription_active = app._longform_active = False
        app._model_operation_active = app._model_initializing = False
        app._backend_init_failed = False
        app._model_idle_since = None
        app._model_idle_unloaded = app._model_waking = False
        app._model_wake_done = threading.Event()
        app._model_wake_done.set()
        app._notify_user = mock.Mock()
        loaded = {'ready': True}

        def unload():
            loaded['ready'] = False
            return True

        def reload():
            loaded['ready'] = True
            return True

        app.whisper_manager = types.SimpleNamespace(
            is_ready=lambda: loaded['ready'],
            active_backend_is_local=lambda: local,
            unload_model=mock.Mock(side_effect=unload),
            reload_model=mock.Mock(side_effect=reload),
            realtime_client_missing=lambda: False,
            realtime_busy=lambda: False,
            close_realtime_connection=mock.Mock(),
            _model_manually_unloaded=False,
        )
        return app

    def _press(self, app):
        """Run _start_recording through its gate; True if the recording was accepted."""
        app.text_injector = mock.Mock()
        app.text_injector.stream_busy.return_value = False
        app._recording_starting = False
        app._start_settled = threading.Event()
        app._start_settled.set()
        app._start_owner = None
        app._ptt_press_time = None
        app._ptt_locked = False
        app._recording_session = None
        app._playback_lock = threading.Lock()
        app._playback_session = None
        app._playback_shutdown = False
        app._release_blocked_capture = mock.Mock()
        app._clear_mic_osd_preview_text = mock.Mock()
        app._clear_zero_volume_signal = mock.Mock(side_effect=RuntimeError('stop after gate'))
        app._hide_mic_osd = mock.Mock()
        app._stop_audio_level_monitoring = mock.Mock()
        app._write_recording_status = mock.Mock()
        app.playback_suppressor = types.SimpleNamespace(is_active=False)
        app._recording_control_server = types.SimpleNamespace(has_capture_subscriber=lambda: False)
        app.is_recording = False
        app._start_recording()
        return app._clear_mic_osd_preview_text.called

    def _join_wake(self):
        for thread in threading.enumerate():
            if thread.name == 'ModelWake':
                thread.join(timeout=5)

    def test_unloads_only_after_the_idle_time(self):
        app = self._app(minutes=10)
        self.assertFalse(app._unload_model_if_idle(now=1000))        # first sighting starts the clock
        self.assertFalse(app._unload_model_if_idle(now=1000 + 599))
        app.whisper_manager.unload_model.assert_not_called()
        self.assertTrue(app._unload_model_if_idle(now=1000 + 600))
        app.whisper_manager.unload_model.assert_called_once_with()
        self.assertTrue(self.marker.exists())                        # tray shows the sleep icon
        self.assertTrue(app._model_idle_unloaded)
        app._notify_user.assert_not_called()                         # nobody is watching; stay quiet
        self.assertFalse(app._unload_model_if_idle(now=5000))        # nothing left to unload
        app.whisper_manager.unload_model.assert_called_once_with()

    def test_use_restarts_the_clock(self):
        for flag in ('is_recording', 'is_processing', '_file_transcription_active',
                     '_longform_active', '_model_operation_active', '_model_initializing'):
            with self.subTest(flag=flag):
                app = self._app(minutes=1)
                app._unload_model_if_idle(now=0)
                setattr(app, flag, True)
                self.assertFalse(app._unload_model_if_idle(now=500))
                setattr(app, flag, False)
                self.assertFalse(app._unload_model_if_idle(now=559))
                app.whisper_manager.unload_model.assert_not_called()
                self.assertTrue(app._unload_model_if_idle(now=560))

    def test_off_remote_and_bad_values_never_unload(self):
        for kwargs in ({'minutes': 0}, {'minutes': -5}, {'minutes': 'soon'}, {'minutes': None},
                       {'minutes': float('nan')}, {'minutes': 1, 'local': False}):
            with self.subTest(**kwargs):
                app = self._app(**kwargs)
                app._unload_model_if_idle(now=0)
                self.assertFalse(app._unload_model_if_idle(now=10**6))
                app.whisper_manager.unload_model.assert_not_called()

    def test_a_failing_check_never_reaches_the_main_loop(self):
        app = self._app()
        app.whisper_manager.is_ready = mock.Mock(side_effect=RuntimeError('boom'))
        self.assertFalse(app._unload_model_if_idle(now=0))

    def _idle_unloaded_app(self):
        app = self._app(minutes=1)
        app._unload_model_if_idle(now=0)
        self.assertTrue(app._unload_model_if_idle(now=60))
        return app

    def test_wake_reloads_once_and_clears_the_marker(self):
        app = self._idle_unloaded_app()
        gate = threading.Event()
        reload = app.whisper_manager.reload_model.side_effect

        def slow_reload():
            gate.wait(timeout=5)
            return reload()

        app.whisper_manager.reload_model.side_effect = slow_reload
        self.assertTrue(app._wake_idle_model())
        self.assertTrue(app._wake_idle_model())      # a second press joins the load in flight
        gate.set()
        self._join_wake()
        app.whisper_manager.reload_model.assert_called_once_with()
        self.assertFalse(self.marker.exists())
        self.assertFalse(app._model_idle_unloaded)
        self.assertFalse(app._model_waking)
        self.assertFalse(app._wake_idle_model())     # loaded: nothing to wake

    def test_an_unload_by_hand_is_never_woken(self):
        app = self._app()
        app._handle_model_operation('unload')
        self.assertTrue(self.marker.exists())
        self.assertFalse(app._model_idle_unloaded)
        self.assertFalse(app._wake_idle_model())
        app.whisper_manager.reload_model.assert_not_called()

        # Even when the idle timer got there first
        app = self._idle_unloaded_app()
        app._handle_model_operation('unload')
        self.assertFalse(app._wake_idle_model())

    def _slow_reload(self, app, result=True):
        gate = threading.Event()
        reload = app.whisper_manager.reload_model.side_effect

        def slow():
            gate.wait(timeout=5)
            return reload() if result else False

        app.whisper_manager.reload_model.side_effect = slow
        return gate

    def test_first_press_records_while_the_model_loads(self):
        app = self._idle_unloaded_app()
        app.whisper_manager._model_manually_unloaded = True
        gate = self._slow_reload(app)
        self.assertTrue(self._press(app))            # accepted, not refused
        self.assertTrue(app._model_operation_active)
        self.assertFalse(app._model_wake_done.is_set())
        self.assertTrue(self._press(app))            # a second press mid-load records too
        app._notify_user.assert_not_called()

        waited = []
        waiter = threading.Thread(target=lambda: (app._await_model_wake(), waited.append(True)))
        waiter.start()
        waiter.join(timeout=0.2)
        self.assertEqual(waited, [])                 # the transcription holds for the load
        gate.set()
        waiter.join(timeout=5)
        self._join_wake()
        self.assertEqual(waited, [True])
        app.whisper_manager.reload_model.assert_called_once_with()
        self.assertFalse(self.marker.exists())
        self.assertFalse(app._model_idle_unloaded)
        self.assertFalse(app._model_operation_active)
        self.assertFalse(app._model_waking)
        app._notify_user.assert_not_called()         # the text arriving is the announcement

    def test_failed_wake_says_so_and_the_next_press_tries_again(self):
        app = self._idle_unloaded_app()
        app.whisper_manager._model_manually_unloaded = True
        gate = self._slow_reload(app, result=False)
        self.assertTrue(self._press(app))
        gate.set()
        self._join_wake()
        self.assertTrue(app._model_wake_done.is_set())   # waiters are released either way
        self.assertIn('failed', app._notify_user.call_args.args[1])
        self.assertTrue(app._model_idle_unloaded)
        self.assertFalse(app._model_operation_active)

        app.whisper_manager.reload_model.side_effect = lambda: True
        self.assertTrue(self._press(app))
        self._join_wake()
        self.assertEqual(app.whisper_manager.reload_model.call_count, 2)
        self.assertFalse(app._model_idle_unloaded)

    def test_press_is_still_refused_after_an_unload_by_hand(self):
        app = self._app()
        app._handle_model_operation('unload')
        app.whisper_manager._model_manually_unloaded = True
        app._notify_user.reset_mock()
        self.assertFalse(self._press(app))
        self.assertIn('hyprwhspr model reload', app._notify_user.call_args.args[1])
        app.whisper_manager.reload_model.assert_not_called()

    def test_press_is_refused_during_a_load_that_is_not_a_wake(self):
        app = self._app()
        app._model_operation_active = True
        self.assertFalse(self._press(app))
        self.assertIn('Model load in progress', app._notify_user.call_args.args[1])

    def test_file_transcription_and_long_form_wake_the_model(self):
        app = self._idle_unloaded_app()
        ok, message = app._handle_file_transcribe('/tmp/audio.wav')
        self._join_wake()
        self.assertFalse(ok)
        self.assertIn('loading', message)
        app.whisper_manager.reload_model.assert_called_once_with()

        app = self._idle_unloaded_app()
        self.assertFalse(app._claim_longform_recording())
        self._join_wake()
        self.assertFalse(app._longform_active)
        app.whisper_manager.reload_model.assert_called_once_with()
        self.assertTrue(app._claim_longform_recording())   # loaded now

        app = self._app()                                  # unloaded by hand: old answer, no load
        app._handle_model_operation('unload')
        ok, message = app._handle_file_transcribe('/tmp/audio.wav')
        self.assertIn('hyprwhspr model reload', message)
        app.whisper_manager.reload_model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
