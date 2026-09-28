"""Slow backends load in the background and only announce readiness when slow."""
import contextlib
import io
import threading
import types
import unittest
from unittest import mock

from tests.test_suspend_resume_recovery import _import_main_isolated


class BackgroundInitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def _app(self):
        main = self.main
        app = main.hyprwhsprApp.__new__(main.hyprwhsprApp)
        app._backend_init_lock = threading.Lock()
        app._recording_lock = threading.Lock()
        app._model_initializing = False
        app._backend_init_failed = False
        app._notify_when_ready = False
        app._file_transcription_active = False
        app._model_operation_active = False
        app._longform_active = False
        return app

    def _init(self, elapsed, ok=True, waiting=False):
        main = self.main
        app = self._app()
        app._notify_when_ready = waiting
        app.whisper_manager = types.SimpleNamespace(initialize=lambda: ok)
        app._notify_user = mock.Mock()
        # Run the init thread inline so the test can assert on its effects.
        inline = lambda target, **kwargs: types.SimpleNamespace(start=target)
        with mock.patch.object(main.threading, 'Thread', inline), \
             mock.patch.object(main.time, 'monotonic', side_effect=[0, elapsed]), \
             contextlib.redirect_stdout(io.StringIO()):
            app._start_backend_init_background()
        return app

    def test_ready_notification_only_after_a_noticeable_load(self):
        self.assertFalse(self._init(1)._notify_user.called)
        self.assertTrue(self._init(self.main.READY_NOTIFY_AFTER_S)._notify_user.called)

    def test_ready_is_announced_to_anyone_told_to_wait_even_when_quick(self):
        app = self._init(1, waiting=True)
        self.assertTrue(app._notify_user.called)
        self.assertFalse(app._notify_when_ready)

    def test_longform_cannot_start_before_the_backend_loads(self):
        for loading, failed in ((True, False), (False, True)):
            app = self._app()
            app._model_initializing, app._backend_init_failed = loading, failed
            app._start_backend_init_background = mock.Mock()
            self.assertFalse(app._claim_longform_recording())
            self.assertFalse(app._longform_active)
            self.assertTrue(app._notify_when_ready)
            # A failed init is retried, as for a refused normal start.
            self.assertEqual(app._start_backend_init_background.called, failed)
        app = self._app()
        self.assertTrue(app._claim_longform_recording())
        self.assertTrue(app._longform_active)

    def test_failed_init_marks_failure_without_notifying(self):
        app = self._init(30, ok=False)
        self.assertTrue(app._backend_init_failed)
        self.assertFalse(app._model_initializing)
        self.assertFalse(app._notify_user.called)


if __name__ == '__main__':
    unittest.main()
