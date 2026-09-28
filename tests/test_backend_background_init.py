"""Slow backends load in the background and only announce readiness when slow."""
import threading
import types
import unittest
from unittest import mock

from tests.test_suspend_resume_recovery import _import_main_isolated


class BackgroundInitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def test_onnx_asr_loads_in_background(self):
        # A first-start model download must not hold shortcuts and the FIFO hostage.
        self.assertIn('onnx-asr', self.main.SLOW_BACKENDS)

    def _init(self, elapsed, ok=True):
        main = self.main
        app = main.hyprwhsprApp.__new__(main.hyprwhsprApp)
        app._backend_init_lock = threading.Lock()
        app._model_initializing = False
        app._backend_init_failed = False
        app.whisper_manager = types.SimpleNamespace(initialize=lambda: ok)
        app._notify_user = mock.Mock()
        # Run the init thread inline so the test can assert on its effects.
        inline = lambda target, **kwargs: types.SimpleNamespace(start=target)
        with mock.patch.object(main.threading, 'Thread', inline), \
             mock.patch.object(main.time, 'monotonic', side_effect=[0, elapsed]), \
             mock.patch('builtins.print'):
            app._start_backend_init_background()
        return app

    def test_ready_notification_only_after_a_noticeable_load(self):
        self.assertFalse(self._init(1)._notify_user.called)
        self.assertTrue(self._init(self.main.READY_NOTIFY_AFTER_S)._notify_user.called)

    def test_failed_init_marks_failure_without_notifying(self):
        app = self._init(30, ok=False)
        self.assertTrue(app._backend_init_failed)
        self.assertFalse(app._model_initializing)
        self.assertFalse(app._notify_user.called)


if __name__ == '__main__':
    unittest.main()
