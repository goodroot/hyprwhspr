"""The service reports missing Python packages at start."""
import ast
import contextlib
import io
import unittest
from pathlib import Path
from unittest import mock

from tests.test_suspend_resume_recovery import FakeConfig, _import_main_isolated, patch_app_global


ROOT = Path(__file__).resolve().parents[1]


class StartupDependencyCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def _report(self, missing, config=None):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app.config = FakeConfig(config or {'transcription_backend': 'vulkan'})
        app._notify_user = mock.Mock()
        finder = mock.Mock(return_value=missing)
        output = io.StringIO()
        with patch_app_global(self.main.hyprwhsprApp, 'missing_imports', finder), \
                contextlib.redirect_stdout(output):
            result = app._report_missing_dependencies()
        return app, finder, output.getvalue(), result

    def test_missing_package_is_logged_with_the_fix(self):
        _app, _finder, logged, result = self._report(('soxr',))
        self.assertEqual(result, ('soxr',))
        self.assertIn('[ERROR] Missing Python modules: soxr', logged)
        self.assertIn('hyprwhspr setup (Reinstall backend: yes)', logged)

    def test_missing_package_raises_a_persistent_notification(self):
        app, _finder, _logged, _result = self._report(('soxr', 'soundfile'))
        (title, message), kwargs = app._notify_user.call_args
        self.assertEqual(kwargs, {'urgency': 'critical'})
        self.assertIn('Missing Python modules: soxr, soundfile', message)
        self.assertIn('hyprwhspr setup', message)
        self.assertIn('reinstall the backend', message)

    def test_complete_environment_stays_silent(self):
        app, _finder, logged, result = self._report(())
        self.assertEqual(result, ())
        self.assertEqual(logged, '')
        self.assertFalse(app._notify_user.called)

    def test_checks_the_configured_backend_and_provider(self):
        _app, finder, _logged, _result = self._report((), {
            'transcription_backend': 'realtime-ws', 'websocket_provider': 'elevenlabs'})
        finder.assert_called_once_with('realtime-ws', 'elevenlabs')

    def test_run_reports_before_initializing_the_backend(self):
        tree = ast.parse((ROOT / 'lib' / 'main.py').read_text(encoding='utf-8'))
        run = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == 'run')
        lines = {}
        for node in ast.walk(run):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                lines.setdefault(node.func.attr, node.lineno)
        self.assertIn('_report_missing_dependencies', lines)
        for init in ('initialize', '_start_backend_init_background'):
            self.assertLess(lines['_report_missing_dependencies'], lines[init])


if __name__ == '__main__':
    unittest.main()
