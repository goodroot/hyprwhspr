"""The service reports missing Python packages at start."""
import ast
import contextlib
import io
import os
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
                mock.patch.object(self.main.hyprwhsprApp, '_dependencies_changed_since_setup',
                                  return_value=False), \
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

    def _changed(self, backend, state, fingerprints, env=None):
        """Run the drift check with fake state and one plan fingerprint per variant."""
        plans = {variant: mock.Mock(fingerprint=fp, family=family)
                 for variant, (fp, family) in fingerprints.items()}
        with patch_app_global(self.main.hyprwhsprApp, 'get_state', lambda key: state.get(key)), \
                patch_app_global(self.main.hyprwhsprApp, 'resolve_dependency_plan',
                                 lambda b, p, variant: plans[variant]), \
                mock.patch.dict('os.environ', env or {}, clear=False):
            if not env:
                os.environ.pop('HYPRWHSPR_GENERATION', None)
            return self.main.hyprwhsprApp._dependencies_changed_since_setup(backend, None)

    def test_stale_venv_after_an_update_is_detected(self):
        state = {'dependency_plan_fingerprint': 'old', 'dependency_family': 'pywhispercpp'}
        self.assertTrue(self._changed('vulkan', state, {None: ('new', 'pywhispercpp')}))

    def test_synced_venv_is_not_reported(self):
        state = {'dependency_plan_fingerprint': 'same', 'dependency_family': 'pywhispercpp'}
        self.assertFalse(self._changed('vulkan', state, {None: ('same', 'pywhispercpp')}))

    def test_gpu_manifest_counts_as_synced(self):
        state = {'dependency_plan_fingerprint': 'gpu', 'dependency_family': 'onnx'}
        self.assertFalse(self._changed('onnx-asr', state, {None: ('cpu', 'onnx'), 'gpu': ('gpu', 'onnx')}))

    def test_backend_switched_without_setup_is_not_drift(self):
        state = {'dependency_plan_fingerprint': 'old', 'dependency_family': 'onnx'}
        self.assertFalse(self._changed('vulkan', state, {None: ('new', 'pywhispercpp')}))

    def test_no_recorded_plan_or_managed_release_is_silent(self):
        plans = {None: ('new', 'pywhispercpp')}
        self.assertFalse(self._changed('vulkan', {}, plans))
        state = {'dependency_plan_fingerprint': 'old', 'dependency_family': 'pywhispercpp'}
        self.assertFalse(self._changed('vulkan', state, plans, env={'HYPRWHSPR_GENERATION': '{}'}))

    def test_drift_is_logged_without_a_notification(self):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app.config = FakeConfig({'transcription_backend': 'vulkan'})
        app._notify_user = mock.Mock()
        output = io.StringIO()
        with patch_app_global(self.main.hyprwhsprApp, 'missing_imports', mock.Mock(return_value=())), \
                mock.patch.object(self.main.hyprwhsprApp, '_dependencies_changed_since_setup',
                                  return_value=True), \
                contextlib.redirect_stdout(output):
            app._report_missing_dependencies()
        self.assertIn('[WARN] Python dependencies changed since setup', output.getvalue())
        self.assertFalse(app._notify_user.called)

    def _cpu_report(self, backend, installed):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app.config = FakeConfig({'transcription_backend': backend})
        app._notify_user = mock.Mock()
        state = {'installed_backend': installed}
        output = io.StringIO()
        with patch_app_global(self.main.hyprwhsprApp, 'get_state', lambda key: state.get(key, '')), \
                contextlib.redirect_stdout(output):
            result = app._report_cpu_only_build()
        return app, output.getvalue(), result

    def test_gpu_backend_on_a_cpu_build_is_reported(self):
        for backend, shown in (('vulkan', 'vulkan'), ('amd', 'vulkan'), ('nvidia', 'nvidia')):
            with self.subTest(backend=backend):
                app, logged, result = self._cpu_report(backend, 'cpu')
                self.assertTrue(result)
                self.assertIn(f'[WARN] {shown} is configured, but the installed whisper.cpp build is CPU-only', logged)
                (_title, message), kwargs = app._notify_user.call_args
                self.assertEqual(kwargs, {'urgency': 'critical'})
                self.assertIn('hyprwhspr setup', message)
                self.assertIn('reinstall the backend', message)
                self.assertIn('choose CPU', message)
                self.assertIn('or choose CPU', logged)

    def test_matching_or_unknown_build_stays_silent(self):
        for backend, installed in (('vulkan', 'vulkan'), ('nvidia', 'nvidia'), ('cpu', 'cpu'),
                                   ('vulkan', ''), ('onnx-asr', 'cpu')):
            with self.subTest(backend=backend, installed=installed):
                app, logged, result = self._cpu_report(backend, installed)
                self.assertFalse(result)
                self.assertEqual(logged, '')
                self.assertFalse(app._notify_user.called)

    def test_run_reports_cpu_build_before_initializing_the_backend(self):
        tree = ast.parse((ROOT / 'lib' / 'main.py').read_text(encoding='utf-8'))
        run = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == 'run')
        lines = {}
        for node in ast.walk(run):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                lines.setdefault(node.func.attr, node.lineno)
        self.assertIn('_report_cpu_only_build', lines)
        for init in ('initialize', '_start_backend_init_background'):
            self.assertLess(lines['_report_cpu_only_build'], lines[init])

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
