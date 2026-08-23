import contextlib
import importlib.util
import io
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
sys.path.insert(0, str(ROOT / 'lib' / 'src'))

import backend_installer
from cli import models, status


EXPECTED_PYTHONS = (
    '/usr/bin/python3',
    '/usr/bin/python',
    '/bin/python3',
    '/bin/python',
    '/usr/local/bin/python3',
    '/usr/local/bin/python',
)


class InterpreterContractTests(unittest.TestCase):
    def test_all_interpreter_candidate_lists_match(self):
        self.assertEqual(backend_installer.SYSTEM_PYTHON_CANDIDATES, EXPECTED_PYTHONS)
        commands = {
            'bin/hyprwhspr': (
                "source <(sed -n '/^SYSTEM_PYTHON_CANDIDATES=(/,/^)/p' bin/hyprwhspr); "
                "printf '%s\\n' \"${SYSTEM_PYTHON_CANDIDATES[@]}\""
            ),
            'scripts/install-deps.sh': (
                "source scripts/install-deps.sh; "
                "printf '%s\\n' \"${SYSTEM_PYTHON_CANDIDATES[@]}\""
            ),
        }
        for source, command in commands.items():
            result = subprocess.run(
                ['bash', '-c', command], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(tuple(result.stdout.splitlines()), EXPECTED_PYTHONS, source)

    def test_activated_environment_uses_system_python(self):
        versions = {'/usr/bin/python3': (3, 13), '/managed/python': (3, 12)}
        with (
            mock.patch.object(backend_installer, 'SYSTEM_PYTHON_CANDIDATES',
                              ('/usr/bin/python3',)),
            mock.patch.object(backend_installer.os.path, 'isfile', return_value=True),
            mock.patch.object(backend_installer.os, 'access', return_value=True),
            mock.patch.object(backend_installer, '_get_python_version',
                              side_effect=lambda path: versions[path]),
            mock.patch.object(backend_installer.sys, 'executable', '/managed/python'),
            mock.patch.dict(backend_installer.os.environ, {'VIRTUAL_ENV': '/managed'}, clear=False),
        ):
            selected, _ = backend_installer._find_compatible_python()
        self.assertEqual(selected, '/usr/bin/python3')

    def test_over_new_default_falls_back_to_versioned_system_python(self):
        available = {'/usr/bin/python3', '/usr/bin/python3.14'}
        versions = {'/usr/bin/python3': (3, 15), '/usr/bin/python3.14': (3, 14)}
        with (
            mock.patch.object(backend_installer, 'SYSTEM_PYTHON_CANDIDATES',
                              ('/usr/bin/python3',)),
            mock.patch.object(backend_installer.os.path, 'isfile',
                              side_effect=lambda path: path in available),
            mock.patch.object(backend_installer.os, 'access', return_value=True),
            mock.patch.object(backend_installer, '_get_python_version',
                              side_effect=lambda path: versions[path]),
        ):
            selected, description = backend_installer._find_compatible_python()
        self.assertEqual(selected, '/usr/bin/python3.14')
        self.assertEqual(description, 'Python 3.14')

    def test_install_deps_is_sourceable(self):
        result = subprocess.run(
            ['bash', '-c', 'source scripts/install-deps.sh; declare -F main >/dev/null'],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('Dependency Installation', result.stdout)


class CohereAndModelCommandTests(unittest.TestCase):
    def test_shared_download_has_no_fixed_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / 'venv'
            (venv / 'bin').mkdir(parents=True)
            (venv / 'bin' / 'python').touch()
            completed = types.SimpleNamespace(returncode=0)
            with (
                mock.patch.object(backend_installer, 'VENV_DIR', venv),
                mock.patch.object(backend_installer, 'run_command', return_value=completed) as run,
            ):
                self.assertEqual(
                    backend_installer.download_cohere_transcribe_model('token'),
                    (True, None),
                )
        self.assertNotIn('timeout', run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs['env']['HF_HUB_DOWNLOAD_TIMEOUT'], '60')

    def test_empty_child_diagnostic_falls_back_to_process_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / 'venv'
            (venv / 'bin').mkdir(parents=True)
            (venv / 'bin' / 'python').touch()
            failure = subprocess.CalledProcessError(137, ['python'])
            with (
                mock.patch.object(backend_installer, 'VENV_DIR', venv),
                mock.patch.object(backend_installer, 'run_command', side_effect=failure),
            ):
                success, diagnostic = backend_installer.download_cohere_transcribe_model()
        self.assertFalse(success)
        self.assertIn('exit status 137', diagnostic)

    def test_manual_cohere_success_clears_failed_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / 'venv'
            (venv / 'bin').mkdir(parents=True)
            (venv / 'bin' / 'python').touch()
            with (
                mock.patch.object(models, 'VENV_DIR', venv),
                mock.patch.object(models, 'get_credential', return_value='token'),
                mock.patch.object(models, '_download_cohere', return_value=(True, None)),
                mock.patch.object(models, 'set_install_state') as set_state,
            ):
                self.assertTrue(models.download_cohere_transcribe_model())
        set_state.assert_called_once_with('completed')

    def test_model_status_absence_is_still_command_success(self):
        config = mock.Mock()
        config.get_setting.return_value = 'cohere-transcribe'
        with (
            mock.patch.object(models, 'ConfigManager', return_value=config),
            mock.patch.object(models, 'cohere_transcribe_model_status') as report,
        ):
            self.assertTrue(models.model_command('status'))
        report.assert_called_once_with(config)

    def test_invalid_backend_action_fails(self):
        config = mock.Mock()
        config.get_setting.return_value = 'cohere-transcribe'
        with mock.patch.object(models, 'ConfigManager', return_value=config):
            self.assertFalse(models.model_command('invalid'))

    def test_remote_backend_model_operation_fails(self):
        config = mock.Mock()
        config.get_setting.return_value = 'rest-api'
        with mock.patch.object(models, 'ConfigManager', return_value=config):
            self.assertFalse(models.model_command('download'))

    def test_failed_model_command_exits_nonzero(self):
        spec = importlib.util.spec_from_file_location(
            'hyprwhspr_cli_entry', ROOT / 'lib' / 'cli.py')
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        original = models.model_command
        models.model_command = lambda *_args: False
        try:
            with mock.patch.object(sys, 'argv', ['hyprwhspr', 'model', 'status']):
                with self.assertRaises(SystemExit) as raised:
                    entry.main()
        finally:
            models.model_command = original
        self.assertEqual(raised.exception.code, 1)

    def test_setup_download_failure_records_failed_and_not_completed(self):
        credential_module = types.SimpleNamespace(
            get_credential=mock.Mock(return_value='token'))
        states = []
        with (
            mock.patch.object(backend_installer, 'HYPRWHSPR_ROOT', str(ROOT)),
            mock.patch.object(backend_installer, 'init_state'),
            mock.patch.object(backend_installer, '_check_mise_active', return_value=False),
            mock.patch.object(backend_installer, 'get_state', return_value=''),
            mock.patch.object(backend_installer, 'execute_dependency_plan'),
            mock.patch.object(backend_installer, 'download_cohere_transcribe_model',
                              return_value=(False, 'original download traceback')),
            mock.patch.object(backend_installer, 'set_install_state',
                              side_effect=lambda state, error=None: states.append((state, error))),
            mock.patch.dict(sys.modules, {'credential_manager': credential_module}),
        ):
            self.assertFalse(backend_installer.install_backend('cohere-transcribe'))
        self.assertIn(('failed', 'original download traceback'), states)
        self.assertNotIn(('completed', None), states)


class InstallationStatusTests(unittest.TestCase):
    def test_persisted_diagnostic_appears_in_status(self):
        inactive = types.SimpleNamespace(returncode=1)
        output = io.StringIO()
        with (
            mock.patch.object(status, 'get_install_state',
                              return_value=('failed', 'numpy.dtype size changed')),
            mock.patch.object(status, 'run_command', return_value=inactive),
            mock.patch.object(status, 'waybar_status'),
            mock.patch.object(status.paths, 'CONFIG_FILE', Path('/does/not/exist')),
            mock.patch.object(status, 'model_status'),
            mock.patch.object(status, 'check_permissions'),
            mock.patch.object(status, 'ConfigManager', side_effect=RuntimeError),
            contextlib.redirect_stdout(output),
        ):
            status.status_command()
        rendered = output.getvalue()
        self.assertIn('[Installation]', rendered)
        self.assertIn('numpy.dtype size changed', rendered)

    def test_long_diagnostic_is_truncated_with_state_path(self):
        inactive = types.SimpleNamespace(returncode=1)
        output = io.StringIO()
        error = '\n'.join(f'line {number}' for number in range(30))
        with (
            mock.patch.object(status, 'get_install_state', return_value=('failed', error)),
            mock.patch.object(status, 'run_command', return_value=inactive),
            mock.patch.object(status, 'waybar_status'),
            mock.patch.object(status.paths, 'CONFIG_FILE', Path('/does/not/exist')),
            mock.patch.object(status, 'model_status'),
            mock.patch.object(status, 'check_permissions'),
            mock.patch.object(status, 'ConfigManager', side_effect=RuntimeError),
            contextlib.redirect_stdout(output),
        ):
            status.status_command()
        rendered = output.getvalue()
        self.assertIn('line 19', rendered)
        self.assertNotIn('line 20', rendered)
        self.assertIn(str(status.STATE_FILE), rendered)


if __name__ == '__main__':
    unittest.main()
