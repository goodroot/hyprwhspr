import io
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
sys.path.insert(0, str(ROOT / 'lib' / 'src'))

from cli import maintenance


class MaintenanceDependencyRepairTests(unittest.TestCase):
    def test_realtime_repair_uses_provider_plan_transaction(self):
        config = mock.Mock()
        config.get_setting.return_value = 'elevenlabs'
        plan = object()
        with (
            mock.patch.object(maintenance, 'resolve_dependency_plan', return_value=plan) as resolve,
            mock.patch.object(maintenance, 'execute_dependency_plan') as execute,
        ):
            self.assertTrue(maintenance._reinstall_configured_dependencies(
                config, 'realtime-ws'))
        resolve.assert_called_once_with('realtime-ws', 'elevenlabs')
        execute.assert_called_once_with(plan, force_rebuild=True)

    def test_default_pywhispercpp_repair_uses_recorded_local_variant(self):
        config = mock.Mock()
        with (
            mock.patch.object(maintenance, 'get_state', return_value='nvidia'),
            mock.patch.object(maintenance, 'install_backend', return_value=True) as install,
        ):
            self.assertTrue(maintenance._reinstall_configured_dependencies(
                config, 'pywhispercpp'))
        install.assert_called_once_with('nvidia', force_rebuild=True)

    def test_corrupt_venv_repair_does_not_delete_before_transaction(self):
        config = mock.Mock()
        config.get_setting.return_value = 'rest-api'
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / 'venv'
            venv.mkdir()
            marker = venv / 'corrupt-marker'
            marker.touch()
            with (
                mock.patch.object(maintenance, 'VENV_DIR', venv),
                mock.patch.object(maintenance, 'ConfigManager', return_value=config),
                mock.patch.object(maintenance.Prompt, 'ask', return_value='1'),
                mock.patch.object(maintenance, '_reinstall_configured_dependencies', return_value=True) as repair,
                mock.patch.object(maintenance.shutil, 'rmtree') as rmtree,
            ):
                self.assertTrue(maintenance.backend_repair_command())
            repair.assert_called_once_with(config, 'rest-api')
            rmtree.assert_not_called()
            self.assertTrue(marker.exists())

    def test_missing_cloud_imports_are_detected_and_repaired(self):
        config = mock.Mock()
        config.get_setting.side_effect = lambda key, default=None: {
            'transcription_backend': 'realtime-ws',
            'websocket_provider': 'elevenlabs',
        }.get(key, default)
        healthy = mock.Mock(returncode=0)
        missing = mock.Mock(returncode=1)
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / 'venv'
            (venv / 'bin').mkdir(parents=True)
            (venv / 'bin' / 'python').touch()
            with (
                mock.patch.object(maintenance, 'VENV_DIR', venv),
                mock.patch.object(maintenance, 'ConfigManager', return_value=config),
                mock.patch.object(maintenance.subprocess, 'run', side_effect=[healthy, missing]),
                mock.patch.object(maintenance.Prompt, 'ask', return_value='1'),
                mock.patch.object(maintenance, '_reinstall_configured_dependencies', return_value=True) as repair,
            ):
                self.assertTrue(maintenance.backend_repair_command())
            repair.assert_called_once_with(config, 'realtime-ws')


class MaintenanceUinputValidationTests(unittest.TestCase):
    def _path(self, exists):
        path = mock.Mock()
        path.exists.return_value = exists
        return path

    def test_accessible_uinput_passes(self):
        with (
            mock.patch.object(maintenance, 'Path', return_value=self._path(True)),
            mock.patch.object(maintenance.os, 'geteuid', return_value=1000),
            mock.patch.object(maintenance.os, 'access', return_value=True),
            mock.patch.object(maintenance, 'log_success') as success,
        ):
            self.assertTrue(maintenance._validate_uinput_access(True))
        success.assert_called_once_with("✓ /dev/uinput is readable and writable")

    def test_missing_uinput_is_error_when_ydotool_is_only_path(self):
        output = io.StringIO()
        with (
            mock.patch.object(maintenance, 'Path', return_value=self._path(False)),
            mock.patch.object(maintenance.os, 'geteuid', return_value=1000),
            mock.patch.object(maintenance, '_has_non_ydotool_injection_path', return_value=False),
            mock.patch.object(maintenance, 'log_error') as error,
            redirect_stdout(output),
        ):
            self.assertFalse(maintenance._validate_uinput_access(True))
        self.assertIn("does not exist", error.call_args.args[0])
        self.assertIn("log out and back in (or reboot)", output.getvalue())

    def test_inaccessible_uinput_warns_when_another_path_exists(self):
        with (
            mock.patch.object(maintenance, 'Path', return_value=self._path(True)),
            mock.patch.object(maintenance.os, 'geteuid', return_value=1000),
            mock.patch.object(maintenance.os, 'access', return_value=False),
            mock.patch.object(maintenance, '_has_non_ydotool_injection_path', return_value=True),
            mock.patch.object(maintenance, 'log_warning') as warning,
            mock.patch('builtins.print'),
        ):
            self.assertTrue(maintenance._validate_uinput_access(True))
        self.assertIn("not readable and writable", warning.call_args.args[0])
        self.assertIn("Other detected injection paths", warning.call_args.args[0])

    def test_missing_ydotool_and_no_alternative_does_not_claim_fallback(self):
        with (
            mock.patch.object(maintenance, 'Path', return_value=self._path(False)),
            mock.patch.object(maintenance.os, 'geteuid', return_value=1000),
            mock.patch.object(maintenance, '_has_non_ydotool_injection_path', return_value=False),
            mock.patch.object(maintenance, 'log_warning') as warning,
            mock.patch('builtins.print'),
        ):
            self.assertTrue(maintenance._validate_uinput_access(False))
        self.assertIn("No other injection path was detected", warning.call_args.args[0])

    def test_root_validation_warns_instead_of_claiming_user_access(self):
        with (
            mock.patch.object(maintenance, 'Path', return_value=self._path(True)),
            mock.patch.object(maintenance.os, 'geteuid', return_value=0),
            mock.patch.object(maintenance.os, 'access') as access,
            mock.patch.object(maintenance, 'log_success') as success,
            mock.patch.object(maintenance, 'log_warning') as warning,
            mock.patch('builtins.print') as output,
        ):
            self.assertTrue(maintenance._validate_uinput_access(True))
        access.assert_not_called()
        success.assert_not_called()
        self.assertIn("cannot verify", warning.call_args.args[0])
        output.assert_called_once_with("  Re-run 'hyprwhspr validate' without sudo.")


if __name__ == '__main__':
    unittest.main()
