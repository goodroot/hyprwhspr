import sys
import tempfile
import unittest
from pathlib import Path
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


if __name__ == '__main__':
    unittest.main()
