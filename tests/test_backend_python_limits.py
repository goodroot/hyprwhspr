"""Backend-specific interpreter gates, with no real environment mutation."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests import test_python_upgrade_recovery as recovery
from tests.test_managed_install import ManagedFixture
import backend_installer as installer
import dependency_plan as plans
import managed_install as managed

ROOT = Path(__file__).resolve().parents[1]


class BackendPythonLimitTests(unittest.TestCase):
    def test_family_limits(self):
        for key, (_, _, family) in plans.PLAN_SPECS.items():
            with self.subTest(plan=key):
                plans.check_backend_python(family, (3, 14, 0))
                if family in ('onnx', 'faster-whisper'):
                    with self.assertRaisesRegex(ValueError, r'Python 3.15.*--python /path/to/python3.14'):
                        plans.check_backend_python(family, (3, 15, 0))
                else:
                    plans.check_backend_python(family, (3, 15, 0))

    def test_legacy_gate_and_reuse(self):
        for backend in ('onnx-asr', 'faster-whisper'):
            for variant in ('cpu', 'cuda'):
                plan = plans.resolve(ROOT, backend, None, variant, ValueError)
                with self.subTest(backend=backend, variant=variant), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                    stack.enter_context(mock.patch.dict(installer.os.environ, {}, clear=True))
                    stack.enter_context(mock.patch.object(installer, 'VENV_DIR', Path(tmp)))
                    state = stack.enter_context(mock.patch.object(installer, 'get_state', return_value='old'))
                    stack.enter_context(mock.patch.object(installer, '_verify_dependency_plan', return_value=True))
                    find = stack.enter_context(mock.patch.object(installer, '_find_compatible_python',
                                                                 return_value=('/fixture/python3', 'fixture')))
                    version = stack.enter_context(mock.patch.object(installer, '_get_python_version', return_value=(3, 15)))
                    transaction = stack.enter_context(mock.patch.object(installer, 'VenvTransaction'))
                    setup = stack.enter_context(mock.patch.object(installer, 'setup_python_venv'))
                    with self.assertRaisesRegex(installer.DependencyPlanError, 'requires Python 3.14'):
                        installer.execute_dependency_plan(plan)
                    transaction.assert_not_called()
                    setup.assert_not_called()
                    state.return_value = plan.fingerprint
                    installer.execute_dependency_plan(plan)
                    self.assertEqual(find.call_count, 1)
                    transaction.assert_not_called()
                    state.return_value = 'old'
                    version.return_value = (3, 14)
                    stack.enter_context(mock.patch.object(installer, '_snapshot_inherited_dependencies'))
                    stack.enter_context(mock.patch.object(installer, 'run_command'))
                    stack.enter_context(mock.patch.object(installer, '_verify_and_repair_dependency_plan',
                                                         return_value=mock.Mock(ok=True)))
                    stack.enter_context(mock.patch.object(installer, 'commit_dependency_state'))
                    installer.execute_dependency_plan(plan, custom_python='/fixture/python3.14')
                    setup.assert_called_once_with(custom_python='/fixture/python3.14')
                    self.assertEqual(find.call_count, 1)


class ManagedBackendPythonLimitTests(ManagedFixture):
    prepare_environments = recovery.PythonUpgradeRecoveryTests.prepare_environments

    def test_rejection_preserves_active_generation_before_build(self):
        with ExitStack() as stack:
            old, runner, pip = self.prepare_environments(stack)
            identity = dict(old['python'], version=[3, 15, 0])
            stack.enter_context(mock.patch.object(managed, 'interpreter', return_value=identity))
            stack.enter_context(mock.patch.object(managed, 'verify_payload', return_value={'version': old['version']}))
            build = stack.enter_context(mock.patch.object(self.install, 'build'))
            service = stack.enter_context(mock.patch.object(self.install, 'service'))
            for backend in ('onnx-asr', 'faster-whisper'):
                for variant in ('cpu', 'cuda'):
                    with self.subTest(backend=backend, variant=variant), self.assertRaisesRegex(ValueError, 'requires Python 3.14'):
                        self.install.update(local_payload=True, selection={'backend': backend, 'variant': variant})
                    self.assertEqual(managed.read_json(self.install.current), old)
                    for kind in ('cli', 'backend'):
                        self.assertTrue(Path(old[kind]['path']).is_dir())
            build.assert_not_called()
            service.assert_not_called()
            runner.assert_not_called()
            pip.assert_not_called()
