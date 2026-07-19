import builtins
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib' / 'src'))

import backend_installer


class DependencyPlanTests(unittest.TestCase):
    def plan(self, backend, provider=None, variant=None):
        with mock.patch.object(backend_installer, 'HYPRWHSPR_ROOT', str(ROOT)):
            return backend_installer.resolve_dependency_plan(backend, provider, variant)

    def test_all_backend_selections(self):
        cases = {
            ('cpu', None, None): 'requirements-pywhispercpp.txt',
            ('rest-api', 'openai', None): 'requirements-rest.txt',
            ('realtime-ws', 'openai', None): 'requirements-realtime.txt',
            ('realtime-ws', 'google', None): 'requirements-realtime.txt',
            ('realtime-ws', 'custom', None): 'requirements-realtime.txt',
            ('realtime-ws', 'elevenlabs', None): 'requirements-realtime-elevenlabs.txt',
            ('cohere-transcribe', None, None): 'requirements-cohere-transcribe.txt',
            ('onnx-asr', None, None): 'requirements-onnx-asr.txt',
            ('onnx-asr', None, 'gpu'): 'requirements-onnx-asr-gpu.txt',
            ('faster-whisper', None, None): 'requirements-faster-whisper.txt',
            ('faster-whisper', None, 'cuda'): 'requirements-faster-whisper-cuda.txt',
        }
        for args, filename in cases.items():
            with self.subTest(args=args):
                self.assertEqual(self.plan(*args).manifest.name, filename)

    def test_realtime_family_equivalence_and_transport_imports(self):
        plans = [self.plan('realtime-ws', provider) for provider in ('openai', 'google', 'custom')]
        self.assertEqual({plan.family for plan in plans}, {'realtime'})
        self.assertIn('websocket', plans[0].required_imports)
        eleven = self.plan('realtime-ws', 'elevenlabs')
        self.assertNotIn('websocket', eleven.required_imports)

    def test_recursive_include_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'requirements.txt').write_text('numpy\n', encoding='utf-8')
            (root / 'requirements-rest.txt').write_text('-r requirements.txt\nrequests\n', encoding='utf-8')
            with mock.patch.object(backend_installer, 'HYPRWHSPR_ROOT', tmp):
                first = backend_installer.resolve_dependency_plan('rest-api')
                (root / 'requirements.txt').write_text('numpy>=2\n', encoding='utf-8')
                second = backend_installer.resolve_dependency_plan('rest-api')
            self.assertNotEqual(first.fingerprint, second.fingerprint)

    def test_duplicate_manifest_basenames_have_distinct_identities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / 'left' / 'shared.txt'
            right = root / 'right' / 'shared.txt'
            left.parent.mkdir()
            right.parent.mkdir()
            left.write_text('one\n', encoding='utf-8')
            right.write_text('two\n', encoding='utf-8')
            manifest = root / 'root.txt'
            manifest.write_text('', encoding='utf-8')
            first = backend_installer.dependency_manifest_hash([left, right, manifest])
            second = backend_installer.dependency_manifest_hash([right, left, manifest])
        self.assertNotEqual(first, second)

    def test_constraint_changes_fingerprint_and_is_preflighted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            constraint = root / 'constraints.txt'
            constraint.write_text('requests<3\n', encoding='utf-8')
            (root / 'requirements-rest.txt').write_text(
                '-cconstraints.txt\nrequests\n', encoding='utf-8'
            )
            with mock.patch.object(backend_installer, 'HYPRWHSPR_ROOT', tmp):
                first = backend_installer.resolve_dependency_plan('rest-api')
                constraint.write_text('requests<4\n', encoding='utf-8')
                second = backend_installer.resolve_dependency_plan('rest-api')
                constraint.unlink()
                with self.assertRaisesRegex(backend_installer.DependencyPlanError, 'missing'):
                    backend_installer.resolve_dependency_plan('rest-api')
        self.assertNotEqual(first.fingerprint, second.fingerprint)

    def test_requirement_constraint_cycle_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'requirements-rest.txt').write_text('-c constraints.txt\n', encoding='utf-8')
            (root / 'constraints.txt').write_text('-r requirements-rest.txt\n', encoding='utf-8')
            with mock.patch.object(backend_installer, 'HYPRWHSPR_ROOT', tmp):
                with self.assertRaisesRegex(backend_installer.DependencyPlanError, 'Cyclic'):
                    backend_installer.resolve_dependency_plan('rest-api')

    def test_remote_manifest_is_rejected_with_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'requirements-rest.txt').write_text(
                '--requirement=https://example.invalid/requirements.txt\n', encoding='utf-8'
            )
            with mock.patch.object(backend_installer, 'HYPRWHSPR_ROOT', tmp):
                with self.assertRaisesRegex(backend_installer.DependencyPlanError, 'vendor it'):
                    backend_installer.resolve_dependency_plan('rest-api')

    def test_filter_uses_canonical_project_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / 'requirements.txt'
            manifest.write_text('some.package>=1\nother-package\n', encoding='utf-8')
            filtered = backend_installer._filter_requirements(manifest, ['some_package'])
            try:
                content = filtered.read_text(encoding='utf-8')
            finally:
                filtered.unlink()
        self.assertNotIn('some.package', content)
        self.assertIn('other-package', content)

    def test_filter_requirements_flattens_includes_for_temp_file_safety(self):
        """
        Regression test: _filter_requirements() used to copy `-r`/`--requirement`
        lines from the source manifest verbatim into a NamedTemporaryFile under
        /tmp. pip resolves such relative includes against the *including*
        file's own directory, so a manifest like requirements-pywhispercpp.txt
        (which starts with `-r requirements.txt`) produced a temp file whose
        nested include silently pointed at the nonexistent
        /tmp/requirements.txt, failing with "Could not open requirements
        file: ... '/tmp/requirements.txt'" even though the flattened
        dependency list was otherwise correct.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'requirements.txt').write_text(
                'numpy\n# comment\nPyGObject>=3.50\n', encoding='utf-8'
            )
            (root / 'requirements-pywhispercpp.txt').write_text(
                '-r requirements.txt\npywhispercpp==1.5.0\n', encoding='utf-8'
            )
            temp_path = backend_installer._filter_requirements(
                root / 'requirements-pywhispercpp.txt', ['PyGObject']
            )
            try:
                content = temp_path.read_text(encoding='utf-8')
            finally:
                temp_path.unlink()

        self.assertNotIn('-r ', content)
        self.assertNotIn('--requirement', content)
        self.assertIn('numpy', content)
        self.assertIn('pywhispercpp==1.5.0', content)
        self.assertNotIn('PyGObject', content)

    def test_filter_requirements_preserves_include_position_and_constraints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            constraints = root / 'constraints.txt'
            constraints.write_text('numpy<3\n', encoding='utf-8')
            (root / 'common.txt').write_text('included-package', encoding='utf-8')
            (root / 'requirements.txt').write_text(
                'before-package\n-rcommon.txt\nafter-package\n-c constraints.txt\n',
                encoding='utf-8',
            )

            temp_path = backend_installer._filter_requirements(
                root / 'requirements.txt', []
            )
            try:
                content = temp_path.read_text(encoding='utf-8')
            finally:
                temp_path.unlink()

        self.assertEqual(
            content.splitlines(),
            [
                'before-package',
                'included-package',
                'after-package',
                f'--constraint {constraints.resolve()}',
            ],
        )

    def test_missing_include_is_actionable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'requirements-rest.txt').write_text('-r absent.txt\n', encoding='utf-8')
            with mock.patch.object(backend_installer, 'HYPRWHSPR_ROOT', tmp):
                with self.assertRaisesRegex(backend_installer.DependencyPlanError, 'package payload'):
                    backend_installer.resolve_dependency_plan('rest-api')

    def test_failed_install_restores_previous_environment(self):
        plan = self.plan('rest-api')
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / 'venv'
            venv.mkdir()
            (venv / 'old-marker').touch()

            def setup(**_kwargs):
                (venv / 'bin').mkdir(parents=True)
                (venv / 'bin' / 'pip').touch()
                return venv / 'bin' / 'pip'

            with (
                mock.patch.object(backend_installer, 'VENV_DIR', venv),
                mock.patch.object(backend_installer, 'get_state', return_value='old'),
                mock.patch.object(backend_installer, 'setup_python_venv', side_effect=setup),
                mock.patch.object(backend_installer, 'run_command', side_effect=RuntimeError('pip failed')),
            ):
                with self.assertRaisesRegex(RuntimeError, 'pip failed'):
                    backend_installer.execute_dependency_plan(plan)
            self.assertTrue((venv / 'old-marker').exists())

    def test_verified_environment_survives_state_write_failure(self):
        plan = self.plan('rest-api')
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / 'venv'

            def setup(**_kwargs):
                (venv / 'bin').mkdir(parents=True)
                (venv / 'bin' / 'pip').touch()
                return venv / 'bin' / 'pip'

            completed = types.SimpleNamespace(returncode=0)
            with (
                mock.patch.object(backend_installer, 'VENV_DIR', venv),
                mock.patch.object(backend_installer, 'get_state', return_value=''),
                mock.patch.object(backend_installer, 'setup_python_venv', side_effect=setup),
                mock.patch.object(backend_installer, 'run_command', return_value=completed),
                mock.patch.object(backend_installer, 'commit_dependency_state', side_effect=OSError('disk full')),
            ):
                backend_installer.execute_dependency_plan(plan)
            self.assertTrue((venv / 'bin' / 'pip').exists())

    def test_matching_legacy_hash_is_migrated_on_fast_path(self):
        plan = self.plan('rest-api')
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / 'venv'
            (venv / 'bin').mkdir(parents=True)
            (venv / 'bin' / 'pip').touch()
            values = iter(('', plan.fingerprint))
            completed = types.SimpleNamespace(returncode=0)
            with (
                mock.patch.object(backend_installer, 'VENV_DIR', venv),
                mock.patch.object(backend_installer, 'get_state', side_effect=lambda _key: next(values)),
                mock.patch.object(backend_installer, 'run_command', return_value=completed),
                mock.patch.object(backend_installer, 'commit_dependency_state') as commit,
                mock.patch.object(backend_installer, 'setup_python_venv') as setup,
            ):
                backend_installer.execute_dependency_plan(plan)
            commit.assert_called_once_with(plan)
            setup.assert_not_called()

    def test_accelerated_manifest_is_preflighted_before_gpu_or_venv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'requirements.txt').write_text('numpy\n', encoding='utf-8')
            (root / 'requirements-faster-whisper.txt').write_text(
                '-r requirements.txt\nfaster-whisper\n', encoding='utf-8')
            with (
                mock.patch.object(backend_installer, 'HYPRWHSPR_ROOT', tmp),
                mock.patch.object(backend_installer, 'init_state') as init_state,
                mock.patch.object(backend_installer, 'setup_python_venv') as setup,
                mock.patch.object(backend_installer.shutil, 'which') as gpu_check,
            ):
                self.assertFalse(backend_installer.install_backend('faster-whisper'))
            init_state.assert_not_called()
            setup.assert_not_called()
            gpu_check.assert_not_called()

    def test_partial_cleanup_restores_backup_and_removes_new_source_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            venv = root / 'venv'
            backup = root / 'venv.rollback'
            source = root / 'pywhispercpp-src'
            (venv / 'bin').mkdir(parents=True)
            backup.mkdir()
            (backup / 'old-marker').touch()
            source.mkdir()
            (source / 'partial-clone').touch()
            items = {
                'venv_created': True,
                'venv_path': str(venv),
                'venv_backup_path': str(backup),
                'git_clone_created': True,
                'git_clone_path': str(source),
                'packages_installed': [],
            }
            with mock.patch.object(backend_installer, 'VENV_DIR', venv):
                backend_installer._cleanup_partial_installation(items, None)
            self.assertTrue((venv / 'old-marker').exists())
            self.assertFalse(source.exists())

    def test_pywhispercpp_helpers_defer_backend_state_commit(self):
        completed = types.SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as tmp:
            requirements = Path(tmp) / 'requirements.txt'
            requirements.write_text('pywhispercpp\n', encoding='utf-8')
            pip_bin = Path(tmp) / 'venv' / 'bin' / 'pip'
            with (
                mock.patch.object(backend_installer, '_should_skip_pygobject', return_value=False),
                mock.patch.object(backend_installer, 'run_command', return_value=completed),
                mock.patch.object(backend_installer, 'set_state') as set_state,
            ):
                self.assertTrue(backend_installer.install_pywhispercpp_cpu(pip_bin, requirements))
            set_state.assert_not_called()

            wheel = Path(tmp) / 'wheel.whl'
            with (
                mock.patch.object(backend_installer, 'download_pywhispercpp_wheel', return_value=wheel),
                mock.patch.object(backend_installer, 'install_pywhispercpp_from_wheel', return_value=True),
                mock.patch.object(backend_installer, 'set_state') as set_state,
            ):
                self.assertTrue(backend_installer.install_pywhispercpp_cuda(pip_bin))
            set_state.assert_not_called()

    def test_accelerated_cpu_fallback_cleans_only_new_source_and_reports_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'new-source'
            source.mkdir()
            (source / 'failed-build').touch()
            items = {
                'git_clone_created': True,
                'git_clone_path': str(source),
            }
            for requested_variant in ('rocm', 'vulkan'):
                with self.subTest(requested_variant=requested_variant):
                    source.mkdir(exist_ok=True)
                    items.update({
                        'git_clone_created': True,
                        'git_clone_path': str(source),
                    })
                    effective = backend_installer._complete_pywhispercpp_cpu_fallback(items)
                    self.assertEqual(effective, 'cpu')
                    self.assertFalse(source.exists())
                    self.assertFalse(items['git_clone_created'])

            existing = Path(tmp) / 'existing-source'
            existing.mkdir()
            existing_items = {
                'git_clone_created': False,
                'git_clone_path': str(existing),
            }
            effective = backend_installer._complete_pywhispercpp_cpu_fallback(existing_items)
            self.assertEqual(effective, 'cpu')
            self.assertTrue(existing.exists())

class RealtimeTransportIsolationTests(unittest.TestCase):
    def test_audio_base_import_does_not_load_websocket_client(self):
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == 'websocket':
                raise ModuleNotFoundError(name)
            return original_import(name, *args, **kwargs)

        sys.modules.pop('realtime_base', None)
        with mock.patch('builtins.__import__', side_effect=guarded):
            module = importlib.import_module('realtime_base')
            module.RealtimeAudioClientBase()
            with self.assertRaisesRegex(RuntimeError, 'websocket-client'):
                module.WebSocketRealtimeClientBase()


if __name__ == '__main__':
    unittest.main()
