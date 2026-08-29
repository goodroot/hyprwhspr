import builtins
import importlib
import subprocess
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
                plan = self.plan(*args)
                self.assertEqual(plan.manifest.name, filename)
                self.assertIn('soundfile', plan.required_imports)

    def test_realtime_family_equivalence_and_transport_imports(self):
        plans = [self.plan('realtime-ws', provider) for provider in ('openai', 'google', 'custom')]
        self.assertEqual({plan.family for plan in plans}, {'realtime'})
        self.assertIn('websocket', plans[0].required_imports)
        eleven = self.plan('realtime-ws', 'elevenlabs')
        self.assertNotIn('websocket', eleven.required_imports)

    def test_cohere_probes_scientific_audio_imports(self):
        plan = self.plan('cohere-transcribe')
        self.assertIn('pandas', plan.required_imports)
        self.assertIn('scipy', plan.required_imports)
        self.assertIn('numba', plan.required_imports)
        self.assertIn('sklearn', plan.required_imports)
        self.assertIn('librosa', plan.required_imports)
        self.assertIn('soundfile', plan.required_imports)
        for name in ('pandas', 'scipy', 'numba', 'sklearn'):
            self.assertLess(
                plan.required_imports.index(name),
                plan.required_imports.index('transformers'))
        specs = backend_installer._manifest_requirement_specs(plan)
        self.assertEqual(specs['pandas'], 'pandas')
        self.assertEqual(specs['scikit-learn'], 'scikit-learn')
        self.assertEqual(specs['scipy'], 'scipy')
        self.assertEqual(specs['numba'], 'numba')

    def test_numpy_abi_signatures_are_specific(self):
        signatures = (
            '_ARRAY_API not found',
            'A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0',
            'ValueError: numpy.dtype size changed, may indicate binary incompatibility',
            'module compiled against API version 0x10 but this version of numpy is 0xf',
            'ImportError: Numba needs NumPy 2.2 or less. Got NumPy 2.3.',
        )
        for text in signatures:
            with self.subTest(text=text):
                self.assertTrue(backend_installer._has_numpy_abi_signature(text))
        self.assertFalse(backend_installer._has_numpy_abi_signature('No module named requests'))

    def test_distribution_mapping_uses_manifest_canonical_name(self):
        probe = backend_installer.ImportProbe(
            'sklearn', False, distributions=('scikit-learn',))
        self.assertEqual(
            backend_installer._distribution_for_import(probe, {'scikit-learn'}),
            'scikit-learn',
        )
        ambiguous = backend_installer.ImportProbe(
            'thing', False, distributions=('one', 'two'))
        self.assertIsNone(
            backend_installer._distribution_for_import(ambiguous, {'one', 'two'}))

    def test_combined_success_avoids_isolated_probes(self):
        plan = self.plan('rest-api')
        success = types.SimpleNamespace(returncode=0, stdout='', stderr='')
        with (
            mock.patch.object(backend_installer, 'run_command', return_value=success),
            mock.patch.object(backend_installer, '_probe_required_import') as probe,
        ):
            result = backend_installer._verify_dependency_plan_detailed(plan)
        self.assertTrue(result.ok)
        probe.assert_not_called()

    def test_combined_failure_probes_every_import_and_retains_stderr(self):
        plan = self.plan('rest-api')
        failed = types.SimpleNamespace(returncode=1, stdout='combined out', stderr='combined err')
        probes = [
            backend_installer.ImportProbe(name, name != 'soxr', stderr=f'{name} stderr')
            for name in plan.required_imports
        ]
        with (
            mock.patch.object(backend_installer, 'run_command', return_value=failed),
            mock.patch.object(backend_installer, '_probe_required_import', side_effect=probes) as probe,
        ):
            result = backend_installer._verify_dependency_plan_detailed(plan)
        self.assertEqual(probe.call_count, len(plan.required_imports))
        self.assertEqual(result.combined_stderr, 'combined err')
        self.assertEqual(result.failures[0].stderr, 'soxr stderr')

    def test_combined_only_failure_is_not_repairable(self):
        plan = self.plan('rest-api')
        verification = backend_installer.DependencyVerification(
            ok=False, combined_only_failure=True, combined_stderr='import order')
        self.assertEqual(backend_installer._repairable_distributions(plan, verification), [])
        self.assertIn('combined probe failed',
                      backend_installer._format_dependency_diagnostic(plan, verification))

    def test_only_allowlisted_system_abi_failures_are_repaired(self):
        plan = self.plan('rest-api')
        failures = [
            backend_installer.ImportProbe(
                'soxr', False, '/usr/lib/python/site-packages/soxr/__init__.py',
                stderr='_ARRAY_API not found', distributions=('soxr',)),
            backend_installer.ImportProbe(
                'sounddevice', False, '/usr/lib/python/site-packages/sounddevice.py',
                stderr='numpy.dtype size changed', distributions=('sounddevice',)),
            backend_installer.ImportProbe(
                'gi', False, '/usr/lib/python/site-packages/gi/__init__.py',
                stderr='_ARRAY_API not found', distributions=('PyGObject',)),
        ]
        verification = backend_installer.DependencyVerification(False, failures=failures)
        with mock.patch.object(
                backend_installer, '_manifest_package_names',
                return_value={'soxr', 'sounddevice', 'pygobject'}):
            self.assertEqual(
                backend_installer._repairable_distributions(plan, verification),
                ['sounddevice', 'soxr'],
            )

    def test_venv_extension_with_inherited_numpy_relocates_numpy(self):
        plan = self.plan('cohere-transcribe')
        probe = backend_installer.ImportProbe(
            'librosa', False,
            str(backend_installer.VENV_DIR / 'lib/python/site-packages/librosa/__init__.py'),
            stderr='compiled using NumPy 1.x cannot be run in NumPy 2',
            numpy_version='1.26.4',
            numpy_origin='/usr/lib/python3/dist-packages/numpy/__init__.py',
            distributions=('librosa',),
        )
        verification = backend_installer.DependencyVerification(False, failures=[probe])
        self.assertEqual(
            backend_installer._repairable_distributions(plan, verification),
            ['numpy'],
        )
        self.assertEqual(
            backend_installer._repair_requirement_specs(plan, ['numpy']),
            ['numpy>=1.26.0'],
        )

    def test_inherited_pandas_against_venv_numpy_relocates_only_pandas(self):
        plan = self.plan('cohere-transcribe')
        probe = backend_installer.ImportProbe(
            'pandas', False,
            '/usr/lib/python3/dist-packages/pandas/__init__.py',
            traceback=(
                'ValueError: numpy.dtype size changed, may indicate binary '
                'incompatibility. Expected 96 from C header, got 88 from PyObject'
            ),
            numpy_version='2.5.2',
            numpy_origin=str(
                backend_installer.VENV_DIR
                / 'lib/python3.12/site-packages/numpy/__init__.py'),
            distributions=('pandas',),
        )
        verification = backend_installer.DependencyVerification(
            False, failures=[probe])
        self.assertEqual(
            backend_installer._repairable_distributions(plan, verification),
            ['pandas'],
        )
        self.assertEqual(
            backend_installer._repair_requirement_specs(plan, ['pandas']),
            ['pandas'],
        )

    def test_inherited_sklearn_maps_to_scikit_learn_repair(self):
        plan = self.plan('cohere-transcribe')
        probe = backend_installer.ImportProbe(
            'sklearn', False,
            '/usr/lib/python3/dist-packages/sklearn/__init__.py',
            traceback='_ARRAY_API not found',
            numpy_version='2.5.2',
            numpy_origin=str(
                backend_installer.VENV_DIR
                / 'lib/python3.12/site-packages/numpy/__init__.py'),
            distributions=('scikit-learn',),
        )
        verification = backend_installer.DependencyVerification(
            False, failures=[probe])
        self.assertEqual(
            backend_installer._repairable_distributions(plan, verification),
            ['scikit-learn'],
        )

    def test_inherited_cohere_compiled_dependencies_are_repairable(self):
        plan = self.plan('cohere-transcribe')
        cases = (
            ('scipy', 'ValueError: numpy.dtype size changed'),
            ('numba', 'ImportError: Numba needs NumPy 2.2 or less. Got NumPy 2.3.'),
        )
        for distribution, traceback in cases:
            with self.subTest(distribution=distribution):
                probe = backend_installer.ImportProbe(
                    distribution, False,
                    f'/usr/lib/python3/dist-packages/{distribution}/__init__.py',
                    traceback=traceback,
                    numpy_version='2.5.2',
                    numpy_origin=str(
                        backend_installer.VENV_DIR
                        / 'lib/python3.12/site-packages/numpy/__init__.py'),
                    distributions=(distribution,),
                )
                verification = backend_installer.DependencyVerification(
                    False, failures=[probe])
                self.assertEqual(
                    backend_installer._repairable_distributions(
                        plan, verification),
                    [distribution],
                )
                self.assertEqual(
                    backend_installer._repair_requirement_specs(
                        plan, [distribution]),
                    [distribution],
                )

    def test_live_probe_detects_venv_extension_against_inherited_numpy(self):
        plan = self.plan('rest-api')
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / 'venv'
            subprocess.run(
                [sys.executable, '-m', 'venv', '--system-site-packages', str(venv)],
                check=True, capture_output=True, text=True)
            venv_python = venv / 'bin' / 'python'
            purelib = Path(subprocess.run(
                [str(venv_python), '-c',
                 'import sysconfig; print(sysconfig.get_paths()["purelib"])'],
                check=True, capture_output=True, text=True).stdout.strip())
            (purelib / 'soxr.py').write_text(
                'raise ImportError("compiled using NumPy 1.x cannot be run in NumPy 2")\n',
                encoding='utf-8')
            dist_info = purelib / 'soxr-1.0.dist-info'
            dist_info.mkdir()
            (dist_info / 'METADATA').write_text(
                'Metadata-Version: 2.1\nName: soxr\nVersion: 1.0\n', encoding='utf-8')
            (dist_info / 'top_level.txt').write_text('soxr\n', encoding='utf-8')

            with mock.patch.object(backend_installer, 'VENV_DIR', venv):
                probe = backend_installer._probe_required_import('soxr')
                if not probe.numpy_origin:
                    self.skipTest('system interpreter does not expose NumPy')
                self.assertFalse(probe.ok)
                self.assertEqual(probe.distributions, ('soxr',))
                self.assertTrue(str(probe.module_origin).startswith(str(venv)))
                self.assertFalse(str(probe.numpy_origin).startswith(str(venv)))
                verification = backend_installer.DependencyVerification(
                    False, failures=[probe])
                self.assertEqual(
                    backend_installer._repairable_distributions(plan, verification),
                    ['numpy'],
                )

    def test_live_probe_detects_inherited_pandas_against_venv_numpy(self):
        plan = self.plan('cohere-transcribe')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            venv = root / 'venv'
            inherited = root / 'inherited-site'
            pandas = inherited / 'pandas'
            pandas.mkdir(parents=True)
            (pandas / '__init__.py').write_text(
                'raise ValueError("numpy.dtype size changed, may indicate binary '
                'incompatibility. Expected 96 from C header, got 88 from PyObject")\n',
                encoding='utf-8')
            dist_info = inherited / 'pandas-1.0.dist-info'
            dist_info.mkdir()
            (dist_info / 'METADATA').write_text(
                'Metadata-Version: 2.1\nName: pandas\nVersion: 1.0\n',
                encoding='utf-8')
            (dist_info / 'top_level.txt').write_text(
                'pandas\n', encoding='utf-8')

            subprocess.run(
                [sys.executable, '-m', 'venv', '--system-site-packages', str(venv)],
                check=True, capture_output=True, text=True)
            venv_python = venv / 'bin' / 'python'
            purelib = Path(subprocess.run(
                [str(venv_python), '-c',
                 'import sysconfig; print(sysconfig.get_paths()["purelib"])'],
                check=True, capture_output=True, text=True).stdout.strip())
            numpy = purelib / 'numpy'
            numpy.mkdir()
            (numpy / '__init__.py').write_text(
                '__version__ = "2.5.2"\n', encoding='utf-8')
            (purelib / 'inherited-pandas.pth').write_text(
                f'import sys; sys.path.insert(0, {str(inherited)!r})\n',
                encoding='utf-8')

            with mock.patch.object(backend_installer, 'VENV_DIR', venv):
                probe = backend_installer._probe_required_import('pandas')
                self.assertFalse(probe.ok)
                self.assertEqual(probe.distributions, ('pandas',))
                self.assertTrue(str(probe.module_origin).startswith(str(inherited)))
                self.assertEqual(probe.numpy_version, '2.5.2')
                self.assertTrue(str(probe.numpy_origin).startswith(str(venv)))
                verification = backend_installer.DependencyVerification(
                    False, failures=[probe])
                self.assertEqual(
                    backend_installer._repairable_distributions(plan, verification),
                    ['pandas'],
                )

    def test_system_sounddevice_missing_extension_is_repaired(self):
        plan = self.plan('rest-api')
        probe = backend_installer.ImportProbe(
            'sounddevice', False,
            '/usr/lib/python3/site-packages/sounddevice.py',
            traceback="ModuleNotFoundError: No module named '_sounddevice'",
            distributions=('sounddevice',),
        )
        verification = backend_installer.DependencyVerification(False, failures=[probe])
        self.assertEqual(
            backend_installer._repairable_distributions(plan, verification),
            ['sounddevice'],
        )

    def test_user_site_origin_is_inherited(self):
        origin = str(Path.home() / '.local/lib/python3.14/site-packages/soxr/__init__.py')
        self.assertTrue(backend_installer._is_system_origin(origin))

    def test_manifest_specs_are_used_for_repair(self):
        plan = self.plan('rest-api')
        self.assertEqual(
            backend_installer._repair_requirement_specs(
                plan, ['numpy', 'sounddevice', 'soxr']),
            ['numpy>=1.26.0', 'sounddevice>=0.5.0', 'soxr>=0.5.0'],
        )

    def test_combined_timeout_retries_and_accepts_isolated_success(self):
        plan = self.plan('rest-api')
        timed_out = types.SimpleNamespace(returncode=124, stdout='', stderr='')
        probes = [
            backend_installer.ImportProbe(name, True)
            for name in plan.required_imports
        ]
        with (
            mock.patch.object(backend_installer, 'run_command',
                              side_effect=(timed_out, timed_out)) as run,
            mock.patch.object(backend_installer, '_probe_required_import', side_effect=probes),
        ):
            verification = backend_installer._verify_dependency_plan_detailed(plan)
        self.assertTrue(verification.ok)
        self.assertIn('180s', run.call_args_list[1].args[0])

    def test_isolated_probe_uses_one_180_second_budget(self):
        completed = types.SimpleNamespace(returncode=124, stdout='', stderr='')
        with mock.patch.object(
                backend_installer, 'run_command', return_value=completed) as run:
            probe = backend_installer._probe_required_import('torch')
        self.assertTrue(probe.timed_out)
        run.assert_called_once()
        self.assertIn('180s', run.call_args.args[0])

    def test_timeout_diagnostic_is_not_import_order_message(self):
        plan = self.plan('rest-api')
        verification = backend_installer.DependencyVerification(
            False, timed_out=True,
            failures=[backend_installer.ImportProbe('soxr', False, timed_out=True)],
        )
        diagnostic = backend_installer._format_dependency_diagnostic(plan, verification)
        self.assertIn('verification timed out', diagnostic)
        self.assertNotIn('import-order interaction', diagnostic)

    def test_snapshot_numpy_before_and_after_appear_in_diagnostic(self):
        plan = self.plan('rest-api')
        snapshot = [backend_installer.ImportProbe(
            'numpy', True, numpy_version='1.26.4', numpy_origin='/usr/lib/numpy/__init__.py')]
        failed = backend_installer.DependencyVerification(False, failures=[
            backend_installer.ImportProbe(
                'soxr', False, numpy_version='1.26.4',
                numpy_origin='/usr/lib/numpy/__init__.py')])
        diagnostic = backend_installer._format_dependency_diagnostic(plan, failed, snapshot)
        self.assertIn('NumPy before install: 1.26.4', diagnostic)
        self.assertIn('NumPy after install: 1.26.4', diagnostic)

    def test_all_repairs_install_together_then_verify_once(self):
        plan = self.plan('rest-api')
        failed = backend_installer.DependencyVerification(False)
        passed = backend_installer.DependencyVerification(True)
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / 'venv'

            def setup(**_kwargs):
                (venv / 'bin').mkdir(parents=True)
                (venv / 'bin' / 'pip').touch()
                return venv / 'bin' / 'pip'

            completed = types.SimpleNamespace(returncode=0, stdout='', stderr='')
            with (
                mock.patch.object(backend_installer, 'VENV_DIR', venv),
                mock.patch.object(backend_installer, 'get_state', return_value=''),
                mock.patch.object(backend_installer, 'setup_python_venv', side_effect=setup),
                mock.patch.object(backend_installer, '_snapshot_inherited_dependencies'),
                mock.patch.object(backend_installer, '_verify_dependency_plan_detailed',
                                  side_effect=(failed, passed)) as verify,
                mock.patch.object(backend_installer, '_repairable_distributions',
                                  return_value=['sounddevice', 'soxr']),
                mock.patch.object(backend_installer, 'run_command', return_value=completed) as run,
                mock.patch.object(backend_installer, 'commit_dependency_state') as commit,
            ):
                backend_installer.execute_dependency_plan(plan)
        repair_calls = [
            call for call in run.call_args_list
            if '--ignore-installed' in call.args[0]
        ]
        self.assertEqual(len(repair_calls), 1)
        self.assertEqual(
            repair_calls[0].args[0][-2:],
            ['sounddevice>=0.5.0', 'soxr>=0.5.0'],
        )
        self.assertEqual(verify.call_count, 2)
        commit.assert_called_once_with(plan)

    def test_cohere_repairs_numpy_pandas_and_sklearn_together(self):
        plan = self.plan('cohere-transcribe')
        failures = [
            backend_installer.ImportProbe(
                'pandas', False,
                '/usr/lib/python3/dist-packages/pandas/__init__.py',
                traceback='numpy.dtype size changed',
                numpy_version='1.26.4',
                numpy_origin='/usr/lib/python3/dist-packages/numpy/__init__.py',
                distributions=('pandas',)),
            backend_installer.ImportProbe(
                'sklearn', False,
                '/usr/lib/python3/dist-packages/sklearn/__init__.py',
                traceback='_ARRAY_API not found',
                numpy_version='1.26.4',
                numpy_origin='/usr/lib/python3/dist-packages/numpy/__init__.py',
                distributions=('scikit-learn',)),
        ]
        failed = backend_installer.DependencyVerification(
            False, failures=failures)
        passed = backend_installer.DependencyVerification(True)
        completed = types.SimpleNamespace(
            returncode=0, stdout='', stderr='')
        with (
            mock.patch.object(
                backend_installer, '_verify_dependency_plan_detailed',
                side_effect=(failed, passed)) as verify,
            mock.patch.object(
                backend_installer, 'run_command',
                return_value=completed) as run,
        ):
            result = backend_installer._verify_and_repair_dependency_plan(
                plan, Path('/managed/venv/bin/pip'))
        self.assertTrue(result.ok)
        run.assert_called_once()
        self.assertEqual(
            run.call_args.args[0],
            [
                '/managed/venv/bin/pip', 'install', '--ignore-installed',
                'numpy>=1.26.0', 'pandas', 'scikit-learn',
            ],
        )
        self.assertEqual(verify.call_count, 2)

    def test_failed_repair_persists_diagnostic_before_rollback(self):
        plan = self.plan('rest-api')
        probe = backend_installer.ImportProbe(
            'soxr', False, '/usr/lib/python/site-packages/soxr/__init__.py',
            stderr='_ARRAY_API not found', distributions=('soxr',))
        failed = backend_installer.DependencyVerification(False, failures=[probe])
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / 'venv'
            venv.mkdir()
            (venv / 'old-marker').touch()

            def setup(**_kwargs):
                (venv / 'bin').mkdir(parents=True)
                (venv / 'bin' / 'pip').touch()
                return venv / 'bin' / 'pip'

            completed = types.SimpleNamespace(returncode=0, stdout='', stderr='')
            events = []
            with (
                mock.patch.object(backend_installer, 'VENV_DIR', venv),
                mock.patch.object(backend_installer, 'get_state', return_value=''),
                mock.patch.object(backend_installer, 'setup_python_venv', side_effect=setup),
                mock.patch.object(backend_installer, '_snapshot_inherited_dependencies'),
                mock.patch.object(backend_installer, '_verify_dependency_plan_detailed',
                                  side_effect=(failed, failed)),
                mock.patch.object(backend_installer, '_repairable_distributions',
                                  return_value=['soxr']),
                mock.patch.object(backend_installer, 'run_command', return_value=completed),
                mock.patch.object(backend_installer, 'set_install_state',
                                  side_effect=lambda state, diagnostic: events.append((state, diagnostic))),
            ):
                with self.assertRaisesRegex(RuntimeError, 'Dependency verification failed'):
                    backend_installer.execute_dependency_plan(plan)
            self.assertTrue((venv / 'old-marker').exists())
        self.assertEqual(events[0][0], 'failed')
        self.assertIn('Import: soxr', events[0][1])

    def test_repair_pip_failure_preserves_probe_evidence(self):
        plan = self.plan('rest-api')
        probe = backend_installer.ImportProbe(
            'soxr', False, '/usr/lib/python/site-packages/soxr/__init__.py',
            stderr='_ARRAY_API not found', distributions=('soxr',))
        failed = backend_installer.DependencyVerification(False, failures=[probe])
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / 'venv'

            def setup(**_kwargs):
                (venv / 'bin').mkdir(parents=True)
                (venv / 'bin' / 'pip').touch()
                return venv / 'bin' / 'pip'

            install_ok = types.SimpleNamespace(returncode=0, stdout='', stderr='')
            repair_failed = types.SimpleNamespace(
                returncode=1, stdout='', stderr='network unavailable')
            states = []
            with (
                mock.patch.object(backend_installer, 'VENV_DIR', venv),
                mock.patch.object(backend_installer, 'get_state', return_value=''),
                mock.patch.object(backend_installer, 'setup_python_venv', side_effect=setup),
                mock.patch.object(backend_installer, '_snapshot_inherited_dependencies',
                                  return_value=[]),
                mock.patch.object(backend_installer, '_verify_dependency_plan_detailed',
                                  return_value=failed),
                mock.patch.object(backend_installer, '_repairable_distributions',
                                  return_value=['soxr']),
                mock.patch.object(backend_installer, 'run_command',
                                  side_effect=(install_ok, repair_failed)),
                mock.patch.object(backend_installer, 'set_install_state',
                                  side_effect=lambda state, error: states.append((state, error))),
            ):
                with self.assertRaisesRegex(RuntimeError, 'network unavailable'):
                    backend_installer.execute_dependency_plan(plan)
        self.assertIn('_ARRAY_API not found', states[0][1])
        self.assertIn('Repair pip failure', states[0][1])

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
