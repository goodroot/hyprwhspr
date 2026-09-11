"""Second-review lifecycle regressions; host operations and downloads are mocked."""
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import unittest
from unittest import mock

from tests.test_managed_install import ManagedFixture
from tests.test_setup_command_scope import install as cli_install, uninstall, systemd
from tests.test_noctalia_integration import noctalia
import backend_installer as backend
import managed_install as managed
import managed_integrations as integrations
from legacy_units import generated_unit

ROOT = Path(__file__).resolve().parents[1]


class SecondReviewTests(ManagedFixture):
    def test_managed_binding_is_stable_and_old_generation_binding_is_repaired(self):
        directory = self.root / 'config/hypr'
        directory.mkdir(parents=True)
        path = directory / 'bindings.conf'
        old = self.install.data / 'releases/old'
        path.write_text(f'bindd = SUPER, F9, Dictation, exec, {old}/config/hyprland/hyprwhspr-tray.sh record\n')
        with mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}'}), mock.patch.object(cli_install, 'HYPRWHSPR_ROOT', str(old)), mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(integrations, 'Installation', return_value=self.install), mock.patch.object(cli_install.shutil, 'which', return_value=None):
            self.assertTrue(cli_install._setup_hyprland_bindings())
            first = path.read_text()
            self.assertTrue(cli_install._setup_hyprland_bindings())
        self.assertEqual(path.read_text(), first)
        self.assertIn('SUPER, F9, Dictation', first)
        self.assertIn(shlex.quote(str(self.install.data / 'launcher')) + ' --managed-tray record', first)
        self.assertNotIn('/releases/', first)

    def test_update_plans_migration_of_recorded_binding_before_old_root_cleanup(self):
        path = self.root / 'bindings.conf'
        path.write_text(f'bind = SUPER, D, exec, {self.install.data}/releases/old/config/hyprland/hyprwhspr-tray.sh record\n')
        managed.atomic_json(self.install.receipt, {'files': {str(path): {'kind': 'integration', 'sha256': managed.digest(path)}}})
        changes = self.install.binding_migrations()
        self.assertIn('--managed-tray record', changes[path])
        path.write_text('user changed the binding')
        self.assertEqual(self.install.binding_migrations(), {})

    def test_managed_gpu_selection_uses_detection_and_preflight_probes(self):
        for selected, variant in [('onnx-asr', 'gpu'), ('faster-whisper', 'cuda'), ('nvidia', 'nvidia'), ('amd', 'amd'), ('vulkan', 'vulkan')]:
            with self.subTest(selected=selected), mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}'}), mock.patch.object(backend, 'resolve_dependency_plan'), mock.patch.object(backend, '_managed_select') as select, mock.patch.object(backend, '_detect_nvidia_gpu_listing', return_value='GPU'), mock.patch.object(backend, 'setup_nvidia_support', return_value=True), mock.patch.object(backend, 'setup_amd_support', return_value=True), mock.patch.object(backend, 'setup_vulkan_support', return_value=True):
                self.assertTrue(backend.install_backend(selected))
                select.assert_called_once_with(selected, None, False, variant=variant)

    def test_missing_acceleration_falls_back_before_building(self):
        for selected, probe in [('nvidia', 'setup_nvidia_support'), ('amd', 'setup_amd_support'), ('vulkan', 'setup_vulkan_support')]:
            with self.subTest(selected=selected), mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}'}), mock.patch.object(backend, 'resolve_dependency_plan'), mock.patch.object(backend, probe, return_value=False), mock.patch.object(backend, '_managed_select') as select:
                self.assertTrue(backend.install_backend(selected))
                select.assert_called_once_with('cpu', None, False, variant='cpu')

    def test_accelerated_build_failure_retries_fresh_cpu_selection(self):
        for name, variant in [('nvidia', 'nvidia'), ('onnx-asr', 'gpu'), ('faster-whisper', 'cuda')]:
            selection = {'backend': name, 'variant': variant}
            with mock.patch.object(self.install, 'build', side_effect=[subprocess.CalledProcessError(1, ['pip']), {'path': 'cpu'}]) as build:
                environment, effective = self.install.build_backend(ROOT, {}, selection, {}, {}, False)
            self.assertEqual(environment, {'path': 'cpu'})
            self.assertEqual(effective['variant'], 'cpu' if name == 'nvidia' else None)
            self.assertEqual(selection['variant'], variant)
            self.assertTrue(build.call_args_list[1].args[-1])

    def test_disk_or_packaging_failure_does_not_trigger_cpu_fallback(self):
        for error in (OSError('disk full'), ValueError('bad manifest')):
            with mock.patch.object(self.install, 'build', side_effect=error) as build:
                with self.assertRaises(type(error)):
                    self.install.build_backend(ROOT, {}, {'backend': 'onnx-asr', 'variant': 'gpu'}, {}, {}, False)
                build.assert_called_once()

    def test_local_backend_rebuild_never_fetches_release(self):
        root = self.install.data / 'releases/current'
        root.mkdir(parents=True)
        old_cli = self.install.data / 'environments/cli'
        old_backend = self.install.data / 'environments/old-backend'
        old_cli.mkdir(parents=True)
        old_backend.mkdir()
        old = {'version': 'v1.2.3', 'root': str(root), 'python': {'path': '/python'},
               'cli': {'path': str(old_cli)}, 'backend': {'path': str(old_backend)},
               'selection': {'backend': 'onnx-asr', 'variant': 'gpu'}}
        managed.atomic_json(self.install.current, old)
        attempted = []
        def build(payload, identity, kind, selection, previous, tx, force):
            if kind == 'cli':
                return old['cli']
            target = self.install.data / 'environments' / ('build-' + str(len(attempted)))
            attempted.append(target)
            tx['created'].append(str(target))
            managed.atomic_json(self.install.journal, tx)
            target.mkdir()
            if selection.get('variant') == 'gpu':
                raise subprocess.CalledProcessError(1, ['GPU build'])
            return {'path': str(target)}
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(managed, 'interpreter', return_value=old['python']))
            stack.enter_context(mock.patch.object(managed, 'verify_payload', return_value={'version': old['version']}))
            release = stack.enter_context(mock.patch.object(self.install, 'release', side_effect=AssertionError('network forbidden')))
            stack.enter_context(mock.patch.object(Path, 'rglob', side_effect=FileNotFoundError('live daemon temp disappeared')))
            stack.enter_context(mock.patch.object(self.install, 'build', side_effect=build))
            for name, result in [('validate_config', None), ('check_integrations', None), ('service_state', {}), ('daemon_running', False), ('install_launcher', None), ('clean_legacy', None)]:
                stack.enter_context(mock.patch.object(self.install, name, return_value=result))
            current = self.install.update(force_backend=True, local_payload=True)
        release.assert_not_called()
        self.assertEqual(current['root'], old['root'])
        self.assertIsNone(current['selection']['variant'])
        self.assertFalse(attempted[0].exists())
        self.assertTrue(attempted[1].exists())
        self.assertFalse(old_backend.exists())
        self.assertTrue(old_cli.exists())

    def test_backend_repair_launcher_selects_local_rebuild(self):
        managed.atomic_json(self.install.current, {'root': str(ROOT)})
        with mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(self.install, 'update') as update:
            self.assertEqual(managed.launch(['backend', 'repair']), 0)
        update.assert_called_once_with(force_backend=True, local_payload=True)

    def test_historical_main_and_resume_units_are_recognized_exactly(self):
        catalog = json.loads((ROOT / 'share/legacy-systemd-units.json').read_text())
        for name, templates in catalog['units'].items():
            for template in templates:
                with self.subTest(name=name, commit=template['commit']):
                    path = self.root / name
                    content = template['content'].replace('/usr/lib/hyprwhspr', str(self.root / 'old checkout'))
                    path.write_text(content)
                    self.assertTrue(generated_unit(path, self.root / 'old checkout'))
                    path.write_text(content + '\nEnvironment=CUSTOM=value\n')
                    self.assertFalse(generated_unit(path, self.root / 'old checkout'))

    def test_service_stop_and_reload_failures_explain_preserved_runtime(self):
        for failed_action in ('stop', 'daemon-reload'):
            case = self.root / failed_action
            units = case / 'units'
            units.mkdir(parents=True)
            unit = units / 'hyprwhspr.service'
            unit.write_text('recognized')
            venv = case / 'venv'
            venv.mkdir()
            command_link = case / '.local/bin/hyprwhspr'
            command_link.parent.mkdir(parents=True)
            command_link.symlink_to(case / 'data/src/bin/hyprwhspr')
            def run(command, **kwargs):
                if command[2] == failed_action:
                    raise subprocess.CalledProcessError(1, command)
                return mock.Mock(returncode=0)
            with ExitStack() as stack:
                for name, path in {'USER_HOME': case, 'USER_CONFIG_DIR': case / 'config', 'USER_SYSTEMD_DIR': units,
                                   'USER_BASE': case / 'data', 'STATE_DIR': case / 'state', 'VENV_DIR': venv,
                                   'PYWHISPERCPP_SRC_DIR': case / 'sources', 'PYWHISPERCPP_MODELS_DIR': case / 'models',
                                   'CREDENTIALS_FILE': case / 'credentials'}.items():
                    stack.enter_context(mock.patch.object(uninstall, name, path))
                stack.enter_context(mock.patch.object(uninstall, '_generated_legacy_unit', return_value=True))
                stack.enter_context(mock.patch.object(uninstall, 'run_command', side_effect=run))
                with self.assertRaisesRegex(RuntimeError, 'runtime preserved'):
                    uninstall.uninstall_command(yes=True, skip_permissions=True)
            self.assertTrue(venv.exists())
            self.assertFalse(command_link.is_symlink(), 'Independent command cleanup must still run')
            self.assertEqual(unit.exists(), failed_action == 'stop')

    def test_successful_model_download_survives_receipt_failure(self):
        models = self.root / 'models'
        def download(url, path, **kwargs):
            path.write_bytes(b'model')
        with mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}'}), mock.patch.object(backend, 'PYWHISPERCPP_MODELS_DIR', models), mock.patch.object(backend, 'check_model_validity', return_value=False), mock.patch.object(backend.urllib.request, 'urlretrieve', side_effect=download), mock.patch.object(backend, 'set_state'), mock.patch.object(managed, 'record_file', side_effect=ValueError('corrupt receipt')), mock.patch.object(backend, 'log_warning') as warning:
            self.assertTrue(backend.download_pywhispercpp_model('base'))
        self.assertEqual((models / 'ggml-base.bin').read_bytes(), b'model')
        self.assertIn('ownership could not be recorded', warning.call_args.args[0])

    def test_noctalia_disable_precedes_managed_file_removal(self):
        self.check_noctalia(True)

    def test_noctalia_disable_failure_preserves_plugin_files(self):
        self.check_noctalia(False)

    def check_noctalia(self, disable_ok):
        plugin = self.root / 'plugin'
        plugin.mkdir()
        widget = plugin / 'widget.luau'
        widget.write_text('owned widget')
        managed.atomic_json(self.install.receipt, {'files': {str(widget): {'sha256': managed.digest(widget), 'original': None}}})
        dst = {'plugin_dir': plugin, 'settings': self.root / 'settings', 'template_input': self.root / 'input', 'template_output': self.root / 'output'}
        def disable(*args):
            self.assertTrue(widget.exists(), 'Must disable before deleting the plugin')
            self.assertEqual(args, ('plugins', 'disable', noctalia.NOCTALIA_PLUGIN_ID))
            return disable_ok
        with mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}'}), mock.patch.object(noctalia, '_noctalia_paths', return_value=dst), mock.patch.object(noctalia, 'HYPRWHSPR_ROOT', str(ROOT)), mock.patch.object(noctalia.shutil, 'which', return_value='/usr/bin/noctalia'), mock.patch.object(noctalia, '_noctalia_msg', side_effect=disable), mock.patch.object(noctalia, 'log_info') as info, mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(integrations, 'Installation', return_value=self.install):
            self.assertEqual(noctalia.setup_noctalia('remove'), disable_ok)
        self.assertEqual(widget.exists(), not disable_ok)
        self.assertTrue(any('widget list' in call.args[0] for call in info.call_args_list))

    def test_truncated_interpreter_hint_does_not_abort_stable_launcher(self):
        root = self.root / 'payload'
        module = root / 'lib/src/managed_install.py'
        module.parent.mkdir(parents=True)
        module.write_text('print("resolver reached")\n')
        data = self.root / 'launcher-data/hyprwhspr'
        data.mkdir(parents=True)
        (data / 'current.json').write_text(json.dumps({'root': str(root)}))
        for hint in ('', '/missing/python'):
            (data / 'interpreter').write_text(hint)
            env = dict(os.environ, XDG_DATA_HOME=str(data.parent))
            result = subprocess.run(['bash', str(ROOT / 'scripts/managed-launcher.sh')], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('resolver reached', result.stdout)

    def test_failed_atomic_interpreter_write_preserves_old_hint(self):
        hint = self.root / 'interpreter'
        hint.write_text('/old/python\n')
        with mock.patch.object(managed.os, 'replace', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                managed.atomic_text(hint, '/new/python\n')
        self.assertEqual(hint.read_text(), '/old/python\n')
        self.assertEqual(list(self.root.iterdir()), [hint])


if __name__ == '__main__':
    unittest.main()
