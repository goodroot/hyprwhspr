"""Regression coverage for managed/legacy lifecycle review findings."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib/src'))
import managed_install as managed
import managed_integrations as integrations
import diagnostics
from tests.test_setup_command_scope import uninstall, systemd
from tests.test_managed_install import ManagedFixture


class LifecycleReviewTests(ManagedFixture):
    def test_stale_or_malformed_inherited_lock_can_acquire_normally(self):
        for inherited in ('999999', 'not-a-number'):
            with mock.patch.dict(os.environ, {'HYPRWHSPR_INSTALL_LOCK_FD': inherited}):
                with self.install.lock():
                    self.assertIsNotNone(self.install.lock_fd)
                self.assertIsNone(self.install.lock_fd)

    def test_closed_grandchild_lock_reports_busy_when_parent_owns_it(self):
        with self.install.lock():
            with mock.patch.dict(os.environ, {'HYPRWHSPR_INSTALL_LOCK_FD': '999999'}):
                with self.assertRaisesRegex(RuntimeError, 'Another installation'):
                    with managed.Installation(self.install.data, self.install.state).lock():
                        pass

    def test_valid_inherited_lock_resets_instance_state(self):
        with self.install.lock():
            other = managed.Installation(self.install.data, self.install.state)
            with mock.patch.dict(os.environ, {'HYPRWHSPR_INSTALL_LOCK_FD': str(self.install.lock_fd)}):
                with other.lock():
                    self.assertEqual(other.lock_fd, self.install.lock_fd)
                self.assertIsNone(other.lock_fd)
                self.assertNotIn('HYPRWHSPR_INSTALL_LOCK_FD', managed.clean_env())

    def test_managed_tray_receives_resolved_generation_paths(self):
        generation = {'root': str(self.root / 'release'), 'cli': {'path': '/cli'},
                      'backend': {'path': str(self.root / 'backend environment')}}
        managed.atomic_json(self.install.current, generation)
        with mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(managed.os, 'execve', side_effect=SystemExit) as execute:
            with self.assertRaises(SystemExit):
                managed.launch(['--managed-tray', 'status'])
        env = execute.call_args.args[2]
        self.assertEqual(env['HYPRWHSPR_ROOT'], generation['root'])
        self.assertEqual(env['HYPRWHSPR_BACKEND_ENV'], generation['backend']['path'])

    def test_owned_launcher_can_upgrade_and_modified_launcher_is_preserved(self):
        root = self.root / 'release'
        source = root / 'scripts/managed-launcher.sh'
        source.parent.mkdir(parents=True)
        source.write_text('#!/bin/sh\n# version one\n')
        managed.atomic_json(self.install.current, {'python': {'path': '/recorded/python'}})
        with mock.patch.object(self.install, 'command_path', return_value=self.root / 'bin/command'):
            self.install.install_launcher(root)
            source.write_text('#!/bin/sh\n# version two\n')
            self.install.install_launcher(root)
            launcher = self.install.data / 'launcher'
            self.assertEqual(launcher.read_text(), source.read_text())
            launcher.write_text('user customization')
            with self.assertRaisesRegex(RuntimeError, 'Modified or unowned'):
                self.install.install_launcher(root)
            self.assertEqual(launcher.read_text(), 'user customization')

    def test_launcher_rollback_restores_old_content_without_losing_new_receipts(self):
        launcher = self.install.data / 'launcher'
        launcher.parent.mkdir(parents=True)
        launcher.write_text('old launcher')
        original = {'sha256': managed.digest(launcher)}
        launcher.write_text('new launcher')
        personal = self.root / 'personal'
        personal.write_text('settings')
        managed.atomic_json(self.install.receipt, {'files': {str(launcher): {'sha256': managed.digest(launcher)},
            str(personal): {'sha256': managed.digest(personal), 'kind': 'personal'}}})
        tx = {'phase': 'activated', 'old': {'root': 'old'}, 'created': [],
              'running': False, 'integrations': {str(launcher): {'content': 'old launcher', 'mode': 0o755}},
              'receipt': {'files': {str(launcher): original}}}
        managed.atomic_json(self.install.journal, tx)
        with mock.patch.object(self.install, 'service_state', return_value={}):
            self.install.recover()
        self.assertEqual(launcher.read_text(), 'old launcher')
        files = managed.read_json(self.install.receipt)['files']
        self.assertEqual(files[str(launcher)], original)
        self.assertIn(str(personal), files)

    def test_config_validation_failure_includes_captured_reason(self):
        config = self.root / 'config/hyprwhspr/config.json'
        config.parent.mkdir(parents=True)
        config.write_text('{}')
        failure = subprocess.CalledProcessError(1, ['validator'], stderr=b'audio_volume exceeds maximum')
        with mock.patch.object(managed, 'run', side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, 'audio_volume exceeds maximum'):
                self.install.validate_config(ROOT, {'path': '/cli'})
        self.assertEqual(config.read_text(), '{}')

    def test_noninteractive_fresh_install_reports_success_and_setup_guidance(self):
        with mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(self.install, 'update') as update, mock.patch('builtins.open', side_effect=OSError('no tty')), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(managed.main(['bootstrap']), 0)
        update.assert_called_once()
        self.assertIn('Installation completed', output.getvalue())
        self.assertIn('hyprwhspr setup', output.getvalue())

    def test_concurrent_receipt_updates_merge_with_integration_editor(self):
        paths = [self.root / f'file-{i}' for i in range(16)]
        for path in paths:
            path.write_text('content')
        style = self.root / 'style'
        style.write_text('before\n')
        with mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(integrations, 'Installation', return_value=self.install):
            with integrations.edit_files([style], [style]):
                style.write_text('added\nbefore\n')
                with ThreadPoolExecutor(max_workers=8) as executor:
                    list(executor.map(managed.record_file, paths))
        entries = managed.read_json(self.install.receipt)['files']
        self.assertEqual(set(entries), {str(path) for path in paths + [style]})

    def test_corrupt_receipt_does_not_make_config_or_credential_save_fail(self):
        from config_manager import ConfigManager
        import credential_manager
        config = object.__new__(ConfigManager)
        config.config_dir = self.root / 'personal'
        config.config_dir.mkdir()
        config.config_file = config.config_dir / 'config.json'
        config.default_config = {}
        config.config = {'recording_mode': 'toggle'}
        config.verbose = False
        managed.atomic_json(self.install.receipt, {'files': None})
        with mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}'}), mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(credential_manager, 'CREDENTIALS_FILE', self.root / 'credentials'), mock.patch.object(credential_manager, '_ensure_credentials_dir'), redirect_stdout(io.StringIO()):
            self.assertTrue(config.save_config())
            credential_manager._save_credentials({'provider': 'private'})
        self.assertTrue(config.config_file.exists())
        self.assertEqual(json.loads((self.root / 'credentials').read_text()), {'provider': 'private'})


class LegacyUninstallReviewTests(ManagedFixture):
    def run_uninstall(self, *, unit=None, foreign_link=False, purge=False, keep=False, permissions=False):
        from cli import _shared
        checkout = self.root / 'clone elsewhere'
        template = checkout / 'config/systemd/hyprwhspr.service'
        template.parent.mkdir(parents=True)
        template.write_text('ExecStart=/usr/lib/hyprwhspr/bin/hyprwhspr\n')
        user_units = self.root / 'units'
        user_units.mkdir()
        unit_path = user_units / 'hyprwhspr.service'
        if unit:
            unit_path.write_text(template.read_text().replace('/usr/lib/hyprwhspr', str(checkout)) if unit == 'generated' else 'custom unit')
        command = self.root / '.local/bin/hyprwhspr'
        command.parent.mkdir(parents=True)
        command.symlink_to('/foreign/bin/hyprwhspr' if foreign_link else checkout / 'bin/hyprwhspr')
        venv = self.root / 'venv'
        venv.mkdir()
        model = self.root / 'models/owned.bin'
        model.parent.mkdir()
        model.write_text('owned model')
        external = model.with_name('external.bin')
        external.write_text('external model')
        managed.atomic_json(self.install.receipt, {'files': {str(model): {'sha256': managed.digest(model), 'kind': 'model'}},
            'permissions': [{'kind': 'group', 'group': 'input', 'user': 'test-user', 'added': True},
                            {'kind': 'group', 'group': 'audio', 'user': 'test-user', 'added': False}]})
        patches = {'USER_HOME': self.root, 'USER_SYSTEMD_DIR': user_units,
                   'USER_CONFIG_DIR': self.root / 'config', 'VENV_DIR': venv,
                   'USER_BASE': self.install.data, 'STATE_DIR': self.install.state,
                   'PYWHISPERCPP_SRC_DIR': self.root / 'source', 'PYWHISPERCPP_MODELS_DIR': model.parent,
                   'CREDENTIALS_FILE': self.root / 'credentials'}
        with ExitStack() as stack:
            for key, value in patches.items():
                stack.enter_context(mock.patch.object(uninstall, key, value))
            stack.enter_context(mock.patch.object(_shared, 'HYPRWHSPR_ROOT', str(checkout)))
            command_mock = stack.enter_context(mock.patch.object(uninstall, 'run_command', return_value=mock.Mock(returncode=0)))
            sudo = stack.enter_context(mock.patch.object(uninstall, 'run_sudo_command', return_value=mock.Mock(returncode=0)))
            output = stack.enter_context(redirect_stdout(io.StringIO()))
            uninstall.uninstall_command(yes=True, purge=purge, keep_models=keep,
                                        remove_permissions=permissions, skip_permissions=not permissions)
        return unit_path, command, venv, model, external, command_mock, sudo, output.getvalue()

    def test_generated_unit_and_clone_anywhere_link_are_removed(self):
        unit, command, venv, _, _, calls, _, _ = self.run_uninstall(unit='generated')
        self.assertFalse(unit.exists())
        self.assertFalse(command.is_symlink())
        self.assertFalse(venv.exists())
        self.assertIn(mock.call(['systemctl', '--user', 'stop', 'hyprwhspr.service'], check=True), calls.call_args_list)

    def test_customized_unit_is_preserved_without_aborting_cleanup(self):
        unit, command, venv, _, _, _, _, output = self.run_uninstall(unit='custom')
        self.assertTrue(unit.exists())
        self.assertTrue(venv.exists())
        self.assertFalse(command.is_symlink())
        self.assertIn('Preserving customized', output)

    def test_foreign_command_symlink_is_preserved(self):
        _, command, *_ = self.run_uninstall(foreign_link=True)
        self.assertTrue(command.is_symlink())

    def test_legacy_flags_apply_only_to_recorded_ownership(self):
        with mock.patch.object(managed, 'group_membership_present', return_value=True):
            _, _, _, model, external, _, sudo, _ = self.run_uninstall(purge=True, permissions=True)
        self.assertFalse(model.exists())
        self.assertTrue(external.exists())
        sudo.assert_called_once_with(['gpasswd', '-d', 'test-user', 'input'], check=True)

    def test_keep_models_and_skip_permissions_are_effective(self):
        _, _, _, model, _, _, sudo, output = self.run_uninstall(purge=True, keep=True)
        self.assertTrue(model.exists())
        sudo.assert_not_called()
        self.assertIn('--keep-models', output)
        self.assertIn('--skip-permissions', output)

    def test_managed_service_conflict_returns_false(self):
        root = self.root / 'release'
        (root / 'bin').mkdir(parents=True)
        executable = root / 'bin/hyprwhspr'
        executable.write_text('launcher')
        executable.chmod(0o755)
        (root / 'config/systemd').mkdir(parents=True)
        (root / 'config/systemd/hyprwhspr.service').write_text('[Service]\nExecStart=/usr/lib/hyprwhspr/bin/hyprwhspr\n')
        with mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}'}), mock.patch.object(systemd, 'HYPRWHSPR_ROOT', str(root)), mock.patch.object(systemd, 'USER_SYSTEMD_DIR', self.root / 'units'), mock.patch.object(systemd, '_validate_hyprwhspr_root', return_value=True), mock.patch.object(managed, 'write_owned', side_effect=RuntimeError('unowned service')), mock.patch.object(systemd, 'run_command') as run, mock.patch.object(systemd, 'log_error') as error:
            self.assertFalse(systemd.setup_systemd())
        run.assert_not_called()
        self.assertTrue(any('Inspect and back up' in call.args[0] and 'hyprwhspr systemd install' in call.args[0] for call in error.call_args_list))


class MetadataAndTrayReviewTests(ManagedFixture):
    def test_malformed_packaged_version_degrades_without_git(self):
        spec = importlib.util.spec_from_file_location('review_cli', ROOT / 'lib/cli.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        for content in ('{', '{}', '{"version":null}', '[]'):
            (self.root / 'release.json').write_text(content)
            with mock.patch.object(cli, '__file__', str(self.root / 'lib/cli.py')), mock.patch.object(subprocess, 'run') as run:
                self.assertEqual(cli._get_version(), 'unknown')
                run.assert_not_called()

    def test_diagnostics_uses_managed_interpreter_and_release_metadata(self):
        (self.root / 'release.json').write_text('{"version":"v1.2.3"}')
        interpreter = self.root / 'backend environment/bin/python'
        calls = []
        def probe(args):
            calls.append(args)
            if '-c' in args:
                modules = args[3:]
                return mock.Mock(returncode=0, stdout=json.dumps({'python': [3, 12, 0],
                    'modules': {name: True for name in modules}, 'metadata': {name: True for name in modules}}))
            return mock.Mock(returncode=0, stdout='inactive')
        with mock.patch.dict(os.environ, {'HYPRWHSPR_BACKEND_ENV': str(interpreter.parent.parent)}), mock.patch.object(diagnostics, 'ROOT', self.root), mock.patch.object(diagnostics, 'validate_config', return_value=(diagnostics.build_default_config(), [], False)), mock.patch.object(diagnostics.Probes, 'run', side_effect=probe), mock.patch.object(diagnostics.Probes, 'live', return_value=None), mock.patch.object(diagnostics.shutil, 'which', return_value=None):
            report, _ = diagnostics.build_report()
        self.assertEqual(calls[0][0], str(interpreter))
        self.assertFalse(any(call[0] == 'git' for call in calls))
        self.assertEqual(report['application_version'], 'v1.2.3')

    def test_tray_backend_probes_do_not_require_legacy_venv(self):
        source = (ROOT / 'config/hyprland/hyprwhspr-tray.sh').read_text()
        function = source[source.index('model_exists() {'):source.index('mic_present() {')]
        interpreter = self.root / 'backend environment/bin/python'
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text('#!/bin/sh\nexit 0\n')
        interpreter.chmod(0o755)
        config = self.root / '.config/hyprwhspr/config.json'
        config.parent.mkdir(parents=True)
        for backend in ('onnx-asr', 'faster-whisper'):
            config.write_text(json.dumps({'transcription_backend': backend}))
            env = os.environ.copy()
            env.update(HOME=str(self.root), SYSTEM_PYTHON=sys.executable,
                       HYPRWHSPR_BACKEND_ENV=str(interpreter.parent.parent))
            result = subprocess.run(['bash', '-c', function + '\nmodel_exists'], env=env, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
