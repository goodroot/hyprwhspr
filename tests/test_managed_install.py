"""Lifecycle tests never touch a desktop, network, or real user directories."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib/src'))
import managed_install as managed


class ManagedFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='managed install ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.install = managed.Installation(self.root / 'data', self.root / 'state')
        self.env = mock.patch.dict(os.environ, {'HOME': str(self.root), 'XDG_DATA_HOME': str(self.root / 'xdg-data'),
            'XDG_STATE_HOME': str(self.root / 'xdg-state'), 'XDG_CONFIG_HOME': str(self.root / 'config'),
            'XDG_RUNTIME_DIR': str(self.root / 'runtime')})
        self.env.start()
        self.addCleanup(self.env.stop)


class ManagedInstallTests(ManagedFixture):
    def archive(self, name, kind=tarfile.REGTYPE):
        archive = self.root / 'payload.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.linkname = '/tmp/elsewhere'
            member.size = 1 if kind == tarfile.REGTYPE else 0
            tar.addfile(member, io.BytesIO(b'x') if member.size else None)
        return archive

    def test_unsafe_archives_rejected_before_extraction(self):
        for name, kind in [('../escape', tarfile.REGTYPE), ('/absolute', tarfile.REGTYPE),
                           ('link', tarfile.SYMTYPE), ('hard', tarfile.LNKTYPE), ('device', tarfile.CHRTYPE)]:
            with self.subTest(name=name):
                archive = self.archive(name, kind)
                with self.assertRaises(ValueError):
                    managed.extract(archive, self.root / 'unpacked', managed.digest(archive))
                self.assertFalse((self.root / 'unpacked').exists())

    def test_checksum_failure_writes_nothing(self):
        archive = self.archive('file')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            managed.extract(archive, self.root / 'unpacked', '0' * 64)
        self.assertFalse((self.root / 'unpacked').exists())

    def test_interpreter_probe_resolves_manager_executable(self):
        for manager in ('uv', 'mise', 'pyenv'):
            resolved = {'path': f'/managed/{manager}/python', 'version': [3, 12, 2], 'identity': '3.12.2'}
            with mock.patch.object(managed, 'run', return_value=mock.Mock(stdout=json.dumps(resolved))) as run:
                self.assertEqual(managed.interpreter(f'/shims/{manager}'), resolved)
                self.assertIn('-I', run.call_args.args[0])

    def test_environment_disables_inherited_python_and_pip_locations(self):
        with mock.patch.dict(os.environ, {'PYTHONPATH': '/hostile', 'PYTHONHOME': '/hostile',
                'PIP_TARGET': '/shared', 'VIRTUAL_ENV': '/active', 'MISE_DATA_DIR': '/manager'}):
            env = managed.clean_env()
            for key in ('PYTHONPATH', 'PYTHONHOME', 'PIP_TARGET', 'VIRTUAL_ENV'):
                self.assertNotIn(key, env)
            self.assertEqual(env['PYTHONNOUSERSITE'], '1')
            self.assertEqual(env['MISE_DATA_DIR'], '/manager')

    def test_status_is_read_only(self):
        self.assertIsNone(self.install.status()['generation'])
        self.assertFalse(self.install.state.exists())
        self.assertFalse(self.install.data.exists())

    def test_concurrent_mutations_fail(self):
        with self.install.lock():
            with self.assertRaisesRegex(RuntimeError, 'Another installation'):
                with managed.Installation(self.install.data, self.install.state).lock():
                    pass

    def transaction(self, phase):
        old = {'root': 'old'}
        staged = self.install.data / 'environments/new'
        staged.mkdir(parents=True)
        tx = {'phase': phase, 'old': old, 'running': True, 'created': [str(staged)], 'garbage': [str(staged)]}
        managed.atomic_json(self.install.current, {'root': 'new'})
        managed.atomic_json(self.install.journal, tx)
        return old, staged

    def test_recovery_at_every_boundary(self):
        for phase in ('staging', 'activating', 'activated', 'committed'):
            with self.subTest(phase=phase):
                old, staged = self.transaction(phase)
                with mock.patch.object(self.install, 'service_state', return_value={'ActiveState': 'active'}), mock.patch.object(self.install, 'service') as service:
                    self.install.recover()
                self.assertFalse(staged.exists())
                self.assertFalse(self.install.journal.exists())
                expected = old if phase in ('activating', 'activated') else {'root': 'new'}
                self.assertEqual(managed.read_json(self.install.current), expected)
                if phase in ('activating', 'activated'):
                    self.assertEqual(service.call_args_list, [mock.call('stop'), mock.call('start')])

    def test_lifecycle_launch_resolves_generation_after_rollback(self):
        generations = []
        for name in ('old', 'new'):
            root = self.install.data / 'releases' / name
            environment = self.install.data / 'environments' / name
            root.mkdir(parents=True)
            (environment / 'bin').mkdir(parents=True)
            (environment / 'bin/python').write_text('placeholder', encoding='utf-8')
            generations.append({'root': str(root), 'cli': {'path': str(environment)},
                                'xdg': {'config': str(self.root / name)}})
        old, new = generations
        managed.atomic_json(self.install.current, new)
        managed.atomic_json(self.install.journal, {
            'phase': 'activated', 'old': old, 'created': [new['root'], new['cli']['path']]})

        def child(command, **kwargs):
            self.assertFalse(Path(new['cli']['path']).exists())
            self.assertEqual(managed.read_json(self.install.current), old)
            self.assertEqual(command[0], str(Path(old['cli']['path']) / 'bin/python'))
            self.assertEqual(command[2], str(Path(old['root']) / 'lib/cli.py'))
            self.assertEqual(json.loads(kwargs['env']['HYPRWHSPR_GENERATION']), old)
            self.assertEqual(kwargs['env']['XDG_CONFIG_HOME'], old['xdg']['config'])
            self.assertEqual(kwargs['pass_fds'], (self.install.lock_fd,))
            self.assertIsNotNone(self.install.lock_fd)
            return subprocess.CompletedProcess(command, 0)

        with mock.patch.object(managed, 'Installation', return_value=self.install), \
             mock.patch.dict(os.environ, {'HYPRWHSPR_RESOLVED_GENERATION': json.dumps(new)}), \
             mock.patch.object(self.install, 'stop_for_recovery', return_value=True), \
             mock.patch.object(self.install, 'daemon_paths', return_value=set()), \
             mock.patch.object(managed.subprocess, 'run', side_effect=child) as execute:
            self.assertEqual(managed.launch(['config']), 0)
        execute.assert_called_once()
        self.assertFalse(self.install.journal.exists())

    def test_cleanup_failure_retires_journal_and_queues_retry(self):
        _, staged = self.transaction('committed')
        with mock.patch.object(managed.shutil, 'rmtree', side_effect=OSError('disk failure')):
            with self.assertRaisesRegex(RuntimeError, 'Cleanup incomplete'):
                self.install.recover(strict=True)
        self.assertFalse(self.install.journal.exists())
        self.assertIn(str(staged), managed.read_json(self.install.state / 'deferred-cleanup.json'))
        self.install.recover()
        self.assertFalse(staged.exists())

    def test_no_cleanup_outside_owned_paths(self):
        shared = self.root / 'shared-models'
        shared.mkdir()
        with self.assertRaises(RuntimeError):
            self.install.remove([str(shared)])
        self.assertTrue(shared.exists())

    def test_unexpected_service_failure_restores_pointer_and_preserves_runtime(self):
        old, staged = self.transaction('activated')
        with mock.patch.object(self.install, 'service_state', return_value={'ActiveState': 'active'}), mock.patch.object(self.install, 'service', side_effect=RuntimeError('service failed')):
            self.install.recover()
        self.assertFalse(self.install.journal.exists())
        self.assertEqual(managed.read_json(self.install.current), old)
        self.assertTrue(staged.exists())
        self.assertIn(str(staged), managed.read_json(self.install.state / 'deferred-cleanup.json'))

    def test_foreign_command_conflict(self):
        command = self.root / 'command'
        command.symlink_to('/foreign/bin/hyprwhspr')
        with mock.patch.object(self.install, 'command_path', return_value=command):
            with self.assertRaisesRegex(RuntimeError, 'Conflicting command'):
                self.install.check_integrations()
        self.assertEqual(os.readlink(command), '/foreign/bin/hyprwhspr')

    def test_noop_update_does_not_restart_service(self):
        old_root = self.install.data / 'releases/old'
        old_root.mkdir(parents=True)
        old = {'version': 'v1.0.0', 'root': str(old_root), 'python': {'path': '/python'},
               'cli': {'path': str(self.install.data / 'environments/cli')},
               'backend': None, 'selection': None, 'xdg': self.install.xdg_roots()}
        managed.atomic_json(self.install.current, old)
        def extract(archive, root, checksum):
            root.mkdir(parents=True)
            (root / 'release.json').write_text(json.dumps({'version': old['version']}))
        with mock.patch.object(managed, 'interpreter', return_value=old['python']), mock.patch.object(managed, 'extract', side_effect=extract), mock.patch.object(managed, 'verify_payload'), mock.patch.object(self.install, 'release', return_value={'version': old['version'], 'sha256': '0'*64}), mock.patch.object(self.install, 'build', return_value=old['cli']), mock.patch.object(self.install, 'validate_config'), mock.patch.object(self.install, 'service') as service:
            self.assertEqual(self.install.update(), old)
            service.assert_not_called()
        self.assertEqual(list((self.install.data / 'releases').iterdir()), [old_root])

    def test_missing_recorded_interpreter_fails(self):
        with mock.patch.object(managed, 'run', side_effect=FileNotFoundError):
            with self.assertRaisesRegex(RuntimeError, '--python'):
                managed.interpreter('/missing/python')

    def test_healthy_environment_is_reused(self):
        root = Path(__file__).resolve().parents[1]
        from dependency_manifest import fingerprint, parse_graph
        identity = {'path': '/python', 'identity': 'test'}
        previous = {'path': '/existing', 'key': {'python': identity,
            'dependencies': fingerprint(parse_graph(root / 'requirements-cli.txt', ValueError).manifests), 'selection': None}}
        with mock.patch.object(self.install, 'healthy', return_value=True), mock.patch.object(managed, 'run') as run:
            result = self.install.build(root, identity, 'cli', None, {'cli': previous}, {}, False)
        self.assertEqual(result, previous)
        run.assert_not_called()

    def test_failed_download_preserves_active_generation(self):
        old = {'version': 'v1.0.0', 'python': {'path': '/python'}}
        managed.atomic_json(self.install.current, old)
        with mock.patch.object(managed, 'interpreter', return_value=old['python']), mock.patch.object(self.install, 'release', side_effect=OSError('download failed')):
            with self.assertRaises(OSError):
                self.install.update()
        self.assertEqual(managed.read_json(self.install.current), old)
        self.assertFalse(self.install.journal.exists())

    def test_modified_integration_is_preserved(self):
        with mock.patch.object(managed, 'Installation', return_value=self.install):
            path = self.root / 'unit'
            managed.write_owned(path, 'owned')
            path.write_text('customized')
            with self.assertRaisesRegex(RuntimeError, 'preserved'):
                managed.write_owned(path, 'replacement')
            self.assertEqual(path.read_text(), 'customized')


class TransactionAcceptanceTests(ManagedFixture):
    def exercise_update(self, *, failure=None, repair=False, fresh=False):
        from contextlib import ExitStack
        old_root = self.install.data / 'releases/old'
        old_cli = self.install.data / 'environments/old-cli'
        old_backend = self.install.data / 'environments/old-backend'
        for path in (old_root, old_cli, old_backend):
            path.mkdir(parents=True)
        old = {'format': 1, 'version': 'v1.0.0', 'root': str(old_root),
               'python': {'path': '/recorded/python'}, 'cli': {'path': str(old_cli)},
               'backend': {'path': str(old_backend)}, 'selection': {'backend': 'cpu'}}
        if not fresh:
            managed.atomic_json(self.install.current, old)
        def extract(archive, root, checksum):
            root.mkdir(parents=True)
            (root / 'release.json').write_text(json.dumps({'version': 'v2.0.0'}))
        def build(root, identity, kind, selection, previous, tx, force):
            path = self.install.data / 'environments' / ('new-' + kind)
            tx['created'].append(str(path))
            managed.atomic_json(self.install.journal, tx)
            path.mkdir(parents=True)
            return {'path': str(path)}
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(managed, 'interpreter', return_value=old['python']))
            stack.enter_context(mock.patch.object(managed, 'extract', side_effect=extract))
            stack.enter_context(mock.patch.object(managed, 'run'))
            stack.enter_context(mock.patch.object(managed, 'verify_payload'))
            mocks = {}
            for name, value in {'release': {'version': 'v2.0.0', 'sha256': '0'*64},
                    'validate_config': None, 'check_integrations': None, 'install_launcher': None,
                    'stable': None, 'clean_legacy': None, 'legacy_selection': None, 'package_installed': False,
                    'daemon_running': False, 'service_state': {'ActiveState': 'active'}}.items():
                mocks[name] = stack.enter_context(mock.patch.object(self.install, name, return_value=value))
            mocks['build'] = stack.enter_context(mock.patch.object(self.install, 'build', side_effect=build))
            mocks['service'] = stack.enter_context(mock.patch.object(self.install, 'service'))
            if failure:
                mocks[failure].side_effect = RuntimeError(failure + ' failed')
                with self.assertRaisesRegex(RuntimeError, failure):
                    self.install.update('v2.0.0', repair=repair)
                self.assertEqual(managed.read_json(self.install.current), None if fresh else old)
                self.assertTrue(old_root.exists())
                self.assertTrue(old_backend.exists())
                return
            new = self.install.update('v2.0.0', repair=repair)
            self.assertEqual(managed.read_json(self.install.current), new)
            self.assertTrue(Path(new['cli']['path']).exists())
            if not fresh:
                self.assertFalse(old_root.exists())
                self.assertFalse(old_backend.exists())
            self.assertFalse(self.install.journal.exists())
            self.assertFalse(any((self.install.data / 'transactions').iterdir()))
            mocks['release'].assert_called_once()
            if repair:
                self.assertTrue(all(call.args[-1] for call in mocks['build'].call_args_list))

    def test_success_leaves_only_selected_runtime(self):
        self.exercise_update()

    def test_repair_rebuilds_environments(self):
        self.exercise_update(repair=True)

    def test_fresh_install_can_have_no_backend(self):
        self.exercise_update(fresh=True)
        self.assertIsNone(managed.read_json(self.install.current)['backend'])

    def test_failures_before_and_after_activation_restore_old(self):
        for failure in ('release', 'build', 'validate_config', 'install_launcher', 'stable'):
            with self.subTest(failure=failure):
                # Every failure gets a fully isolated installation.
                self.install = managed.Installation(self.root / failure / 'data', self.root / failure / 'state')
                self.exercise_update(failure=failure)


class OwnershipAcceptanceTests(ManagedFixture):
    def test_uninstall_preserves_personal_data_and_shared_models(self):
        self.check_uninstall(False, False)

    def test_purge_only_removes_recorded_model_files(self):
        self.check_uninstall(True, False)

    def test_keep_models_overrides_purge(self):
        self.check_uninstall(True, True)

    def check_uninstall(self, purge, keep):
        config = self.root / 'config/config.json'
        model = self.root / 'shared-models/owned.bin'
        external = model.with_name('external.bin')
        for path in (config, model, external):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('user data')
        managed.atomic_json(self.install.receipt, {'files': {
            str(config): {'sha256': managed.digest(config), 'kind': 'personal'},
            str(model): {'sha256': managed.digest(model), 'kind': 'model'}}})
        with mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(self.install, 'check_integrations'), mock.patch.object(self.install, 'service_state', return_value={}), mock.patch.object(self.install, 'daemon_running', return_value=False):
            args = ['--yes'] + (['--purge'] if purge else []) + (['--keep-models'] if keep else [])
            managed.uninstall(args)
            managed.uninstall(args)
        self.assertTrue(external.exists())
        self.assertEqual(config.exists(), not purge)
        self.assertEqual(model.exists(), not purge or keep)

    def test_exact_added_block_removed_after_unrelated_edits(self):
        import managed_integrations as integrations
        path = self.root / 'style.css'
        path.write_text('original\n')
        with mock.patch.object(integrations, 'Installation', return_value=self.install):
            with integrations.edit_files([path], [path]):
                path.write_text('@import "hyprwhspr.css";\noriginal\n')
            path.write_text(path.read_text() + 'user addition\n')
            with integrations.edit_files([path], [path], 'remove'):
                pass
        self.assertEqual(path.read_text(), 'original\nuser addition\n')

    def test_modified_shared_json_is_preserved(self):
        import managed_integrations as integrations
        path = self.root / 'bar.json'
        path.write_text('{"other":true}')
        with mock.patch.object(integrations, 'Installation', return_value=self.install):
            with integrations.edit_files([path], [path]):
                path.write_text('{"other":true,"hyprwhspr":true}')
            path.write_text('{"user":true,"hyprwhspr":true}')
            with integrations.edit_files([path], [path], 'remove'):
                pass
        self.assertEqual(path.read_text(), '{"user":true,"hyprwhspr":true}')


class MigrationTests(ManagedFixture):
    def test_clean_clone_and_old_venv_removed_after_commit(self):
        self.check_clone('', '', removed=True)

    def test_dirty_clone_and_venv_preserved(self):
        self.check_clone('?? personal.txt', '', removed=False)

    def test_local_commits_preserve_clone_and_venv(self):
        self.check_clone('', 'local-commit', removed=False)

    def check_clone(self, status, commits, removed):
        clone = self.install.data / 'src'
        venv = self.install.data / 'venv'
        clone.mkdir(parents=True)
        venv.mkdir()
        (venv / 'pyvenv.cfg').write_text('home = /python')
        managed.atomic_json(self.install.state / 'install-state.json', {'installed_backend': 'cpu'})
        with mock.patch.object(managed, 'run', side_effect=[mock.Mock(stdout='https://github.com/goodroot/hyprwhspr.git'), mock.Mock(stdout=status), mock.Mock(stdout=commits)]):
            self.install.clean_legacy()
        self.assertEqual(clone.exists(), not removed)
        self.assertEqual(venv.exists(), not removed)

    def test_service_paths_preserve_custom_xdg_and_spaces(self):
        root = Path(__file__).resolve().parents[1]
        with mock.patch.object(managed, 'Installation', return_value=self.install):
            unit = managed.service_content(root, self.install.data / 'launcher')
        self.assertIn('ExecStart="' + str(self.install.data / 'launcher') + '"', unit)
        self.assertIn('XDG_STATE_HOME=' + str(self.install.state.parent), unit)
        self.assertNotIn('Environment=HYPRWHSPR_ROOT=', unit)

    def test_service_stability_rejects_continuous_restarts(self):
        ticks = iter(range(1000))
        restarts = iter(range(1000))
        with mock.patch.object(managed.time, 'monotonic', side_effect=lambda: next(ticks)), mock.patch.object(managed.time, 'sleep'), mock.patch.object(self.install, 'service_state', side_effect=lambda: {'ActiveState': 'active', 'SubState': 'running', 'MainPID': '1', 'NRestarts': str(next(restarts))}):
            with self.assertRaisesRegex(RuntimeError, 'stable'):
                self.install.stable()

    def test_service_stability_accepts_fifteen_seconds(self):
        ticks = iter(range(1000))
        with mock.patch.object(managed.time, 'monotonic', side_effect=lambda: next(ticks)), mock.patch.object(managed.time, 'sleep'), mock.patch.object(self.install, 'service_state', return_value={'ActiveState': 'active', 'SubState': 'running', 'MainPID': '1', 'NRestarts': '0'}):
            self.install.stable()


if __name__ == '__main__':
    unittest.main()
