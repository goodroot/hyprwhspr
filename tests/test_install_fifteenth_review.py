"""Optional recovery failures and pre-existing permission provenance."""
from contextlib import ExitStack
import io
import os
from pathlib import Path
import subprocess
from unittest import mock

from tests import test_managed_install as fixtures
from tests import test_install_tenth_review as tenth
from tests.test_install_second_review import noctalia
from tests.test_setup_command_scope import setup, uninstall
import managed_install as managed
import managed_integrations as integrations


class FifteenthReviewTests(fixtures.ManagedFixture):
    def queue(self, job=None):
        target = self.root / 'config.conf'
        target.write_text('new')
        runtime = self.install.data / 'environments/held'
        runtime.mkdir(parents=True)
        queue = self.install.state / 'deferred-restoration.json'
        managed.atomic_json(queue, {'files': {str(target): job or {'snapshot': {'content': 'old', 'mode': 0o644}, 'expected': {'sha256': managed.digest(target)}}},
                                   'protected': [str(runtime)]})
        managed.atomic_json(self.install.state / 'deferred-cleanup.json', [str(runtime)])
        return queue, target, runtime

    def test_unreadable_queue_does_not_block_repeated_recovery_or_delete_runtime(self):
        queue, _, runtime = self.queue()
        original = managed.read_json
        def read(path, *args):
            if Path(path) == queue:
                raise PermissionError('queue unreadable')
            return original(path, *args)
        with mock.patch.object(managed, 'read_json', read):
            for _ in range(3):
                self.install.recover()
                self.assertTrue(runtime.exists())
        self.install.recover()
        self.assertTrue(runtime.exists())
        self.assertTrue(list(self.install.state.glob('deferred-restoration.json.unreadable-*')))
        self.assertIn(str(runtime), managed.read_json(self.install.state / 'unverified-generations.json'))

    def test_malformed_queue_and_job_are_deferred(self):
        queue, target, runtime = self.queue()
        for content in ('[]', '{broken', '{"files": {"' + str(target) + '": null}, "protected": ["' + str(runtime) + '"]}'):
            queue.write_text(content)
            for _ in range(2):
                self.install.recover()
                self.assertTrue(runtime.exists())
        self.assertEqual(target.read_text(), 'new')

    def test_queue_unlink_and_write_failures_are_deferred(self):
        queue, target, runtime = self.queue()
        actual = Path.unlink
        def unlink(path, *args, **kwargs):
            if path == queue:
                raise PermissionError('queue unlink')
            return actual(path, *args, **kwargs)
        with mock.patch.object(Path, 'unlink', unlink):
            self.install.recover()
        self.assertTrue(queue.exists())
        self.assertTrue(runtime.exists())
        self.assertEqual(target.read_text(), 'old')
        target.write_text('new')  # Make the next restore require a write again.
        original = managed.atomic_json
        def write(path, content):
            if path == queue:
                raise OSError('queue ENOSPC')
            return original(path, content)
        with mock.patch.object(managed, 'atomic_json', write), mock.patch.object(managed, 'atomic_text', side_effect=PermissionError('config')):
            self.install.recover()
        self.assertTrue(queue.exists())
        self.install.recover()
        self.assertFalse(runtime.exists())

    def test_future_pre_restore_exception_cannot_skip_pointer_restore(self):
        old = {'root': 'old'}
        managed.atomic_json(self.install.current, {'root': 'new'})
        managed.atomic_json(self.install.journal, {'phase': 'activated', 'old': old, 'created': []})
        with mock.patch.object(self.install, 'recovery_step', side_effect=TypeError('future mistake')):
            with self.assertRaisesRegex(TypeError, 'future mistake'):
                self.install.recover()
        self.assertEqual(managed.read_json(self.install.current), old)
        self.assertTrue(self.install.journal.exists())

    def test_unreadable_daemon_lock_is_running_and_cleanup_is_conservative(self):
        runtime = self.root / 'runtime/hyprwhspr'
        runtime.mkdir(parents=True)
        lock = runtime / 'hyprwhspr.lock'
        lock.touch()
        env = self.install.data / 'environments/active'
        env.mkdir(parents=True)
        managed.atomic_json(self.install.current, {'backend': {'path': str(env)}})
        actual = Path.open
        def open_file(path, *args, **kwargs):
            if path == lock:
                raise PermissionError('runtime lock')
            return actual(path, *args, **kwargs)
        with mock.patch.object(Path, 'open', open_file):
            self.assertTrue(self.install.daemon_running())
            self.install.remove([str(env)])
        self.assertTrue(env.exists())

    def test_recognized_rule_has_no_authorship_and_old_receipt_is_preserved(self):
        rules = self.root / 'rules'
        rules.mkdir()
        rule = rules / '99-uinput.rules'
        rule.write_text(setup.UINPUT_RULE_CONTENT)
        with mock.patch.object(managed, 'Installation', return_value=self.install), \
                mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}'}):
            setup._select_uinput_rule(rules)
        self.assertFalse(managed.read_json(self.install.receipt)['permissions'][0]['added'])
        with self.install.receipts() as receipt:
            receipt['permissions'][0]['added'] = True  # previous installer bug
        with mock.patch.object(managed, 'run') as run:
            tenth.TenthReviewTests.call_uninstall(self)
        run.assert_not_called()
        self.assertTrue(rule.exists())

    def legacy_uninstall(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        for key, value in {'USER_HOME': self.root, 'USER_SYSTEMD_DIR': self.root / 'units',
                'USER_CONFIG_DIR': self.root / 'config', 'USER_BASE': self.install.data,
                'VENV_DIR': self.root / 'venv', 'STATE_DIR': self.install.state,
                'PYWHISPERCPP_SRC_DIR': self.root / 'source', 'PYWHISPERCPP_MODELS_DIR': self.root / 'models',
                'CREDENTIALS_FILE': self.root / 'credentials'}.items():
            stack.enter_context(mock.patch.object(uninstall, key, value))
        stack.enter_context(mock.patch.object(managed, 'Installation', return_value=self.install))
        stack.enter_context(mock.patch.object(uninstall, 'run_command', return_value=mock.Mock(returncode=0)))
        return lambda: uninstall.uninstall_command(yes=True, remove_permissions=True)

    def test_legacy_rule_reload_retry_after_successful_unlink(self):
        rule = self.root / 'owned-rule'
        rule.write_text('owned')
        managed.atomic_json(self.install.receipt, {'permissions': [{'kind': 'rule', 'path': str(rule),
                            'sha256': managed.digest(rule), 'added': True}]})
        call = self.legacy_uninstall()
        def sudo(command, **kwargs):
            if command[0] == 'rm':
                rule.unlink()
            else:
                raise subprocess.CalledProcessError(1, command)
        with mock.patch.object(uninstall, 'run_sudo_command', side_effect=sudo):
            with self.assertRaises(RuntimeError):
                call()
        self.assertTrue(managed.read_json(self.install.receipt)['permissions'][0]['added'])
        with mock.patch.object(uninstall, 'run_sudo_command') as sudo:
            call()
        sudo.assert_called_once_with(['udevadm', 'control', '--reload-rules'], check=True)
        self.assertFalse(managed.read_json(self.install.receipt)['permissions'][0]['added'])

    def test_legacy_adopted_rule_receipt_never_authorizes_removal(self):
        rule = self.root / 'preexisting-rule'
        rule.write_text('preexisting')
        managed.atomic_json(self.install.receipt, {'permissions': [{'kind': 'rule', 'path': str(rule),
            'sha256': managed.digest(rule), 'added': True, 'adopted_legacy': True}]})
        call = self.legacy_uninstall()
        with mock.patch.object(uninstall, 'run_sudo_command') as sudo:
            call()
        sudo.assert_not_called()
        self.assertTrue(rule.exists())

    def test_widget_drift_is_detected_before_copying_plugin(self):
        source = self.root / 'payload/config/noctalia/plugin'
        source.mkdir(parents=True)
        (source / 'widget.luau').write_text('drifted template')
        plugin = self.root / 'plugin'
        dst = {'plugin_dir': plugin, 'settings': self.root / 'settings',
               'template_input': self.root / 'input', 'template_output': self.root / 'output'}
        with mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}'}), \
                mock.patch.object(noctalia, '_noctalia_paths', return_value=dst), \
                mock.patch.object(noctalia, 'HYPRWHSPR_ROOT', str(self.root / 'payload')), \
                mock.patch.object(noctalia, '_setup_noctalia') as editor:
            self.assertFalse(noctalia.setup_noctalia())
        editor.assert_not_called()
        self.assertFalse(plugin.exists())
