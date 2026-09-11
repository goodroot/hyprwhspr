"""Regression coverage for cleanup convergence and integration receipts."""
import io
from pathlib import Path
import subprocess
from unittest import mock

from tests import test_managed_install as fixtures
import managed_install as managed
import managed_integrations as integrations


class EleventhReviewTests(fixtures.ManagedFixture):
    def legacy(self):
        root = self.install.data / 'src'
        root.mkdir(parents=True)
        def git(*args):
            return subprocess.run(['git', '-C', str(root), *args], check=True,
                                  capture_output=True, text=True, timeout=30)
        git('init')
        (root / '.gitignore').write_text('__pycache__/\n*.pyc\nprivate/\n')
        git('add', '.gitignore')
        git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
            '-c', 'commit.gpgsign=false', 'commit', '-m', 'fixture')
        git('remote', 'add', 'origin', f'https://github.com/{managed.REPOSITORY}.git')
        git('update-ref', 'refs/remotes/origin/main', 'HEAD')
        cache = root / 'lib/src/__pycache__'
        cache.mkdir(parents=True)
        (cache / 'main.cpython-311.pyc').write_bytes(b'bytecode')
        venv = self.install.data / 'venv'
        venv.mkdir()
        (venv / 'pyvenv.cfg').touch()
        managed.atomic_json(self.install.state / 'install-state.json', {})
        return root, venv, cache

    def test_real_git_bytecode_checkout_and_runtime_are_reclaimed(self):
        root, venv, _ = self.legacy()
        self.install.clean_legacy()
        self.assertFalse(root.exists())
        self.assertFalse(venv.exists())

    def test_real_git_untracked_and_unknown_ignored_content_is_preserved(self):
        root, venv, cache = self.legacy()
        for relative in ('personal.txt', 'private/notes.txt', 'lib/src/__pycache__/notes.txt'):
            with self.subTest(relative=relative):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('personal')
                self.install.clean_legacy()
                self.assertTrue(root.exists())
                self.assertTrue(venv.exists())
                path.unlink()
        (cache / 'link.pyc').symlink_to(self.root / 'external')
        self.install.clean_legacy()
        self.assertTrue(root.exists())

    def test_deferred_roots_converge_with_running_daemon(self):
        roots = [self.install.data / 'releases' / str(i) for i in range(3)]
        for root in roots:
            root.mkdir(parents=True)
        with mock.patch.object(self.install, 'daemon_running', return_value=True):
            managed.atomic_json(self.install.current, {'root': str(roots[1])})
            with mock.patch.object(managed, '__file__', str(roots[0] / 'lib/src/managed_install.py')):
                self.install.remove([str(roots[0])])
            self.assertTrue(roots[0].exists())
            managed.atomic_json(self.install.current, {'root': str(roots[2])})
            with mock.patch.object(managed, '__file__', str(roots[1] / 'lib/src/managed_install.py')):
                self.install.remove([str(roots[1]), str(roots[2])])
            self.assertFalse(roots[0].exists())
            self.assertTrue(roots[1].exists())
            self.assertTrue(roots[2].exists())
            self.install.remove([])
            self.assertFalse(roots[1].exists())
            self.assertTrue(roots[2].exists())

    def test_migrated_owned_block_removes_without_user_edits(self):
        path = self.root / 'bindings.conf'
        baseline = '# user config\n'
        block = '\n# hyprwhspr\nbindd = SUPER ALT, D, Dictate, exec, /old/tray.sh record\n'
        user = '\n# user addition\nbind = SUPER, X, exec, terminal\n'
        path.write_text(baseline)
        with mock.patch.object(integrations, 'Installation', return_value=self.install):
            with integrations.edit_files([path], [path]):
                path.write_text(baseline + block)
            path.write_text(path.read_text() + user)
            with integrations.edit_files([path], [path]):
                path.write_text(path.read_text().replace('/old/tray.sh', "'/data/launcher' --managed-tray"))
            entry = managed.read_json(self.install.receipt)['files'][str(path)]
            self.assertTrue(integrations.remove_entry(path, entry))
        self.assertEqual(path.read_text(), baseline + user)

    def test_refresh_does_not_claim_user_modified_block(self):
        block = '# owned\ncommand old\n'
        before = block.replace('command', 'user command')
        self.assertEqual(integrations.refresh_insertion(block, before,
                         before.replace('old', 'new')), block)

    def test_non_utf8_integration_preserves_bytes_and_receipt(self):
        path = self.root / 'hyprland.conf'
        path.write_bytes(b'# caf\xe9\n')
        with mock.patch.object(integrations, 'Installation', return_value=self.install), \
                mock.patch('sys.stdout', new_callable=io.StringIO) as output:
            with integrations.edit_files([path], [path]) as proceed:
                self.assertFalse(proceed)
        self.assertIn('file preserved', output.getvalue())
        self.assertEqual(path.read_bytes(), b'# caf\xe9\n')
        self.assertFalse(self.install.receipt.exists())

    def test_group_cleanup_absence_race_and_real_failure(self):
        remove = mock.Mock(side_effect=subprocess.CalledProcessError(1, ['gpasswd']))
        with mock.patch.object(managed, 'group_membership_present', return_value=False):
            managed.remove_added_group('user', 'audio', remove)
        remove.assert_not_called()
        with mock.patch.object(managed, 'group_membership_present', side_effect=[True, False]):
            managed.remove_added_group('user', 'audio', remove)
        with mock.patch.object(managed, 'group_membership_present', return_value=True):
            with self.assertRaises(subprocess.CalledProcessError):
                managed.remove_added_group('user', 'audio', remove)

    def test_missing_group_or_user_is_already_absent(self):
        import pwd
        import grp
        with mock.patch.object(pwd, 'getpwnam', side_effect=KeyError):
            self.assertFalse(managed.group_membership_present('missing', 'audio'))
        with mock.patch.object(pwd, 'getpwnam'), mock.patch.object(grp, 'getgrnam', side_effect=KeyError):
            self.assertFalse(managed.group_membership_present('user', 'missing'))

    def test_daemon_process_paths_override_restored_pointer(self):
        runtime = self.root / 'runtime/hyprwhspr'
        runtime.mkdir(parents=True)
        (runtime / 'hyprwhspr.lock').write_text('12345')
        root = self.install.data / 'releases/staged'
        backend = self.install.data / 'environments/staged'
        managed.atomic_json(self.install.current, {'root': str(self.install.data / 'releases/old')})
        command = f'{backend}/bin/python\0-s\0{root}/lib/main.py\0'.encode()
        with mock.patch.object(self.install, 'daemon_running', return_value=True), \
                mock.patch.object(Path, 'read_bytes', return_value=command) as read:
            self.assertEqual(self.install.daemon_paths(), {root, backend})
        read.assert_called_once()

    def test_managed_absent_membership_retires_receipt_without_sudo(self):
        from tests import test_install_tenth_review as tenth
        entry = {'kind': 'group', 'group': 'audio', 'user': 'test-user', 'added': True}
        managed.atomic_json(self.install.receipt, {'permissions': [entry]})
        with mock.patch.object(managed, 'group_membership_present', return_value=False), \
                mock.patch.object(managed, 'run') as run:
            tenth.TenthReviewTests.call_uninstall(self)
        run.assert_not_called()
        self.assertFalse(managed.read_json(self.install.receipt)['permissions'][0]['added'])

    def test_legacy_absent_membership_retires_receipt_without_sudo(self):
        from tests import test_install_review_regressions as first
        with mock.patch.object(managed, 'group_membership_present', return_value=False):
            *_, sudo, _ = first.LegacyUninstallReviewTests.run_uninstall(self, permissions=True)
        sudo.assert_not_called()
        self.assertFalse(managed.read_json(self.install.receipt)['permissions'][0]['added'])

    def test_real_git_tracked_change_diagnostic(self):
        root, _, _ = self.legacy()
        (root / '.gitignore').write_text('# changed\n__pycache__/\n')
        with mock.patch('sys.stdout', new_callable=io.StringIO) as output:
            self.install.clean_legacy()
        self.assertIn('tracked change ( M): .gitignore', output.getvalue())
        self.assertNotIn('ignored content', output.getvalue())
        self.assertTrue(root.exists())

    def test_real_git_rename_diagnostic_keeps_both_complete_paths(self):
        root, _, _ = self.legacy()
        subprocess.run(['git', '-C', str(root), 'mv', '.gitignore', 'renamed file'],
                       check=True, capture_output=True, timeout=30)
        with mock.patch('sys.stdout', new_callable=io.StringIO) as output:
            self.install.clean_legacy()
        self.assertIn('tracked change (R ): .gitignore -> renamed file', output.getvalue())
        self.assertNotIn('ignored content', output.getvalue())
        self.assertTrue(root.exists())

    def test_hyprland_discovery_returns_files_and_follows_external_sources(self):
        config = self.root / 'config/hypr/hyprland.conf'
        config.parent.mkdir(parents=True)
        external = self.root / 'external.conf'
        external.write_text('# external\n')
        config.write_text(f'source = {external}\nsource = $unresolved/optional.conf\nsource = absent/*.conf\n')
        self.assertEqual(self.install.hyprland_files(), {config, external})
