"""Partial integration failures and recovery after host changes."""
import io
import json
import os
from pathlib import Path
import subprocess
from unittest import mock

from tests import test_managed_install as fixtures
from tests import test_install_review_regressions as first
from tests.test_setup_command_scope import setup
from cli import waybar, noctalia
import managed_install as managed
import managed_integrations as integrations


class ThirteenthReviewTests(fixtures.ManagedFixture):
    def test_atomic_shared_removal_failure_preserves_content_and_mode(self):
        path = self.root / 'bindings.conf'
        for use_blocks in (False, True):
            with self.subTest(blocks=use_blocks):
                path.write_text('user\nowned\n')
                path.chmod(0o640)
                entry = {'original': 'user\n', 'sha256': managed.digest(path),
                         'restore_sha256': managed.digest(path)}
                if use_blocks:
                    entry['insertions'] = ['owned\n']
                with mock.patch.object(managed.os, 'replace', side_effect=OSError('ENOSPC')):
                    with self.assertRaises(OSError):
                        integrations.remove_entry(path, entry)
                self.assertEqual(path.read_text(), 'user\nowned\n')
                self.assertEqual(path.stat().st_mode & 0o777, 0o640)
                self.assertTrue(integrations.remove_entry(path, entry))
                self.assertEqual(path.read_text(), 'user\n')
                self.assertEqual(path.stat().st_mode & 0o777, 0o640)

    def test_post_edit_bad_text_commits_other_files_and_preserves_original_error(self):
        good, bad = self.root / 'good.conf', self.root / 'bad.conf'
        good.write_text('user\n')
        bad.write_text('original\n')
        with mock.patch.object(integrations, 'Installation', return_value=self.install):
            with self.assertRaisesRegex(RuntimeError, 'editor failed'):
                with integrations.edit_files([bad, good], [bad, good]):
                    good.write_text('user\nowned\n')
                    bad.write_bytes(b'\x80')
                    raise RuntimeError('editor failed')
        entries = managed.read_json(self.install.receipt)['files']
        self.assertIn(str(good), entries)
        self.assertTrue(integrations.remove_entry(good, entries[str(good)]))
        self.assertEqual(good.read_text(), 'user\n')

    def test_remove_unreadable_reference_commits_other_removals(self):
        config = self.root / 'config.jsonc'
        config.write_bytes(b'\x80')
        module = self.root / 'hyprwhspr-module.jsonc'
        module.write_text('owned')
        independent = self.root / 'independent'
        independent.write_text('owned')
        managed.atomic_json(self.install.receipt, {'files': {
            str(p): {'sha256': managed.digest(p)} for p in (module, independent)}})
        removal = {}
        with mock.patch.object(integrations, 'Installation', return_value=self.install):
            with integrations.edit_files([module, independent], mode='remove', removal=removal):
                pass
        self.assertFalse(removal['complete'])
        self.assertTrue(module.exists())
        self.assertFalse(independent.exists())
        self.assertNotIn(str(independent), managed.read_json(self.install.receipt)['files'])

    def test_recorded_interpreter_fallback_and_explicit_choice(self):
        identity = {'path': '/usr/bin/python3.14', 'version': [3, 14, 0], 'identity': 'new'}
        with mock.patch.object(managed, 'run', side_effect=[FileNotFoundError(), mock.Mock(stdout=json.dumps(identity))]) as run:
            self.assertEqual(managed.interpreter('/removed/python3.12', fallback=True), identity)
        self.assertEqual([c.args[0][0] for c in run.call_args_list], ['/removed/python3.12', '/usr/bin/python3'])
        with mock.patch.object(managed, 'run', side_effect=FileNotFoundError()) as run:
            with self.assertRaises(RuntimeError):
                managed.interpreter('/explicit/missing')
        self.assertEqual(run.call_count, 1)

    def test_relative_owned_command_symlink_is_accepted(self):
        command = self.install.command_path()
        command.parent.mkdir(parents=True)
        target = self.install.data / 'src/bin/hyprwhspr'
        command.symlink_to(os.path.relpath(target, command.parent))
        with mock.patch.object(self.install, 'service_state', return_value={}):
            self.install.check_integrations()
            command.unlink()
            command.symlink_to('../../../foreign')
            with self.assertRaisesRegex(RuntimeError, 'Conflicting command'):
                self.install.check_integrations()

    def permission_setup(self, missing_user=False, rule_exists=True, uinput_writable=True,
                         member=False, active=False, need_input_group=True, sudo_code=0, issues=None):
        import pwd
        import grp
        with mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}', 'SUDO_USER': 'target'}), \
                mock.patch.object(pwd, 'getpwnam', side_effect=KeyError('target') if missing_user else None,
                                  return_value=mock.Mock(pw_gid=777)), \
                mock.patch.object(os, 'getgrouplist', return_value=[777, 888] + ([555] if member else [])) as groups, \
                mock.patch.object(os, 'getgroups', return_value=[555] if active else []), \
                mock.patch.object(os, 'getegid', return_value=777), \
                mock.patch.object(grp, 'getgrnam', side_effect=lambda name: mock.Mock(gr_gid={'input': 555}[name])), \
                mock.patch.object(grp, 'getgrgid', side_effect=AssertionError('Do not resolve unrelated GIDs')), \
                mock.patch.object(setup, 'run_sudo_command', return_value=mock.Mock(returncode=sudo_code)) as sudo, \
                mock.patch.object(setup, '_select_uinput_rule', return_value=mock.Mock(exists=lambda: rule_exists)), \
                mock.patch.object(setup, 'UACCESS_RULE_PATH', mock.Mock(exists=lambda: rule_exists)), \
                mock.patch.object(setup, '_write_udev_rule', return_value=True) as write_rule, \
                mock.patch.object(Path, 'exists', return_value=True), \
                mock.patch.object(os, 'access', return_value=uinput_writable), \
                mock.patch.object(managed, 'record_permission') as record:
            self.relogin = setup.setup_permissions(need_input_group=need_input_group, issues=issues)
        self.write_rule = write_rule
        return groups, sudo, record

    USERMOD = ['usermod', '-a', '-G', 'input', 'target']

    def test_permission_failure_reports_action_without_claiming_logout_will_fix_it(self):
        issues = []
        self.permission_setup(sudo_code=1, issues=issues)
        self.assertFalse(self.relogin)
        self.assertEqual(issues, ['Resolve the permission errors above, then rerun: hyprwhspr setup'])

    def test_permission_snapshot_uses_target_primary_gid_and_adds_only_input(self):
        groups, sudo, record = self.permission_setup()
        groups.assert_called_once_with('target', 777)
        self.assertIn(mock.call(self.USERMOD, check=False), sudo.call_args_list)
        self.assertEqual([c.args[0] for c in record.call_args_list],
                         [{'kind': 'group', 'user': 'target', 'group': 'input', 'added': True}])
        self.assertTrue(self.relogin)

    def test_unknown_user_snapshot_does_not_skip_usermod_or_claim_ownership(self):
        _, sudo, record = self.permission_setup(missing_user=True)
        self.assertIn(mock.call(self.USERMOD, check=False), sudo.call_args_list)
        record.assert_not_called()

    def test_existing_member_is_not_re_added_and_needs_no_login(self):
        _, sudo, record = self.permission_setup(member=True, active=True)
        self.assertEqual(sudo.call_args_list, [])
        record.assert_not_called()
        self.assertFalse(self.relogin)

    def test_member_added_by_an_earlier_run_still_needs_a_login(self):
        _, sudo, _ = self.permission_setup(member=True, active=False)
        self.assertEqual(sudo.call_args_list, [])
        self.assertTrue(self.relogin)

    def test_compositor_shortcut_changes_no_groups_and_needs_no_login(self):
        groups, sudo, record = self.permission_setup(need_input_group=False)
        groups.assert_not_called()
        self.assertEqual(sudo.call_args_list, [])
        record.assert_not_called()
        self.assertFalse(self.relogin)

    UDEV_RELOAD = [['udevadm', 'control', '--reload-rules'], ['udevadm', 'trigger', '--name-match=uinput']]

    def test_existing_uinput_rule_with_usable_device_is_not_reloaded_or_retriggered(self):
        _, sudo, _ = self.permission_setup()
        commands = [call.args[0] for call in sudo.call_args_list]
        self.assertEqual(commands, [self.USERMOD])
        self.write_rule.assert_not_called()

    def test_new_uinput_rules_are_written_reloaded_and_retriggered(self):
        _, sudo, _ = self.permission_setup(rule_exists=False)
        commands = [call.args[0] for call in sudo.call_args_list]
        self.assertEqual(commands[-2:], self.UDEV_RELOAD)
        self.assertEqual([c.args[1] for c in self.write_rule.call_args_list],
                         [setup.UACCESS_RULE_CONTENT, setup.UINPUT_RULE_CONTENT])

    def test_uaccess_rule_sorts_before_the_seat_rules_that_apply_it(self):
        # systemd runs the uaccess builtin from 73-seat-late.rules; a later file never gets the ACL.
        self.assertLess(setup.UACCESS_RULE_PATH.name, '73-seat-late.rules')
        self.assertIn('TAG+="uaccess"', setup.UACCESS_RULE_CONTENT)

    def test_written_rule_is_recorded_as_added(self):
        rule = self.root / '70-test.rules'
        def tee(command, input_data=None, check=False):
            rule.write_bytes(input_data)
            return mock.Mock(returncode=0)
        with mock.patch.object(setup, 'run_sudo_command', side_effect=tee), \
                mock.patch.object(managed, 'record_permission') as record:
            self.assertTrue(setup._write_udev_rule(rule, setup.UACCESS_RULE_CONTENT))
        self.assertEqual(record.call_args.args[0],
                         {'kind': 'rule', 'path': str(rule), 'added': True, 'sha256': managed.digest(rule)})

    def test_existing_rule_is_retriggered_when_device_is_still_unwritable(self):
        # The documented repair path: the rule was left by a failed run or written
        # by hand, so /dev/uinput still carries its pre-rule ownership.
        _, sudo, _ = self.permission_setup(uinput_writable=False)
        commands = [call.args[0] for call in sudo.call_args_list]
        self.assertEqual(commands[-2:], self.UDEV_RELOAD)

    def test_unreadable_recorded_model_is_preserved(self):
        real_digest = managed.digest
        seen = set()
        def digest(path):
            if Path(path).suffix == '.bin':
                if path in seen:
                    raise PermissionError('unreadable model')
                seen.add(path)
            return real_digest(path)
        with mock.patch.object(managed, 'digest', side_effect=digest):
            _, _, _, model, _, _, _, output = first.LegacyUninstallReviewTests.run_uninstall(self, purge=True)
        self.assertTrue(model.exists())
        self.assertIn('Preserving unreadable model', output)

    def test_waybar_removal_reports_preserved_entries(self):
        base = self.root / 'config/waybar'
        base.mkdir(parents=True)
        module = base / 'hyprwhspr-module.jsonc'
        module.write_text('unowned')
        with mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}'}), \
                mock.patch.object(integrations, 'Installation', return_value=self.install):
            self.assertFalse(waybar.setup_waybar('remove'))
            with self.assertRaisesRegex(RuntimeError, 'removal incomplete'):
                waybar.waybar_command('remove')
            managed.atomic_json(self.install.receipt, {'files': {str(module): {'sha256': managed.digest(module)}}})
            self.assertTrue(waybar.setup_waybar('remove'))
        self.assertFalse(module.exists())

    def test_noctalia_legacy_path_does_not_import_managed_helpers(self):
        import builtins
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name in ('managed_install', 'managed_integrations'):
                raise AssertionError('eager managed import')
            return original_import(name, *args, **kwargs)
        with mock.patch.object(builtins, '__import__', side_effect=guarded):
            import importlib
            importlib.reload(noctalia)
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(noctalia, '_setup_noctalia', return_value=True):
                self.assertTrue(noctalia.setup_noctalia('remove'))
