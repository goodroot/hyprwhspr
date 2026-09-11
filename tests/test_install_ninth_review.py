"""Receipt filtering and safe text updates across lifecycle and desktop setup."""
from contextlib import ExitStack
import os
from pathlib import Path
from unittest import mock

from tests.test_managed_install import ManagedFixture
from tests.test_setup_command_scope import install as cli_install, setup
from tests import test_install_sixth_review as sixth_review
import managed_install as managed
import managed_integrations as integrations


class NinthReviewTests(ManagedFixture):
    def test_model_receipts_are_never_opened_as_bindings(self):
        model = self.root / 'ggml-base.bin'
        model.write_bytes(b'\x80model')
        path = self.root / 'bindings.conf'
        path.write_text(f'bind = SUPER, D, exec, {self.install.data}/src/config/hyprland/hyprwhspr-tray.sh record\n')
        managed.atomic_json(self.install.receipt, {'files': {
            str(model): {'kind': 'model'}, str(path): {'kind': 'integration', 'sha256': managed.digest(path)}}})
        original = Path.read_text
        def read(candidate, *args, **kwargs):
            if candidate == model:
                raise AssertionError('Binary model was opened as text')
            return original(candidate, *args, **kwargs)
        with mock.patch.object(Path, 'read_text', read):
            changes = self.install.binding_migrations()
        self.assertIn('--managed-tray record', changes[path])

    def test_invalid_config_encoding_is_reported_not_raised(self):
        path = self.root / 'config/hypr/bindings.conf'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'# caf\xe9')
        self.assertEqual(self.install.binding_migrations(), {})

    def test_unrelated_optional_and_variable_includes_do_not_retain_checkout(self):
        path = self.root / 'config/hypr/hyprland.conf'
        path.parent.mkdir(parents=True)
        path.write_text('source = $configs/keybinds.conf\nsource = optional/*.conf\n# source = disabled.conf\n')
        self.assertEqual(self.install.legacy_binding_references(), [])
        path.write_text(path.read_text() + 'source = $configs/hyprwhspr.conf\n')
        self.assertEqual(self.install.legacy_binding_references(), [])

    def test_symlinked_waybar_stylesheet_migrates_through_target(self):
        legacy, _, style, _ = sixth_review.SixthReviewTests.seed_bars(self)
        target = self.root / 'dotfiles/waybar.css'
        target.parent.mkdir()
        target.write_text(style.read_text())
        style.unlink()
        style.symlink_to(target)
        with mock.patch.object(self.install, 'bar_processes', return_value={}):
            changes = self.install.bar_migrations()
        self.assertIn(target, changes)
        with mock.patch.object(integrations, 'Installation', return_value=self.install):
            with integrations.edit_files(changes, changes):
                for path, content in changes.items():
                    managed.atomic_text(path, content)
        self.assertTrue(style.is_symlink())
        self.assertNotIn(str(legacy), style.read_text())
        self.assertEqual(self.install.legacy_bar_references(), [])

    def test_latin1_bar_is_preserved_without_cleanup_exception(self):
        legacy, _, style, _ = sixth_review.SixthReviewTests.seed_bars(self)
        style.write_bytes(b'/* caf\xe9 */')
        with mock.patch.object(self.install, 'bar_processes', return_value={}):
            self.assertNotIn(style, self.install.bar_migrations())
        self.assertIn(style, self.install.legacy_bar_references())
        self.install.clean_legacy()
        self.assertTrue(legacy.exists())

    def test_foreign_binding_is_reported_without_duplicate_or_silent_success(self):
        path = self.root / 'config/hypr/bindings.conf'
        path.parent.mkdir(parents=True)
        original = 'bind = SUPER, D, exec, /old-checkout/config/hyprland/hyprwhspr-tray.sh record\n'
        path.write_text(original)
        with mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}'}), mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(cli_install, 'HYPRWHSPR_ROOT', '/current-release'), mock.patch.object(cli_install, 'log_warning') as warning:
            self.assertFalse(cli_install._edit_hyprland_bindings())
        self.assertEqual(path.read_text(), original)
        self.assertTrue(any('targets another installation' in call.args[0] for call in warning.call_args_list))

    def test_binding_rewrite_failure_preserves_original_utf8_bytes(self):
        path = self.root / 'config/hypr/bindings.conf'
        path.parent.mkdir(parents=True)
        original = f'# café — 🎙\nbind = SUPER, D, exec, {self.install.data}/src/config/hyprland/hyprwhspr-tray.sh record\n'
        path.write_text(original, encoding='utf-8')
        with mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}'}), mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(cli_install, 'HYPRWHSPR_ROOT', '/current-release'), mock.patch.object(managed.os, 'replace', side_effect=OSError('disk full')):
            self.assertFalse(cli_install._edit_hyprland_bindings())
        self.assertEqual(path.read_bytes(), original.encode('utf-8'))

    def test_exact_legacy_udev_rule_is_adopted_without_writing_a_duplicate(self):
        rules = self.root / 'rules'
        rules.mkdir()
        old = rules / '99-uinput.rules'
        old.write_text(setup.UINPUT_RULE_CONTENT, encoding='utf-8')
        with mock.patch.object(managed, 'Installation', return_value=self.install), \
                mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}'}):
            self.assertEqual(setup._select_uinput_rule(rules), old)
            self.assertEqual(setup._select_uinput_rule(rules), old)
        permissions = managed.read_json(self.install.receipt)['permissions']
        self.assertEqual(len(permissions), 1)
        self.assertFalse(permissions[0]['added'])
        self.assertTrue(permissions[0]['adopted_legacy'])
        self.assertEqual(permissions[0]['sha256'], managed.digest(old))
        self.assertFalse((rules / '99-hyprwhspr-uinput.rules').exists())

    def test_unrelated_legacy_rule_is_not_adopted(self):
        rules = self.root / 'rules'
        rules.mkdir()
        old = rules / '99-uinput.rules'
        old.write_text('unrelated rule')
        with mock.patch.object(managed, 'record_permission') as record:
            self.assertEqual(setup._select_uinput_rule(rules), rules / '99-hyprwhspr-uinput.rules')
        record.assert_not_called()
        self.assertEqual(old.read_text(), 'unrelated rule')

    def test_recognized_preexisting_rule_is_preserved_even_with_permission_selection(self):
        rules = self.root / 'rules'
        rules.mkdir()
        old = rules / '99-uinput.rules'
        old.write_text(setup.UINPUT_RULE_CONTENT)
        def run(command, **kwargs):
            self.assertIn(command, (['sudo', 'rm', str(old)], ['sudo', 'udevadm', 'control', '--reload-rules']))
            if command[1] == 'rm':
                old.unlink()
        with mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(self.install, 'check_integrations'), mock.patch.object(self.install, 'service_state', return_value={}), mock.patch.object(self.install, 'daemon_running', return_value=False), mock.patch.object(managed, 'run', side_effect=run) as command:
            setup._select_uinput_rule(rules)
            managed.uninstall(['--yes'])
            command.assert_not_called()
            self.assertTrue(old.exists())
            managed.uninstall(['--yes', '--remove-permissions'])
        self.assertTrue(old.exists())
        command.assert_not_called()
