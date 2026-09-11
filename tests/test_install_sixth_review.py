"""End-to-end lifecycle regressions for stale services and legacy bar migration."""
import json
import os
from pathlib import Path
import subprocess
from unittest import mock

from tests.test_managed_install import ManagedFixture
from tests.test_install_second_review import ROOT, noctalia
import cli.waybar as waybar
import managed_install as managed
import managed_integrations as integrations


class SixthReviewTests(ManagedFixture):
    def test_uninstall_reloads_removed_unit_and_missing_fragment_allows_reinstall(self):
        unit = self.root / 'units/hyprwhspr.service'
        unit.parent.mkdir()
        unit.write_text('owned service')
        managed.atomic_json(self.install.receipt, {'files': {str(unit): {'sha256': managed.digest(unit)}}})
        with mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(self.install, 'service_state', return_value={'FragmentPath': str(unit)}), mock.patch.object(self.install, 'daemon_running', return_value=False), mock.patch.object(self.install, 'service') as service, mock.patch.object(managed, 'run') as run:
            self.assertEqual(managed.uninstall(['--yes']), 0)
            service.assert_called_once_with('disable')
            self.assertEqual(run.call_args.args[0], ['systemctl', '--user', 'daemon-reload'])
            self.assertIsNone(self.install.check_integrations())
        self.assertFalse(unit.exists())

    def test_reload_failure_is_retried_after_unit_receipt_is_forgotten(self):
        unit = self.root / 'hyprwhspr.service'
        unit.write_text('owned')
        managed.atomic_json(self.install.receipt, {'files': {str(unit): {'sha256': managed.digest(unit)}}})
        with mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(self.install, 'service_state', return_value={}), mock.patch.object(self.install, 'daemon_running', return_value=False), mock.patch.object(self.install, 'service'), mock.patch.object(managed, 'run', side_effect=[subprocess.CalledProcessError(1, ['reload']), None]) as run:
            with self.assertRaisesRegex(RuntimeError, 'daemon-reload failed'):
                managed.uninstall(['--yes'])
            self.assertFalse(unit.exists())
            self.assertEqual(managed.uninstall(['--yes']), 0)
            self.assertEqual(run.call_count, 2)

    def test_recorded_missing_fragment_is_not_hashed(self):
        path = self.root / 'missing.service'
        managed.atomic_json(self.install.receipt, {'files': {str(path): {'sha256': 'old'}}})
        with mock.patch.object(self.install, 'service_state', return_value={'FragmentPath': str(path)}), mock.patch.object(managed, 'digest', side_effect=AssertionError('missing file was hashed')):
            self.assertIsNone(self.install.check_integrations())

    def seed_bars(self):
        legacy = self.install.data / 'src'
        css = legacy / 'config/waybar/hyprwhspr-style.css'
        css.parent.mkdir(parents=True)
        css.write_text('/* legacy css */')
        template = legacy / 'config/noctalia/plugin/widget.luau'
        template.parent.mkdir(parents=True)
        template.write_text((ROOT / 'config/noctalia/plugin/widget.luau').read_text())
        module, style, widget, _ = self.install.legacy_bar_paths()
        for path in (module, style, widget):
            path.parent.mkdir(parents=True, exist_ok=True)
        module.write_text(json.dumps({'exec': str(legacy / 'config/hyprland/hyprwhspr-tray.sh') + ' status'}))
        style.write_text('@import "' + str(css) + '";\n/* user style */')
        widget.write_text(template.read_text())
        return legacy, module, style, widget

    def test_legacy_bars_migrate_before_checkout_cleanup(self):
        legacy, module, style, widget = self.seed_bars()
        with mock.patch.object(self.install, 'bar_processes', return_value={}):
            changes = self.install.bar_migrations()
        with mock.patch.object(integrations, 'Installation', return_value=self.install):
            with integrations.edit_files(changes, changes) as proceed:
                self.assertTrue(proceed)
                for path, content in changes.items():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    managed.atomic_text(path, content, 0o755 if path.suffix == '.sh' else 0o644)
        self.assertEqual(self.install.legacy_bar_references(), [])
        self.assertNotIn(str(legacy), module.read_text())
        self.assertIn('/* user style */', style.read_text())
        self.assertIn(integrations.MANAGED_WIDGET_MARKER, widget.read_text())
        self.assertIn('local TRAY_REL = "/config/hyprland/hyprwhspr-tray.sh"', widget.read_text())
        self.assertIn('table.insert(candidates, "/usr/lib/hyprwhspr")', widget.read_text())
        with mock.patch.object(managed, 'run', side_effect=[mock.Mock(stdout='https://github.com/goodroot/hyprwhspr.git'), mock.Mock(stdout=''), mock.Mock(stdout='')]):
            self.install.clean_legacy()
        self.assertFalse(legacy.exists())
        self.assertTrue((style.parent / 'hyprwhspr-style.css').exists())
        self.assertTrue((self.install.data / 'integrations/legacy-tray.sh').exists())

    def test_customized_widget_preserves_legacy_checkout(self):
        legacy, _, _, widget = self.seed_bars()
        widget.write_text(widget.read_text().replace('local candidates = {}', 'local custom_candidates = {}') + '\n-- custom')
        with mock.patch.object(self.install, 'bar_processes', return_value={}):
            self.assertNotIn(widget, self.install.bar_migrations())
        with mock.patch.object(managed, 'run') as run:
            self.install.clean_legacy()
        run.assert_not_called()
        self.assertTrue(legacy.exists())

    def test_live_bar_keeps_old_checkout_until_restart(self):
        legacy, *_ = self.seed_bars()
        managed.atomic_json(self.install.state / 'legacy-bar-processes.json', {'123': 'start'})
        with mock.patch.object(self.install, 'bar_processes', return_value={'123': 'start'}), mock.patch.object(managed, 'run') as run:
            self.install.clean_legacy()
        run.assert_not_called()
        self.assertTrue(legacy.exists())

    def test_refused_bar_installs_return_false(self):
        module = self.root / 'config/waybar/hyprwhspr-module.jsonc'
        module.parent.mkdir(parents=True)
        module.write_text('unowned')
        with mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}'}), mock.patch.object(integrations, 'Installation', return_value=self.install), mock.patch.object(waybar, '_setup_waybar') as editor:
            self.assertFalse(waybar.setup_waybar())
            editor.assert_not_called()
        plugin = self.root / 'plugin'
        plugin.mkdir()
        (plugin / 'widget.luau').write_text('unowned')
        paths = {'plugin_dir': plugin, 'settings': self.root / 'settings', 'template_input': self.root / 'input', 'template_output': self.root / 'output'}
        with mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}'}), mock.patch.object(integrations, 'Installation', return_value=self.install), mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(noctalia, '_noctalia_paths', return_value=paths), mock.patch.object(noctalia, 'HYPRWHSPR_ROOT', str(ROOT)), mock.patch.object(noctalia, '_setup_noctalia') as editor:
            self.assertFalse(noctalia.setup_noctalia())
            editor.assert_not_called()

    def test_old_deferred_failures_do_not_fail_later_operation_completion(self):
        old = self.install.data / 'environments/old'
        old.mkdir(parents=True)
        with mock.patch.object(managed.shutil, 'rmtree', side_effect=PermissionError('owned by root')):
            with self.assertRaises(managed.CleanupError):
                self.install.remove([str(old)])
            self.install.recover(strict=True)
        self.assertIn(str(old), self.install.status()['deferred_cleanup'])

    def test_atomic_rollback_failure_keeps_existing_config(self):
        path = self.root / 'hyprland.conf'
        path.write_text('current')
        managed.atomic_json(self.install.journal, {'phase': 'activated', 'old': {}, 'created': [],
            'integrations': {str(path): {'content': 'previous', 'mode': 0o600}}})
        with mock.patch.object(self.install, 'service_state', return_value={}), mock.patch.object(managed.os, 'replace', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.install.recover()
        self.assertEqual(path.read_text(), 'current')
        self.assertTrue(self.install.journal.exists())
        with mock.patch.object(self.install, 'service_state', return_value={}):
            self.install.recover()
        self.assertEqual(path.read_text(), 'previous')
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_host_pip_and_runtime_clipboard_are_declared(self):
        script = (ROOT / 'scripts/install-deps.sh').read_text()
        for name in ('install_deps_dnf', 'install_deps_zypper'):
            body = script.split(name + '() {', 1)[1].split('\n}', 1)[0]
            self.assertIn('python3-pip', body)
        self.assertIn('pyperclip>=', (ROOT / 'requirements.txt').read_text())
        self.assertIn('import dbus', script)

    def test_direct_bar_install_commands_report_refusal(self):
        for module, name in ((waybar, 'waybar'), (noctalia, 'noctalia')):
            with mock.patch.object(module, 'setup_' + name, return_value=False):
                with self.assertRaisesRegex(RuntimeError, 'installation incomplete'):
                    getattr(module, name + '_command')('install')
