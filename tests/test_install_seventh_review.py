"""Isolated Python and incomplete uninstall regressions."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
from unittest import mock

from tests.test_managed_install import ManagedFixture
from tests import test_install_sixth_review as sixth_review
from tests.test_install_second_review import ROOT, uninstall as legacy_uninstall
import managed_install as managed
import managed_integrations as integrations


class SeventhReviewTests(ManagedFixture):
    def test_actual_isolated_launcher_never_writes_payload_bytecode(self):
        root = self.root / 'payload'
        names = ['bin/hyprwhspr', 'lib/cli.py', 'lib/main.py', 'requirements-cli.txt', 'requirements.txt', 'share/config.schema.json', 'lib/src/dependency_manifest.py', 'lib/src/managed_install.py']
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('')
        (root / 'lib/src/managed_install.py').write_text('import sys\nassert sys.dont_write_bytecode\nimport dependency_manifest\n')
        managed.atomic_json(root / 'release.json', {'format': 1, 'version': 'v1.0.0', 'files': {name: managed.digest(root / name) for name in names}})
        base = self.root / 'xdg-data/hyprwhspr'
        base.mkdir(parents=True)
        managed.atomic_json(base / 'current.json', {'root': str(root)})
        for _ in range(2):
            result = subprocess.run(['bash', str(ROOT / 'scripts/managed-launcher.sh')], capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='0'))
            self.assertEqual(result.returncode, 0, result.stderr)
            managed.verify_payload(root)
        self.assertEqual(list(root.rglob('*.pyc')), [])
        bootstrap = (ROOT / 'scripts/install.sh').read_text()
        self.assertIn("[python,'-I','-B',str(helper)", bootstrap)
        self.assertIn('"$python_path" -I -B -', bootstrap)

    def test_old_id_widget_and_missing_legacy_css_can_migrate(self):
        legacy, _, style, widget = sixth_review.SixthReviewTests.seed_bars(self)
        old_widget = self.install.legacy_bar_paths()[3]
        old_widget.parent.mkdir(parents=True)
        old_widget.write_text(widget.read_text().replace('noctwhspr', 'hyprwhspr') + '\n-- older plugin metadata')
        (legacy / 'config/waybar/hyprwhspr-style.css').unlink()
        with mock.patch.object(self.install, 'bar_processes', return_value={}):
            changes = self.install.bar_migrations(ROOT)
        self.assertIn(integrations.MANAGED_WIDGET_MARKER, changes[old_widget])
        self.assertIn(str(self.install.data / 'integrations/legacy-tray.sh'), changes[old_widget])
        # The packaged root fallbacks must survive the patch.
        self.assertIn('local TRAY_REL = "/config/hyprland/hyprwhspr-tray.sh"', changes[old_widget])
        self.assertIn('table.insert(candidates, "/usr/lib/hyprwhspr")', changes[old_widget])
        self.assertIn('-- older plugin metadata', changes[old_widget])
        self.assertIn(style, changes)
        self.assertIn(style.parent / 'hyprwhspr-style.css', changes)

    def managed_files(self):
        root = self.install.data / 'releases/current'
        cli = self.install.data / 'environments/cli'
        root.mkdir(parents=True)
        cli.mkdir(parents=True)
        managed.atomic_json(self.install.current, {'root': str(root), 'cli': {'path': str(cli)}})
        launcher = self.install.data / 'launcher'
        launcher.write_text('launcher')
        interpreter = self.install.data / 'interpreter'
        interpreter.write_text('/python')
        command = self.install.command_path()
        command.parent.mkdir(parents=True)
        command.symlink_to(launcher)
        extra = self.root / 'integration'
        extra.write_text('owned')
        files = {str(command): {'target': str(launcher)}, **{str(path): {'sha256': managed.digest(path)} for path in (launcher, interpreter, extra)}}
        managed.atomic_json(self.install.receipt, {'files': files})
        return root, cli, launcher, interpreter, command, extra

    def test_failed_integration_cleanup_leaves_launcher_for_retry(self):
        root, cli, launcher, interpreter, command, extra = self.managed_files()
        original = Path.unlink
        def unlink(path, *args, **kwargs):
            if path == extra:
                raise PermissionError('locked integration')
            return original(path, *args, **kwargs)
        with mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(self.install, 'check_integrations'), mock.patch.object(self.install, 'service_state', return_value={}), mock.patch.object(self.install, 'daemon_running', return_value=False):
            with mock.patch.object(Path, 'unlink', unlink), self.assertRaisesRegex(RuntimeError, 'Uninstall incomplete'):
                managed.uninstall(['--yes'])
            for path in (root, cli, launcher, interpreter, command, self.install.current):
                self.assertTrue(path.exists(), path)
            self.assertEqual(managed.uninstall(['--yes']), 0)
        self.assertFalse(command.is_symlink())

    def test_custom_unit_and_foreign_wrapper_do_not_abort_other_cleanup(self):
        root, cli, launcher, interpreter, command, extra = self.managed_files()
        command.unlink()
        command.write_text('user wrapper')
        unit = self.root / 'hyprwhspr.service'
        unit.write_text('custom unit')
        with mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(self.install, 'service_state', return_value={'FragmentPath': str(unit)}), mock.patch.object(self.install, 'daemon_running', return_value=False):
            self.assertEqual(managed.uninstall(['--yes']), 0)
        self.assertFalse(extra.exists())
        self.assertTrue(unit.exists())
        self.assertTrue(root.exists())
        self.assertTrue(launcher.exists())
        self.assertEqual(command.read_text(), 'user wrapper')

    def test_legacy_dangling_and_unreadable_units_allow_independent_cleanup(self):
        for unreadable in (False, True):
            case = self.root / str(unreadable)
            units = case / 'units'
            units.mkdir(parents=True)
            unit = units / 'hyprwhspr.service'
            if unreadable:
                unit.write_text('unit')
            else:
                unit.symlink_to(case / 'missing-unit')
            venv = case / 'venv'
            venv.mkdir()
            config = case / 'config'
            config.mkdir()
            (config / 'config.json').write_text('{}')
            with ExitStack() as stack:
                for key, value in {'USER_HOME': case, 'USER_SYSTEMD_DIR': units, 'USER_CONFIG_DIR': config, 'VENV_DIR': venv, 'USER_BASE': case / 'data', 'STATE_DIR': case / 'state', 'PYWHISPERCPP_SRC_DIR': case / 'sources', 'PYWHISPERCPP_MODELS_DIR': case / 'models', 'CREDENTIALS_FILE': case / 'credentials'}.items():
                    stack.enter_context(mock.patch.object(legacy_uninstall, key, value))
                stack.enter_context(mock.patch.object(legacy_uninstall, 'run_command', return_value=mock.Mock(returncode=0)))
                if unreadable:
                    stack.enter_context(mock.patch.object(managed, 'digest', side_effect=PermissionError('unreadable unit')))
                legacy_uninstall.uninstall_command(yes=True, purge=True, skip_permissions=True)
            self.assertFalse((config / 'config.json').exists())
            self.assertEqual(venv.exists(), unreadable)

    def test_forward_compositor_reload_failure_does_not_roll_back(self):
        root, cli, *_ = self.managed_files()
        old = managed.read_json(self.install.current)
        old.update(version='v1.0.0', python={'path': '/python'}, selection=None, backend=None)
        managed.atomic_json(self.install.current, old)
        new_cli = self.install.data / 'environments/new-cli'
        new_cli.mkdir()
        binding = self.root / 'bindings.conf'
        binding.write_text('old binding')
        with ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, {'HYPRLAND_INSTANCE_SIGNATURE': 'stale'}))
            stack.enter_context(mock.patch.object(managed, 'interpreter', return_value=old['python']))
            stack.enter_context(mock.patch.object(managed, 'verify_payload', return_value={'version': old['version']}))
            stack.enter_context(mock.patch.object(managed.shutil, 'which', return_value='/hyprctl'))
            stack.enter_context(mock.patch.object(managed, 'run', side_effect=subprocess.CalledProcessError(1, ['hyprctl'], stderr='stale compositor')))
            import managed_integrations
            stack.enter_context(mock.patch.object(managed_integrations, 'Installation', return_value=self.install))
            for name, value in {'build': {'path': str(new_cli)}, 'legacy_selection': None, 'validate_config': None, 'check_integrations': None, 'service_state': {}, 'daemon_running': False, 'install_launcher': None, 'clean_legacy': None, 'binding_migrations': {binding: 'new binding'}, 'bar_migrations': {}}.items():
                stack.enter_context(mock.patch.object(self.install, name, return_value=value))
            result = self.install.update(local_payload=True)
        self.assertEqual(managed.read_json(self.install.current), result)
        self.assertEqual(binding.read_text(), 'new binding')
