"""Execute real payload entry points without touching desktop services."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest import mock

from tests.test_managed_install import ManagedFixture
from tests.test_install_second_review import ROOT
import managed_install as managed
import managed_integrations as integrations


class EighthReviewTests(ManagedFixture):
    def test_actual_historical_widget_migrates(self):
        # Fixture: config/noctalia/plugin/widget.luau at 2ba47e9^ (before rename).
        content = (ROOT / 'tests/fixtures/noctalia-pre-rename-widget.luau').read_text(encoding='utf-8')
        updated = integrations.managed_widget_content(content, '/managed/tray')
        self.assertIn('local tray = "/managed/tray"', updated)
        self.assertNotIn('/config/hyprland/hyprwhspr-tray.sh', updated)
        self.assertIn('local STATES', updated)

    def test_shared_symlink_edits_and_removal_preserve_link_and_user_changes(self):
        target = self.root / 'dotfiles/style.css'
        target.parent.mkdir()
        target.write_text('/* user */\n', encoding='utf-8')
        link = self.root / 'style.css'
        link.symlink_to(target)
        with mock.patch.object(integrations, 'Installation', return_value=self.install):
            with integrations.edit_files([link], [link]) as proceed:
                self.assertTrue(proceed)
                link.write_text('@import "app.css";\n/* user */\n', encoding='utf-8')
            self.assertIn(str(target), managed.read_json(self.install.receipt)['files'])
            target.write_text(target.read_text() + '/* later user edit */\n')
            with integrations.edit_files([link], [link], 'remove'):
                pass
        self.assertTrue(link.is_symlink())
        self.assertEqual(target.read_text(), '/* user */\n/* later user edit */\n')

    def test_dedicated_symlink_still_refused(self):
        target = self.root / 'foreign'
        target.write_text('foreign')
        link = self.root / 'module.jsonc'
        link.symlink_to(target)
        with mock.patch.object(integrations, 'Installation', return_value=self.install):
            with integrations.edit_files([link]) as proceed:
                self.assertFalse(proceed)
        self.assertEqual(target.read_text(), 'foreign')

    def test_sourced_bindings_migrate_and_variables_preserve_checkout(self):
        config = self.root / 'config/hypr'
        config.mkdir(parents=True)
        sourced = self.root / 'outside/binds.conf'
        sourced.parent.mkdir()
        sourced.write_text(f'bind = SUPER, D, exec, {self.install.data}/src/config/hyprland/hyprwhspr-tray.sh record\n')
        (config / 'hyprland.conf').write_text(f'source = {sourced}\n')
        changes = self.install.binding_migrations()
        self.assertIn('--managed-tray record', changes[sourced])
        sourced.write_text('bind = SUPER, D, exec, $app/config/hyprland/hyprwhspr-tray.sh record\n')
        self.assertIn(sourced, self.install.legacy_binding_references())
        legacy = self.install.data / 'src'
        legacy.mkdir(parents=True)
        with mock.patch.object(managed, 'run') as run:
            self.install.clean_legacy()
        run.assert_not_called()
        self.assertTrue(legacy.exists())

    def test_utf8_io_under_ascii_locale(self):
        code = '''import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import managed_install as m
p = Path(sys.argv[2])
p.write_bytes(b'{"notes":"\\xc3\\xa9"}')
assert m.read_json(p)['notes'] == '\\u00e9'
m.atomic_text(p, '\\u00e9 \\u2014 \\U0001f399')
assert p.read_bytes() == '\\u00e9 \\u2014 \\U0001f399'.encode('utf-8')
m.atomic_json(p, {'notes': '\\u00e9'})
assert m.read_json(p)['notes'] == '\\u00e9'
'''
        result = subprocess.run([sys.executable, '-I', '-B', '-X', 'utf8=0', '-c', code, str(ROOT / 'lib/src'), str(self.root / 'notes.json')], env=dict(os.environ, LC_ALL='C', LANG='C'), capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_payload_fallback_launcher_and_main_dispatch_never_write_bytecode(self):
        payload = self.root / 'fallback'
        (payload / 'bin').mkdir(parents=True)
        (payload / 'lib/src').mkdir(parents=True)
        shutil.copy2(ROOT / 'bin/hyprwhspr', payload / 'bin/hyprwhspr')
        (payload / 'release.json').write_text('{}')
        (payload / 'lib/helper.py').write_text('VALUE=1')
        (payload / 'lib/cli.py').write_text('import sys\nassert sys.dont_write_bytecode\nimport helper\n')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='0', XDG_DATA_HOME=str(self.root / 'empty-data'))
        result = subprocess.run(['bash', str(payload / 'bin/hyprwhspr'), '--help'], env=env, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        shutil.copy2(ROOT / 'lib/main.py', payload / 'lib/main.py')
        (payload / 'lib/src/managed_install.py').write_text('import sys\nassert sys.dont_write_bytecode\ndef main(args): return 0\n')
        result = subprocess.run([sys.executable, '-I', str(payload / 'lib/main.py'), 'update'], env=env, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(list(payload.rglob('*.pyc')), [])

    def test_actual_managed_uninstall_deletes_its_executing_payload(self):
        data = self.root / 'xdg-data/hyprwhspr'
        state = self.root / 'xdg-state/hyprwhspr'
        payload = data / 'releases/current'
        (payload / 'lib/src').mkdir(parents=True)
        cli = data / 'environments/cli'
        cli.mkdir(parents=True)
        for name in ('managed_install.py', 'managed_integrations.py', 'legacy_units.py'):
            content = (ROOT / 'lib/src' / name).read_text(encoding='utf-8')
            if name == 'managed_install.py':
                content = content.replace("if __name__ == '__main__':", "Installation.service_state = lambda self: {}\nInstallation.daemon_running = lambda self: False\nif __name__ == '__main__':")
            (payload / 'lib/src' / name).write_text(content, encoding='utf-8')
        launcher = data / 'launcher'
        shutil.copy2(ROOT / 'scripts/managed-launcher.sh', launcher)
        launcher.chmod(0o755)
        hint = data / 'interpreter'
        hint.write_text(sys.executable + '\n')
        command = self.root / '.local/bin/hyprwhspr'
        command.parent.mkdir(parents=True)
        command.symlink_to(launcher)
        managed.atomic_json(data / 'current.json', {'root': str(payload), 'cli': {'path': str(cli)}})
        managed.atomic_json(state / 'ownership.json', {'files': {str(command): {'target': str(launcher)}, str(launcher): {'sha256': managed.digest(launcher)}, str(hint): {'sha256': managed.digest(hint)}}})
        result = subprocess.run([str(launcher), 'uninstall', '--yes'], env=os.environ.copy(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(payload.exists(), result.stdout)
        self.assertFalse(cli.exists())
        self.assertFalse(command.is_symlink())
        self.assertEqual(managed.read_json(state / 'deferred-cleanup.json'), [])
