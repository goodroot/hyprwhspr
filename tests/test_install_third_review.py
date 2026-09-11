"""Third review regressions; all user paths and external operations are isolated."""
import os
from pathlib import Path
import subprocess
from unittest import mock

from tests.test_managed_install import ManagedFixture
from tests.test_install_second_review import noctalia, ROOT
import managed_install as managed
import managed_integrations as integrations


class ThirdReviewTests(ManagedFixture):
    def test_health_ignores_unrelated_distro_dependency_errors(self):
        def probe(command, **kwargs):
            if 'check' in command:
                raise subprocess.CalledProcessError(1, command, stderr='glances requires pyinstrument')
        with mock.patch.object(managed, 'run', side_effect=probe) as run:
            self.assertTrue(self.install.healthy('/owned/env', ['rich', 'jsonschema']))
            self.assertIn('import rich', run.call_args.args[0][-1])
        with mock.patch.object(managed, 'run', side_effect=subprocess.CalledProcessError(1, ['python'])):
            self.assertFalse(self.install.healthy('/owned/env', ['rich']))

    def test_running_cli_is_retained_until_later_process_cleanup(self):
        path = self.install.data / 'environments/old-cli'
        path.mkdir(parents=True)
        with mock.patch.object(managed.sys, 'prefix', str(path)):
            self.install.remove([str(path)])
            self.install.recover()
            self.assertTrue(path.exists())
        self.install.recover()
        self.assertFalse(path.exists())

    def test_unowned_integration_skips_editor_without_adopting_file(self):
        path = self.root / 'hyprwhspr-module.jsonc'
        path.write_text('legacy or customized')
        with mock.patch.object(integrations, 'Installation', return_value=self.install):
            with integrations.edit_files([path]) as proceed:
                self.assertFalse(proceed)
        self.assertEqual(path.read_text(), 'legacy or customized')
        self.assertFalse(self.install.receipt.exists())

    def test_unrecorded_legacy_hotkey_migrates_preserving_other_commands(self):
        path = self.root / 'config/hypr/bindings.conf'
        path.parent.mkdir(parents=True)
        foreign = 'bind = SUPER, X, exec, /other/config/hyprland/hyprwhspr-tray.sh record\n'
        path.write_text(f'bind = SUPER, D, exec, {self.install.data}/src/config/hyprland/hyprwhspr-tray.sh record\n' + foreign)
        result = self.install.binding_migrations()[path]
        self.assertIn('--managed-tray record', result)
        self.assertIn(foreign, result)

    def test_modified_waybar_config_keeps_referenced_module_until_retry(self):
        base = self.root / 'waybar'
        base.mkdir()
        config = base / 'config.jsonc'
        module = base / 'hyprwhspr-module.jsonc'
        with mock.patch.object(integrations, 'Installation', return_value=self.install):
            with integrations.edit_files([config, module], [config]):
                config.write_text('{"include": ["hyprwhspr-module.jsonc"]}')
                module.write_text('{}')
            config.write_text('{"include": ["hyprwhspr-module.jsonc"], "user": true}')
            with integrations.edit_files([module, config], [config], 'remove'):
                pass
            self.assertTrue(module.exists())
            with mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(self.install, 'check_integrations'), mock.patch.object(self.install, 'service_state', return_value={}), mock.patch.object(self.install, 'daemon_running', return_value=False):
                managed.uninstall(['--yes'])
            self.assertTrue(module.exists(), 'Whole-install removal must also preserve referenced modules')
            self.assertIn(str(module), managed.read_json(self.install.receipt)['files'])
            config.write_text('{"user": true}')
            with integrations.edit_files([module, config], [config], 'remove'):
                pass
            self.assertFalse(module.exists())

    def test_noctalia_absent_allows_leftover_cleanup(self):
        plugin = self.root / 'plugin'
        plugin.mkdir()
        widget = plugin / 'widget.luau'
        widget.write_text('owned')
        managed.atomic_json(self.install.receipt, {'files': {str(widget): {'sha256': managed.digest(widget), 'original': None}}})
        dst = {'plugin_dir': plugin, 'settings': self.root / 'settings', 'template_input': self.root / 'input', 'template_output': self.root / 'output'}
        with mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}'}), mock.patch.object(noctalia, '_noctalia_paths', return_value=dst), mock.patch.object(noctalia, 'HYPRWHSPR_ROOT', str(ROOT)), mock.patch.object(noctalia.shutil, 'which', return_value=None), mock.patch.object(noctalia, '_noctalia_msg') as message, mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(integrations, 'Installation', return_value=self.install):
            self.assertTrue(noctalia.setup_noctalia('remove'))
        message.assert_not_called()
        self.assertFalse(widget.exists())

    def test_legacy_build_keeps_pip_sources_and_excludes_mise_tools(self):
        mise = self.root / 'mise'
        env = {'MISE_DATA_DIR': str(mise), 'PATH': f'{mise}/shims:/usr/bin',
               'PIP_CONFIG_FILE': '/custom/pip.conf', 'PIP_INDEX_URL': 'https://mirror.invalid',
               'HTTPS_PROXY': 'http://proxy.invalid', 'PYTHONPATH': '/hostile'}
        with mock.patch.dict(os.environ, env):
            result = managed.legacy_build_env()
        self.assertEqual(result['PATH'], '/usr/bin')
        for key in ('PIP_CONFIG_FILE', 'PIP_INDEX_URL', 'HTTPS_PROXY'):
            self.assertEqual(result[key], env[key])
        self.assertEqual(result['PYTHONPATH'], env['PYTHONPATH'])
        self.assertNotIn('MISE_DATA_DIR', result)
        with mock.patch.dict(os.environ, {'MISE_DATA_DIR': str(mise), 'PATH': str(mise / 'shims'), 'PYTHONNOUSERSITE': '1'}):
            result = managed.legacy_build_env()
        self.assertEqual(result['PATH'], os.defpath)
        self.assertEqual(result['PYTHONNOUSERSITE'], '1')
