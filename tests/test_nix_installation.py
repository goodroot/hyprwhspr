"""Nix commands use packaged dependencies and preserve installation ownership."""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib' / 'src'))

from installation import is_nix_install, nix_command_error, require_supported_backend


class NixInstallationTests(unittest.TestCase):
    def test_ownership_is_explicit(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(is_nix_install())
            self.assertIsNone(nix_command_error(argparse.Namespace(command='setup')))
            require_supported_backend('nvidia')
        with mock.patch.dict(os.environ, {'HYPRWHSPR_INSTALL_METHOD': 'nix'}):
            self.assertTrue(is_nix_install())
            for backend in ('cpu', 'pywhispercpp'):
                require_supported_backend(backend)
            for backend in ('nvidia', 'vulkan', 'rest-api', 'onnx-asr', 'unknown'):
                with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, 'CPU transcription'):
                    require_supported_backend(backend)

    def run_cli(self, arguments, config=None, entrypoint='cli.py'):
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env.update(HYPRWHSPR_INSTALL_METHOD='nix', HYPRWHSPR_BACKEND_ENV='/nonexistent/nix-python',
                       XDG_CONFIG_HOME=tmp + '/config', XDG_DATA_HOME=tmp + '/data',
                       XDG_CACHE_HOME=tmp + '/cache', XDG_RUNTIME_DIR=tmp + '/runtime')
            config_path = Path(tmp) / 'config/hyprwhspr/config.json'
            if config is not None:
                config_path.parent.mkdir(parents=True)
                config_path.write_text(json.dumps(config))
            before = config_path.read_bytes() if config_path.exists() else None
            result = subprocess.run([sys.executable, str(ROOT / 'lib' / entrypoint), *arguments],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(config_path.read_bytes() if config_path.exists() else None, before)
            self.assertFalse((Path(tmp) / 'data').exists())
            return result

    def test_installation_mutations_are_rejected_before_importing_handlers(self):
        for command in (['setup'], ['setup', 'auto'], ['install', 'auto'], ['install', 'repair'],
                        ['update'], ['uninstall', '--yes'], ['backend', 'repair'], ['backend', 'reset'],
                        ['state', 'reset'], ['state', 'reset', '--all']):
            with self.subTest(command=command):
                result = self.run_cli(command)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn('Nix', result.stderr)
                self.assertNotIn('Traceback', result.stderr)

    def test_help_and_install_status_remain_available(self):
        for command in (['--help'], ['setup', '--help'], ['install', 'status']):
            with self.subTest(command=command):
                result = self.run_cli(command)
                self.assertEqual(result.returncode, 0, result.stderr)
                if command == ['install', 'status']:
                    self.assertIn('Nix-managed', result.stdout)

    def test_main_recovery_entrypoint_cannot_bypass_nix_ownership(self):
        for command, code in ((['update'], 1), (['install', 'repair'], 1), (['install', 'status'], 0)):
            with self.subTest(command=command):
                result = self.run_cli(command, entrypoint='main.py')
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertIn('Nix', result.stdout + result.stderr)

    def test_unsupported_backend_cannot_download(self):
        result = self.run_cli(['model', 'download', 'base'], {'transcription_backend': 'nvidia'})
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('CPU transcription', result.stdout + result.stderr)
        self.assertNotIn('Configuration loaded', result.stdout)

    def test_legacy_local_backend_is_supported(self):
        with mock.patch.dict(os.environ, {'HYPRWHSPR_INSTALL_METHOD': 'nix'}):
            require_supported_backend('local')

    def test_rejected_download_does_not_migrate_legacy_config(self):
        result = self.run_cli(['model', 'download', 'base'], {
            'transcription_backend': 'nvidia', 'use_hypr_bindings': True,
            'push_to_talk': True,
        })
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('CPU transcription', result.stderr)

    def test_non_installation_commands_are_not_blocked(self):
        with mock.patch.dict(os.environ, {'HYPRWHSPR_INSTALL_METHOD': 'nix'}):
            for command in ('config', 'record', 'status', 'model', 'transcribe', 'test'):
                self.assertIsNone(nix_command_error(argparse.Namespace(command=command)))
            for action in ('status', 'restart'):
                self.assertIsNone(nix_command_error(argparse.Namespace(command='systemd', systemd_action=action)))

    def test_only_nixos_owns_the_service(self):
        # Never exercise these through the CLI: off NixOS they write a real user unit.
        with mock.patch.dict(os.environ, {'HYPRWHSPR_INSTALL_METHOD': 'nix'}):
            for action in ('install', 'enable', 'disable'):
                args = argparse.Namespace(command='systemd', systemd_action=action)
                with self.subTest(action=action):
                    with mock.patch('installation.is_nixos', return_value=True):
                        self.assertIn('NixOS', nix_command_error(args))
                    with mock.patch('installation.is_nixos', return_value=False):
                        self.assertIsNone(nix_command_error(args))

    def test_input_group_advice_survives_nixos_rebuilds(self):
        from installation import input_group_hint
        with mock.patch('installation.is_nixos', return_value=True):
            self.assertIn('services.hyprwhspr.users', input_group_hint())
        with mock.patch('installation.is_nixos', return_value=False):
            self.assertEqual(input_group_hint(), 'sudo usermod -aG input $USER')

    def test_launcher_uses_packaged_python_and_ignores_old_managed_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / 'package'
            (package / 'bin').mkdir(parents=True)
            (package / 'lib').mkdir()
            (package / 'lib/cli.py').touch()
            (package / 'release.json').write_text('{"version":"test"}')
            launcher = package / 'bin/hyprwhspr'
            shutil.copy2(ROOT / 'bin/hyprwhspr', launcher)
            python = root / 'python/bin/python'
            python.parent.mkdir(parents=True)
            python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            python.chmod(0o755)
            old_launcher = root / 'data/hyprwhspr/launcher'
            old_launcher.parent.mkdir(parents=True)
            old_launcher.write_text('#!/bin/sh\nexit 99\n')
            old_launcher.chmod(0o755)
            env = os.environ.copy()
            env.update(HYPRWHSPR_INSTALL_METHOD='nix', HYPRWHSPR_BACKEND_ENV=str(root / 'python'),
                       HYPRWHSPR_NIX_ROOT=str(package), XDG_DATA_HOME=str(root / 'data'))
            for args, script in (([], 'main.py'), (['config', 'show'], 'cli.py'),
                                 (['test'], 'cli.py'), (['transcribe', 'sample.wav'], 'cli.py')):
                with self.subTest(args=args):
                    result = subprocess.run([str(launcher), *args], env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(str(package / 'lib' / script), result.stdout)
            # Another install started from a Nix process tree inherits the
            # wrapper's variables; it must keep its own (managed) behavior.
            env['HYPRWHSPR_NIX_ROOT'] = str(root / 'elsewhere')
            result = subprocess.run([str(launcher), 'config', 'show'], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 99, result.stderr)


class NixIntegrationTests(unittest.TestCase):
    """Integrations reference stable commands; store paths vanish after garbage collection."""

    def setUp(self):
        from tests.test_setup_command_scope import install, systemd
        from tests.test_noctalia_integration import noctalia
        from cli import waybar
        self.install, self.systemd, self.noctalia, self.waybar = install, systemd, noctalia, waybar
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = mock.patch.dict(os.environ, {'HYPRWHSPR_INSTALL_METHOD': 'nix',
                                                   'XDG_CONFIG_HOME': str(self.root / 'config')})
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop('HYPRWHSPR_GENERATION', None)

    def make_profile(self, profile_name='profile'):
        """A package with a profile linking it, behind another install and a store entry on PATH."""
        package = self.root / 'package'
        (package / 'lib/hyprwhspr').mkdir(parents=True, exist_ok=True)
        profile = self.root / profile_name / 'bin'
        other = self.root / 'other/bin'
        for directory in (package / 'bin', profile, other):
            directory.mkdir(parents=True, exist_ok=True)
        for name in ('hyprwhspr', 'hyprwhspr-tray'):
            for executable in (package / 'bin' / name, other / name):
                executable.write_text('#!/bin/sh\n')
                executable.chmod(0o755)
            (profile / name).symlink_to(package / 'bin' / name)
        os.environ['PATH'] = os.pathsep.join(['/nix/store/abc-hyprwhspr/bin', str(other), str(profile)])
        return package / 'lib/hyprwhspr', profile

    def test_profile_command_skips_other_installs_and_store_paths(self):
        from installation import nix_profile_command
        root, profile = self.make_profile()
        self.assertEqual(nix_profile_command('hyprwhspr', root), str(profile / 'hyprwhspr'))
        os.environ['PATH'] = str(self.root / 'other/bin')
        self.assertIsNone(nix_profile_command('hyprwhspr', root))

    def test_waybar_uses_profile_tray_and_removes_style_copy(self):
        root, profile = self.make_profile()
        waybar = self.root / 'config/waybar'
        with mock.patch.object(self.waybar, 'HYPRWHSPR_ROOT', str(ROOT)), \
                mock.patch.object(self.waybar, '_validate_hyprwhspr_root', return_value=True), \
                mock.patch('installation.nix_profile_command', return_value=str(profile / 'hyprwhspr-tray')):
            self.assertIsNot(self.waybar.setup_waybar('install'), False)
            module = json.loads((waybar / 'hyprwhspr-module.jsonc').read_text())['custom/hyprwhspr']
            # Absolute: bars started by systemd lack the profile on PATH.
            self.assertEqual(module['exec'], f"{profile / 'hyprwhspr-tray'} status")
            self.assertTrue((waybar / 'hyprwhspr-style.css').is_file())
            self.waybar.setup_waybar('remove')
        self.assertFalse((waybar / 'hyprwhspr-style.css').exists())

    def test_waybar_commented_removal_keeps_imported_style_copy(self):
        root, profile = self.make_profile()
        waybar = self.root / 'config/waybar'
        with mock.patch.object(self.waybar, 'HYPRWHSPR_ROOT', str(ROOT)), \
                mock.patch.object(self.waybar, '_validate_hyprwhspr_root', return_value=True), \
                mock.patch('installation.nix_profile_command', return_value=str(profile / 'hyprwhspr-tray')):
            self.assertIsNot(self.waybar.setup_waybar('install'), False)
            style = waybar / 'style.css'
            style.write_text(f'@import "{waybar / "hyprwhspr-style.css"}";\n', encoding='utf-8')
            config = waybar / 'config.jsonc'
            config.write_text('// mine\n' + config.read_text(encoding='utf-8'), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(self.waybar.setup_waybar('remove'))
            # style.css still imports it, so deleting it would break the bar's styling.
            self.assertTrue((waybar / 'hyprwhspr-style.css').is_file())

    def test_integrations_fail_without_profile_tray(self):
        from installation import tray_command
        destinations = {name: self.root / name for name in (
            'plugin_dir', 'legacy_plugin_dir', 'settings', 'template_input',
            'legacy_template_input', 'template_output')}
        shim = self.root / 'data/integrations/nix-noctalia-tray'
        os.environ['PATH'] = str(self.root / 'other/bin')
        with mock.patch('output_control.log_error') as error, \
                mock.patch.object(self.waybar, 'HYPRWHSPR_ROOT', str(ROOT)), \
                mock.patch.object(self.waybar, '_validate_hyprwhspr_root', return_value=True), \
                mock.patch.object(self.install, 'HYPRWHSPR_ROOT', str(ROOT)), \
                mock.patch.object(self.noctalia, 'HYPRWHSPR_ROOT', str(ROOT)), \
                mock.patch.object(self.noctalia, '_validate_hyprwhspr_root', return_value=True), \
                mock.patch.object(self.noctalia, '_noctalia_paths', return_value=destinations):
            self.assertIsNone(tray_command(str(ROOT)))
            self.assertFalse(self.waybar.setup_waybar('install'))
            self.assertFalse(self.install._edit_hyprland_bindings())
            self.assertFalse(self.noctalia._setup_noctalia('install', widget_content='widget', tray=shim))
            self.assertEqual(error.call_count, 4)
            # Removal never needs the tray, so it neither fails nor advises installing it.
            error.reset_mock()
            self.assertIsNot(self.waybar.setup_waybar('remove'), False)
            error.assert_not_called()
        self.assertFalse(shim.exists())
        self.assertFalse(destinations['plugin_dir'].exists())
        self.assertFalse((self.root / 'config/waybar/hyprwhspr-module.jsonc').exists())
        self.assertFalse((self.root / 'config/hypr').exists())

    def test_noctalia_widget_uses_shim_on_path(self):
        _, profile = self.make_profile()
        destinations = {name: self.root / name for name in (
            'plugin_dir', 'legacy_plugin_dir', 'settings', 'template_input',
            'legacy_template_input', 'template_output')}
        shim = self.root / 'data/integrations/nix-noctalia-tray'

        def observe_enable(*args):
            if args == ('plugins', 'enable', self.noctalia.NOCTALIA_PLUGIN_ID):
                self.assertTrue(os.access(shim, os.X_OK))
                self.assertIn(str(shim), (destinations['plugin_dir'] / 'widget.luau').read_text())
            return args == ('plugins', 'disable', self.noctalia.NOCTALIA_PLUGIN_ID) and disable_ok

        disable_ok = False
        with mock.patch.object(self.noctalia, '_noctalia_paths', return_value=destinations), \
                mock.patch.object(self.noctalia, 'HYPRWHSPR_ROOT', str(ROOT)), \
                mock.patch.object(self.noctalia, '_validate_hyprwhspr_root', return_value=True), \
                mock.patch.object(self.noctalia, 'DATA_DIR', self.root / 'data'), \
                mock.patch.object(self.noctalia, '_noctalia_msg', side_effect=observe_enable), \
                mock.patch.object(self.noctalia.shutil, 'which', return_value='/usr/bin/noctalia'), \
                mock.patch('installation.nix_profile_command', return_value=str(profile / 'hyprwhspr-tray')):
            self.noctalia.setup_noctalia('install')
            # Not noctalia-tray: the managed installer owns that file.
            self.assertEqual(shim.read_text(), f'#!/bin/sh\nexec {profile / "hyprwhspr-tray"} "$@"\n')
            self.assertTrue(os.access(shim, os.X_OK))
            self.assertIn(str(shim), (destinations['plugin_dir'] / 'widget.luau').read_text())
            # Copies must stay writable, or the next reinstall fails on store modes.
            self.assertTrue(os.access(destinations['plugin_dir'] / 'plugin.toml', os.W_OK))
            # Files stay while a running Noctalia still has the plugin enabled.
            self.assertFalse(self.noctalia.setup_noctalia('remove'))
            self.assertTrue(shim.exists())
            disable_ok = True
            self.noctalia.setup_noctalia('remove')
            self.assertFalse(shim.exists())

    def test_service_runs_this_packages_profile_launcher(self):
        template = (ROOT / 'config/systemd/hyprwhspr.service').read_text()
        root, profile = self.make_profile()
        with mock.patch.object(self.systemd, 'HYPRWHSPR_ROOT', str(root)):
            unit = self.systemd._nix_service_content(template)
            self.assertIn(f'\nExecStart=:"{profile / "hyprwhspr"}"\n', unit)
            self.assertNotIn('HYPRWHSPR_ROOT', unit)
            self.assertNotIn('/usr/lib/hyprwhspr', unit)
            # Another install or a store entry alone is never written into the unit.
            os.environ['PATH'] = os.pathsep.join(['/nix/store/abc-hyprwhspr/bin', str(self.root / 'other/bin')])
            with mock.patch.object(self.systemd, 'log_error'):
                self.assertIsNone(self.systemd._nix_service_content(template))

    def test_failed_noctalia_reinstall_preserves_existing_shim(self):
        shim = self.root / 'data/integrations/nix-noctalia-tray'
        shim.parent.mkdir(parents=True)
        original = '#!/bin/sh\nexec /previous/profile/bin/hyprwhspr-tray "$@"\n'
        shim.write_text(original)
        shim.chmod(0o750)
        with mock.patch.object(self.noctalia, '_validate_hyprwhspr_root', return_value=False):
            self.assertFalse(self.noctalia._setup_noctalia('install', widget_content='widget', tray=shim))
        self.assertEqual(shim.read_text(), original)
        self.assertEqual(shim.stat().st_mode & 0o777, 0o750)

    def test_failed_noctalia_first_install_does_not_publish_shim(self):
        shim = self.root / 'data/integrations/nix-noctalia-tray'
        with mock.patch.object(self.noctalia, '_validate_hyprwhspr_root', return_value=True), \
                mock.patch.object(self.noctalia, 'HYPRWHSPR_ROOT', str(self.root / 'missing')):
            self.assertFalse(self.noctalia._setup_noctalia('install', widget_content='widget', tray=shim))
        self.assertFalse(shim.exists())

    def test_service_escapes_profile_path_for_systemd(self):
        root, profile = self.make_profile('user %h $HOME "p"')
        with mock.patch.object(self.systemd, 'HYPRWHSPR_ROOT', str(root)):
            unit = self.systemd._nix_service_content('ExecStart=old\n')
        escaped = str(profile / 'hyprwhspr').replace('%', '%%').replace('"', '\\"')
        self.assertEqual(unit, f'ExecStart=:"{escaped}"\n')

    def test_service_reports_unsupported_backend_instead_of_crashing(self):
        from app import feedback
        app = mock.Mock(config=mock.Mock(get_setting=lambda key, default=None:
                                         'nvidia' if key == 'transcription_backend' else default))
        with mock.patch.dict(os.environ, {'HYPRWHSPR_INSTALL_METHOD': 'nix'}), \
                mock.patch.object(feedback, 'log') as log:
            self.assertEqual(feedback.FeedbackMixin._report_missing_dependencies(app), [])
        # WhisperManager logs the same error when it refuses the backend.
        log.assert_not_called()
        self.assertIn('CPU transcription', app._notify_user.call_args.args[1])
        self.assertEqual(app._notify_user.call_args.kwargs['urgency'], 'critical')


if __name__ == '__main__':
    unittest.main()
