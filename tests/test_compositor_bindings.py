import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from cli import bindings  # noqa: E402

TRAY = '/opt/hw/config/hyprland/hyprwhspr-tray.sh'
NIRI = 'input {\n}\n\nbinds {\n    Mod+T { spawn "kitty"; }\n}\n'


class CompositorBindingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.home = Path(temp.name)
        self.validated = []
        self.valid = True
        env = {'XDG_CONFIG_HOME': str(self.home / 'config')}
        for key in ('HYPRWHSPR_GENERATION', 'SWAYSOCK'):
            os.environ.pop(key, None)
        patches = [
            mock.patch.dict(os.environ, env),
            mock.patch.object(bindings, 'USER_HOME', self.home),
            mock.patch.object(bindings, 'stable_tray_command', return_value=TRAY),
            mock.patch.object(bindings.shutil, 'which', side_effect=lambda name: f'/usr/bin/{name}'),
            mock.patch.object(bindings, '_passes', side_effect=self.passes),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def passes(self, command):
        # Read the candidate while it exists: it is removed after the swap.
        if command[0] in ('niri', 'sway'):
            self.validated.append(Path(command[-1]).read_text(encoding='utf-8'))
        return self.valid

    def write(self, relative, text):
        path = self.home / 'config' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        return path

    def test_niri_line_goes_inside_the_existing_binds_block(self):
        path = self.write('niri/config.kdl', NIRI)
        self.assertTrue(bindings.setup_compositor_binding('niri'))
        text = path.read_text()
        self.assertEqual(text.count('binds {'), 1)
        self.assertIn(f'binds {{\n    // {bindings.MARKER}\n    Mod+Alt+D {{ spawn "{TRAY}" "record"; }}\n    Mod+T', text)
        self.assertEqual(self.validated, [text])
        self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ['config.kdl'])

    def test_niri_rerun_is_a_no_op(self):
        path = self.write('niri/config.kdl', NIRI)
        self.assertTrue(bindings.setup_compositor_binding('niri'))
        once = path.read_text()
        self.assertTrue(bindings.setup_compositor_binding('niri'))
        self.assertEqual(path.read_text(), once)

    def test_users_own_record_binding_counts_but_other_commands_do_not(self):
        own = NIRI.replace('Mod+T { spawn "kitty"; }', 'Mod+D { spawn "hyprwhspr" "record" "toggle"; }')
        path = self.write('niri/config.kdl', own)
        self.assertTrue(bindings.setup_compositor_binding('niri'))
        self.assertEqual(path.read_text(), own)
        other = NIRI.replace('Mod+T { spawn "kitty"; }', 'Mod+U { spawn "hyprwhspr" "model" "unload"; }\n    // hyprwhspr record')
        path.write_text(other)
        self.assertTrue(bindings.setup_compositor_binding('niri'))
        self.assertIn('Mod+Alt+D', path.read_text())

    def test_failed_validation_leaves_the_file_untouched(self):
        path = self.write('niri/config.kdl', NIRI)
        self.valid = False
        self.assertFalse(bindings.setup_compositor_binding('niri'))
        self.assertEqual(path.read_text(), NIRI)
        self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ['config.kdl'])

    def test_niri_without_a_binds_block_or_binary_is_manual(self):
        path = self.write('niri/config.kdl', 'include "binds.kdl"\n')
        self.assertFalse(bindings.setup_compositor_binding('niri'))
        path.write_text(NIRI)
        with mock.patch.object(bindings.shutil, 'which', return_value=None):
            self.assertFalse(bindings.setup_compositor_binding('niri'))
        self.assertEqual(path.read_text(), NIRI)
        self.assertEqual(self.validated, [])

    def test_symlinked_config_keeps_its_link(self):
        real = self.write('dotfiles/config.kdl', NIRI)
        link = self.home / 'config/niri/config.kdl'
        link.parent.mkdir(parents=True)
        link.symlink_to(real)
        self.assertTrue(bindings.setup_compositor_binding('niri'))
        self.assertTrue(link.is_symlink())
        self.assertIn('Mod+Alt+D', real.read_text())

    def test_managed_tray_command_is_split_into_spawn_arguments(self):
        path = self.write('niri/config.kdl', NIRI)
        with mock.patch.object(bindings, 'stable_tray_command', return_value="'/a b/launcher' --managed-tray"):
            self.assertTrue(bindings.setup_compositor_binding('niri'))
        self.assertIn('spawn "/a b/launcher" "--managed-tray" "record";', path.read_text())

    def test_sway_appends_and_reloads(self):
        path = self.write('sway/config', 'set $mod Mod4\nbindsym $mod+Return exec foot')
        with mock.patch.dict(os.environ, {'SWAYSOCK': '/run/sway.sock'}):
            self.assertTrue(bindings.setup_compositor_binding('sway'))
        self.assertTrue(path.read_text().endswith(
            f'exec foot\n\n# {bindings.MARKER}\nbindsym Mod4+Mod1+d exec {TRAY} record\n'))
        self.assertEqual(bindings._passes.call_args.args[0], ['swaymsg', 'reload'])

    def test_sway_chord_already_bound_is_manual(self):
        for taken in ('bindsym $mod+Mod1+d exec rofi', 'bindsym --release Mod1+Mod4+D exec rofi'):
            path = self.write('sway/config', taken + '\n')
            self.assertFalse(bindings.setup_compositor_binding('sway'))
            self.assertEqual(path.read_text(), taken + '\n')
        path = self.write('sway/config', 'bindsym $mod+d exec rofi\n')
        self.assertTrue(bindings.setup_compositor_binding('sway'))

    def gsettings(self, state, fail=()):
        def run(*args):
            if args[0] == 'get' and args[2] == 'custom-keybindings':
                return state['list']
            if args[0] == 'get':
                return state['bindings'].get(args[1].split(':', 1)[1], "''")
            if args[2] in fail:
                return None
            if args[2] == 'custom-keybindings':
                state['list'] = args[3]
            else:
                state.setdefault('set', {})[args[2]] = args[3]
            return ''
        return mock.patch.object(bindings, '_gsettings', side_effect=run)

    def test_gnome_adds_entry_and_keeps_existing_ones(self):
        state = {'list': "['/other/']", 'bindings': {'/other/': "'<Super>e'"}}
        with self.gsettings(state):
            self.assertTrue(bindings.setup_compositor_binding('gnome'))
            self.assertEqual(state['list'], str(['/other/', bindings.GNOME_PATH]))
            self.assertEqual(state['set'], {'name': 'hyprwhspr dictation', 'command': f'{TRAY} record',
                                            'binding': bindings.GNOME_BINDING})
            state.pop('set')
            self.assertTrue(bindings.setup_compositor_binding('gnome'))
            self.assertNotIn('set', state)

    def test_gnome_empty_list_and_taken_chord(self):
        state = {'list': '@as []', 'bindings': {}}
        with self.gsettings(state):
            self.assertTrue(bindings.setup_compositor_binding('gnome'))
        state = {'list': "['/other/']", 'bindings': {'/other/': "'<Super><Alt>D'"}}
        with self.gsettings(state):
            self.assertFalse(bindings.setup_compositor_binding('gnome'))
        self.assertEqual(state['list'], "['/other/']")

    def test_gnome_failed_write_never_registers_the_entry(self):
        state = {'list': '@as []', 'bindings': {}}
        with self.gsettings(state, fail=('binding',)):
            self.assertFalse(bindings.setup_compositor_binding('gnome'))
        self.assertEqual(state['list'], '@as []')

    def xfconf(self, state, fail=()):
        def run(*args):
            if '-n' in args:
                if 'create' in fail:
                    return None
                state['property'] = args[args.index('-s') + 1]
                return ''
            if '-p' in args:
                return state.get('property')
            return None
        return mock.patch.object(bindings, '_xfconf', side_effect=run)

    def test_xfce_creates_the_shortcut_property(self):
        state = {}
        with self.xfconf(state):
            self.assertTrue(bindings.setup_compositor_binding('xfce'))
        self.assertEqual(state['property'], f'{TRAY} record')

    def test_xfce_rerun_is_a_no_op(self):
        state = {'property': f'{TRAY} record'}
        with self.xfconf(state):
            self.assertTrue(bindings.setup_compositor_binding('xfce'))

    def test_xfce_users_own_record_binding_is_left_alone(self):
        state = {'property': 'hyprwhspr record toggle'}
        with self.xfconf(state):
            self.assertTrue(bindings.setup_compositor_binding('xfce'))
        self.assertEqual(state['property'], 'hyprwhspr record toggle')

    def test_xfce_taken_chord_is_manual(self):
        state = {'property': 'xfce4-popup-applicationsmenu'}
        with self.xfconf(state):
            self.assertFalse(bindings.setup_compositor_binding('xfce'))
        self.assertEqual(state['property'], 'xfce4-popup-applicationsmenu')

    def test_xfce_failed_create_is_manual(self):
        state = {}
        with self.xfconf(state, fail=('create',)):
            self.assertFalse(bindings.setup_compositor_binding('xfce'))
        self.assertNotIn('property', state)

    def test_xfce_without_xfconf_is_manual(self):
        state = {}
        with mock.patch.object(bindings.shutil, 'which', return_value=None), self.xfconf(state):
            self.assertFalse(bindings.setup_compositor_binding('xfce'))
        self.assertNotIn('property', state)

    def test_kde_and_unknown_never_claim_the_shortcut(self):
        self.assertFalse(bindings.setup_compositor_binding('kde'))
        self.assertFalse(bindings.setup_compositor_binding('cosmic'))

    def test_detection_prefers_the_running_compositor(self):
        with mock.patch.object(bindings, '_is_niri_session', return_value=True):
            self.assertEqual(bindings.detect_compositor(), 'niri')
        with mock.patch.object(bindings, '_is_niri_session', return_value=False), \
                mock.patch('sway_focus.is_sway_session', return_value=False), \
                mock.patch.object(bindings, '_is_gnome_or_mutter_session', return_value=False), \
                mock.patch.object(bindings, '_is_kde_plasma_session', return_value=True):
            self.assertEqual(bindings.detect_compositor(), 'kde')

    def test_detection_reports_xfce(self):
        with mock.patch.object(bindings, '_is_niri_session', return_value=False), \
                mock.patch('sway_focus.is_sway_session', return_value=False), \
                mock.patch.object(bindings, '_is_gnome_or_mutter_session', return_value=False), \
                mock.patch.object(bindings, '_is_kde_plasma_session', return_value=False), \
                mock.patch.object(bindings, '_is_xfce_session', return_value=True):
            self.assertEqual(bindings.detect_compositor(), 'xfce')


if __name__ == '__main__':
    unittest.main()
