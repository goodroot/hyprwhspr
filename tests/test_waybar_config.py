"""Waybar setup must not rewrite a user's config.jsonc beyond the entries it adds."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests.test_setup_command_scope import ROOT
from cli import waybar

COMMENTED = '''// My bar
{
    "layer": "top",
    // right side
    "modules-right": ["battery", "clock"],
    "battery": { "format": "\U000f0084 {capacity}%" }
}
'''


class WaybarConfigPreservationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = self.root / 'waybar' / 'config.jsonc'
        self.config.parent.mkdir()
        self.module_file = str(self.config.parent / 'hyprwhspr-module.jsonc')
        for patcher in (mock.patch.object(waybar, 'HYPRWHSPR_ROOT', str(ROOT)),
                        mock.patch.object(waybar, '_validate_hyprwhspr_root', return_value=True),
                        mock.patch.dict(os.environ, {'XDG_CONFIG_HOME': str(self.root)}, clear=True)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_waybar(self, mode, expected=True):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertIs(waybar.setup_waybar(mode), expected)
        return output.getvalue()

    def test_commented_config_is_left_unchanged_with_edit_instructions(self):
        self.config.write_text(COMMENTED, encoding='utf-8')
        output = self.run_waybar('install')
        self.assertEqual(self.config.read_text(encoding='utf-8'), COMMENTED)
        self.assertIn(f'"include": ["{self.module_file}"]', output)
        self.assertIn('"custom/hyprwhspr" to "modules-right"', output)
        self.assertTrue(Path(self.module_file).is_file())

    def test_uncommented_config_keeps_non_ascii_text(self):
        self.config.write_text(json.dumps({'modules-right': ['clock'], 'clock': {'format': '\U000f0084'}}),
                               encoding='utf-8')
        self.run_waybar('install')
        content = self.config.read_text(encoding='utf-8')
        self.assertIn('\U000f0084', content)
        config = json.loads(content)
        self.assertEqual(config['modules-right'], ['clock', 'custom/hyprwhspr'])
        self.assertEqual(config['include'], [self.module_file])

    def test_configured_or_grouped_module_is_not_rewritten_or_duplicated(self):
        original = json.dumps({'include': [self.module_file], 'modules-right': ['group/voice'],
                               'group/voice': {'modules': ['custom/hyprwhspr']}})
        self.config.write_text(original, encoding='utf-8')
        self.run_waybar('install')
        self.assertEqual(self.config.read_text(encoding='utf-8'), original)

    def test_remove_leaves_commented_config_unchanged_with_edit_instructions(self):
        configured = COMMENTED.replace('["battery", "clock"]', '["battery", "custom/hyprwhspr"]')
        self.config.write_text(configured, encoding='utf-8')
        output = self.run_waybar('remove', expected=False)
        self.assertEqual(self.config.read_text(encoding='utf-8'), configured)
        self.assertIn('"custom/hyprwhspr" from "modules-right"', output)

    def test_unused_group_does_not_count_as_placement(self):
        config = {'include': [self.module_file], 'modules-right': ['clock'],
                  'group/unused': {'modules': ['custom/hyprwhspr']}}
        self.config.write_text(json.dumps(config), encoding='utf-8')
        Path(self.module_file).write_text('{}', encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(waybar.waybar_status())
        self.run_waybar('install')
        updated = json.loads(self.config.read_text(encoding='utf-8'))
        self.assertIn('custom/hyprwhspr', updated['modules-right'])

    def test_nested_groups_and_cycles(self):
        for section in ('modules-left', 'modules-center', 'modules-right'):
            with self.subTest(section=section):
                config = {section: ['group/outer'],
                          'group/outer': {'modules': ['group/inner']},
                          'group/inner': {'modules': ['group/outer']}}
                self.assertFalse(waybar._module_placed(config))
                config['group/inner']['modules'].append('custom/hyprwhspr')
                self.assertTrue(waybar._module_placed(config))

    def test_commented_removal_preserves_dependencies_until_retry(self):
        original = '// Keep my comments\n' + json.dumps({'include': [self.module_file]})
        self.config.write_text(original, encoding='utf-8')
        Path(self.module_file).write_text('{}', encoding='utf-8')
        style = self.config.parent / 'style.css'
        css = f'@import "{ROOT / "config/waybar/hyprwhspr-style.css"}";\n'
        style.write_text(css, encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'removal incomplete'):
                waybar.waybar_command('remove')
        self.assertEqual(self.config.read_text(encoding='utf-8'), original)
        self.assertTrue(Path(self.module_file).exists())
        self.assertEqual(style.read_text(encoding='utf-8'), css)
        self.config.write_text('// Keep my comments\n{}', encoding='utf-8')
        self.run_waybar('remove')
        self.assertFalse(Path(self.module_file).exists())
        self.assertEqual(style.read_text(encoding='utf-8'), '')

    def test_string_include_is_extended_and_removed(self):
        self.config.write_text(json.dumps({'include': '/etc/xdg/waybar/base.jsonc',
                                           'modules-right': ['clock']}), encoding='utf-8')
        self.run_waybar('install')
        config = json.loads(self.config.read_text(encoding='utf-8'))
        self.assertEqual(config['include'], ['/etc/xdg/waybar/base.jsonc', self.module_file])
        self.config.write_text(json.dumps({'include': self.module_file,
                                           'modules-right': ['clock', 'custom/hyprwhspr']}), encoding='utf-8')
        self.run_waybar('remove')
        self.assertEqual(json.loads(self.config.read_text(encoding='utf-8')), {'modules-right': ['clock']})

    def test_remove_drops_include_left_empty(self):
        self.config.write_text(json.dumps({'include': [self.module_file],
                                           'modules-right': ['clock', 'custom/hyprwhspr']}), encoding='utf-8')
        self.run_waybar('remove')
        self.assertEqual(json.loads(self.config.read_text(encoding='utf-8')), {'modules-right': ['clock']})

    def test_remove_cleans_all_reachable_placements(self):
        config = {'include': [self.module_file],
                  'modules-left': ['custom/hyprwhspr', 'clock', 'custom/hyprwhspr'],
                  'modules-center': ['custom/hyprwhspr'],
                  'modules-right': ['group/outer'],
                  'group/outer': {'modules': ['group/inner', 'custom/hyprwhspr']},
                  'group/inner': {'modules': ['group/outer', 'custom/hyprwhspr', 'tray']},
                  'group/unused': {'modules': ['custom/hyprwhspr']}}
        self.config.write_text(json.dumps(config), encoding='utf-8')
        Path(self.module_file).write_text('{}', encoding='utf-8')
        self.run_waybar('remove')
        updated = json.loads(self.config.read_text(encoding='utf-8'))
        self.assertNotIn('include', updated)
        self.assertEqual(updated['modules-left'], ['clock'])
        self.assertEqual(updated['modules-center'], [])
        self.assertEqual(updated['modules-right'], ['group/outer'])
        self.assertEqual(updated['group/outer']['modules'], ['group/inner'])
        self.assertEqual(updated['group/inner']['modules'], ['group/outer', 'tray'])
        self.assertEqual(updated['group/unused'], config['group/unused'])
        self.assertFalse(Path(self.module_file).exists())

    def test_commented_removal_reports_each_placement_until_removed(self):
        for location in ('modules-left', 'modules-center', 'group/voice.modules'):
            with self.subTest(location=location):
                if location.startswith('group/'):
                    config = {'modules-left': ['group/voice'],
                              'group/voice': {'modules': ['custom/hyprwhspr']}}
                else:
                    config = {location: ['custom/hyprwhspr']}
                config['include'] = [self.module_file]
                original = '// My bar\n' + json.dumps(config)
                self.config.write_text(original, encoding='utf-8')
                Path(self.module_file).write_text('{}', encoding='utf-8')
                output = self.run_waybar('remove', expected=False)
                self.assertIn(f'"custom/hyprwhspr" from "{location}"', output)
                self.assertEqual(self.config.read_text(encoding='utf-8'), original)
                self.assertTrue(Path(self.module_file).exists())
                # Removing only the include must not allow deletion on retry.
                del config['include']
                self.config.write_text('// My bar\n' + json.dumps(config), encoding='utf-8')
                self.run_waybar('remove', expected=False)
                self.assertTrue(Path(self.module_file).exists())
                self.config.write_text('// My bar\n{}', encoding='utf-8')
                self.run_waybar('remove')
                self.assertFalse(Path(self.module_file).exists())

    def test_remove_keeps_other_includes(self):
        self.config.write_text(json.dumps({'include': ['~/.config/waybar/other.jsonc', self.module_file]}),
                               encoding='utf-8')
        self.run_waybar('remove')
        self.assertEqual(json.loads(self.config.read_text(encoding='utf-8')),
                         {'include': ['~/.config/waybar/other.jsonc']})


if __name__ == '__main__':
    unittest.main()
