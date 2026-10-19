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

    def run_waybar(self, mode):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertIsNot(waybar.setup_waybar(mode), False)
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
        output = self.run_waybar('remove')
        self.assertEqual(self.config.read_text(encoding='utf-8'), configured)
        self.assertIn('"custom/hyprwhspr" from "modules-right"', output)

    def test_remove_drops_include_left_empty(self):
        self.config.write_text(json.dumps({'include': [self.module_file],
                                           'modules-right': ['clock', 'custom/hyprwhspr']}), encoding='utf-8')
        self.run_waybar('remove')
        self.assertEqual(json.loads(self.config.read_text(encoding='utf-8')), {'modules-right': ['clock']})

    def test_remove_keeps_other_includes(self):
        self.config.write_text(json.dumps({'include': ['~/.config/waybar/other.jsonc', self.module_file]}),
                               encoding='utf-8')
        self.run_waybar('remove')
        self.assertEqual(json.loads(self.config.read_text(encoding='utf-8')),
                         {'include': ['~/.config/waybar/other.jsonc']})


if __name__ == '__main__':
    unittest.main()
