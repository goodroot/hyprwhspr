import contextlib
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from cli import setup  # noqa: E402

PREFERENCE_PROMPTS = ('audio ducking', 'Mode', 'Reduction percentage', 'mic-osd', 'tray icon',
                      'keyboard allowlist', 'systemd user service', 'Set up permissions')


class ExpressSetupTests(unittest.TestCase):
    def _run(self, express, existing=None):
        """Run setup to the summary; save_selection raises before any host change.

        Every prompt answers with its own default, as pressing Enter would.
        """
        asked = []
        existing = {'transcription_backend': 'onnx-asr', **(existing or {})}

        def confirm(prompt, default=True, **_):
            asked.append(prompt)
            return express if 'recommended settings' in prompt else default

        def choose(prompt, default=None, **_):
            asked.append(prompt)
            return default

        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        output = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {'XDG_CURRENT_DESKTOP': 'test'}))
            os.environ.pop('HYPRLAND_INSTANCE_SIGNATURE', None)
            for name, value in (('_check_mise_active', (False, '')), ('_setup_command_symlink', None),
                                ('_run_keyboard_selection', None), ('_load_existing_setup_config', existing),
                                ('_prompt_backend_selection', ('onnx-asr', False, False, None)),
                                ('_detect_current_backend', 'onnx-asr'), ('_noctalia_detected', False),
                                ('detect_compositor', None), ('_is_gnome_or_mutter_session', False),
                                ('_is_kde_plasma_session', False), ('_hypr_config_dir', Path(home.name) / 'hypr'),
                                ('_check_mic_osd_availability', (True, ''))):
                stack.enter_context(patch.object(setup, name, return_value=value))
            stack.enter_context(patch.object(setup.Path, 'home', return_value=Path(home.name)))
            stack.enter_context(patch.object(setup, 'Prompt', types.SimpleNamespace(ask=choose)))
            stack.enter_context(patch.object(setup, 'Confirm', types.SimpleNamespace(ask=confirm)))
            stack.enter_context(patch.object(setup, 'save_selection', side_effect=RuntimeError('stop')))
            service = stack.enter_context(patch.object(setup, 'setup_systemd'))
            permissions = stack.enter_context(patch.object(setup, 'setup_permissions'))
            stack.enter_context(contextlib.redirect_stdout(output))
            with self.assertRaises(SystemExit):
                setup.setup_command(python_path='/tmp/mock-python')
        service.assert_not_called()
        permissions.assert_not_called()
        return asked, output.getvalue()

    @staticmethod
    def _summary(text):
        return text.split('Setup Summary', 1)[1].split('Running Setup', 1)[0]

    def test_recommended_asks_no_preference_questions(self):
        asked, _ = self._run(express=True)
        for fragment in PREFERENCE_PROMPTS:
            self.assertFalse([p for p in asked if fragment in p], fragment)
        self.assertIn('Use recommended settings?', asked)
        self.assertEqual(asked[-1], 'Proceed with setup?')

    def test_customize_still_asks_them(self):
        asked, _ = self._run(express=False)
        for fragment in ('audio ducking', 'mic-osd', 'keyboard allowlist', 'systemd user service', 'Set up permissions'):
            self.assertTrue([p for p in asked if fragment in p], fragment)

    def test_recommended_matches_pressing_enter_through_customize(self):
        for existing in ({}, {'audio_ducking': False, 'mic_osd_enabled': False, 'tray_icon_enabled': False},
                         {'audio_ducking_mode': 'pause', 'audio_ducking_percent': 30}):
            self.assertEqual(self._summary(self._run(True, existing)[1]),
                             self._summary(self._run(False, existing)[1]), existing)

    def test_recommended_hides_the_preference_explanations(self):
        _, text = self._run(express=True)
        for heading in ('Audio Ducking', 'Systemd Service', 'Permissions Setup', 'Recording Status Indicator'):
            self.assertNotIn(heading, text)


if __name__ == '__main__':
    unittest.main()
