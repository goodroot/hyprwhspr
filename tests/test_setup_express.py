import contextlib
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from tests.test_setup_command_scope import setup, systemd  # noqa: E402

PREFERENCE_PROMPTS = ('audio ducking', 'Mode', 'Reduction percentage', 'mic-osd', 'tray icon',
                      'keyboard allowlist', 'systemd user service', 'Set up permissions')


class ExpressSetupTests(unittest.TestCase):
    def _run(self, express, existing=None, complete=False, validation=True,
             service_ok=True, relogin=False, permission_error=False, indicator=True,
             compositor=None, binding_ok=True, service_choice=True,
             backend='onnx-asr', events=None, download_ok=True):
        """Run setup to the summary; save_selection raises before any host change.

        Every prompt answers with its own default, as pressing Enter would.
        """
        asked = []
        existing = {'transcription_backend': backend, **(existing or {})}

        def confirm(prompt, default=True, **_):
            asked.append(prompt)
            if prompt == 'Set up systemd user service?':
                return service_choice
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
                                ('_prompt_backend_selection', (backend, False, False, None)),
                                ('_prompt_model_selection', 'small.en'),
                                ('setup_config', None),
                                ('_detect_current_backend', backend), ('_noctalia_detected', False),
                                ('detect_compositor', compositor), ('_is_gnome_or_mutter_session', False),
                                ('_is_kde_plasma_session', False), ('_hypr_config_dir', Path(home.name) / 'hypr'),
                                ('_check_mic_osd_availability', (indicator, 'GTK missing'))):
                stack.enter_context(patch.object(setup, name, return_value=value))
            stack.enter_context(patch.object(setup.Path, 'home', return_value=Path(home.name)))
            stack.enter_context(patch.object(setup, 'Prompt', types.SimpleNamespace(ask=choose)))
            stack.enter_context(patch.object(setup, 'Confirm', types.SimpleNamespace(ask=confirm)))
            stack.enter_context(patch.object(setup, 'save_selection', side_effect=None if complete else RuntimeError('stop')))
            service = stack.enter_context(patch.object(setup, 'setup_systemd', return_value=service_ok))
            if events is not None:
                service.side_effect = lambda *a, **kw: events.append('service') or service_ok
            stack.enter_context(patch.object(setup, 'download_model',
                                side_effect=lambda *a: (events.append('download') if events is not None else None) or download_ok))
            def permissions_result(*args, issues=None, **kwargs):
                if permission_error and issues is not None:
                    issues.append('Resolve the permission errors above, then rerun: hyprwhspr setup')
                return relogin
            permissions = stack.enter_context(patch.object(setup, 'setup_permissions', side_effect=permissions_result))
            if complete:
                config = Mock()
                config.get_setting.side_effect = lambda key, default=None: existing.get(key, default)
                config.set_setting.side_effect = lambda key, value: existing.update({key: value})
                stack.enter_context(patch.object(setup, 'ConfigManager', return_value=config))
                for name, result in (('mic_osd_enable', None), ('mic_osd_disable', None),
                                     ('validate_command', validation), ('setup_compositor_binding', binding_ok),
                                     ('systemd_restart', service_ok)):
                    stack.enter_context(patch.object(setup, name, return_value=result))
                if events is not None:
                    stack.enter_context(patch.object(setup, 'systemd_restart',
                                        side_effect=lambda: events.append('restart') or service_ok))
                stack.enter_context(patch.object(setup.subprocess, 'run',
                                    return_value=Mock(returncode=0 if events is not None else 1)))
                stack.enter_context(patch.object(setup, '_is_running_manually', return_value=False))
            stack.enter_context(contextlib.redirect_stdout(output))
            if complete:
                setup.setup_command(python_path='/tmp/mock-python')
            else:
                with self.assertRaises(SystemExit):
                    setup.setup_command(python_path='/tmp/mock-python')
        if not complete:
            service.assert_not_called()
            permissions.assert_not_called()
        return asked, output.getvalue()

    def test_whisper_download_finishes_before_service_install_and_restart(self):
        events = []
        self._run(True, complete=True, backend='cpu', events=events)
        self.assertEqual(events, ['download', 'service', 'restart'])

    def test_failed_whisper_download_reports_required_repair(self):
        _, text = self._run(True, complete=True, backend='cpu', download_ok=False)
        self.assertIn('hyprwhspr model download small.en', text)
        self.assertIn('Setup finished with steps remaining.', text)
        self.assertNotIn('to try recording.', text)

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

    def test_completed_flow_names_configured_keyboard_shortcut(self):
        _, text = self._run(True, {'primary_shortcut': 'CTRL+F9'}, complete=True)
        self.assertIn('Setup finished.', text)
        self.assertIn('Press CTRL+F9 to try recording.', text)
        self.assertIn('permissions may ask for your password', text)

    def test_external_shortcut_does_not_assume_default_chord(self):
        _, text = self._run(True, {'external_bindings': True}, complete=True)
        self.assertIn('Use your desktop dictation shortcut', text)
        self.assertNotIn('Press SUPER+ALT+D', text)

    def test_incomplete_flows_report_actions_without_ready_message(self):
        for options, expected in (({'validation': False}, 'hyprwhspr validate'),
                                  ({'service_ok': False}, 'background service errors'),
                                  ({'permission_error': True}, 'permission errors'),
                                  ({'relogin': True}, 'Log out and back in'),
                                  ({'compositor': 'niri', 'binding_ok': False}, 'Add the niri shortcut')):
            with self.subTest(options=options):
                _, text = self._run(True, complete=True, **options)
                self.assertIn('Setup finished with steps remaining.', text)
                self.assertIn(expected, text)
                self.assertNotIn('to try recording.', text)

    def test_recommended_keeps_dependency_warning_visible(self):
        _, text = self._run(True, complete=True, indicator=False)
        self.assertIn('Recording overlay dependencies not found.', text)

    def test_declined_service_explains_manual_start(self):
        _, text = self._run(False, complete=True, service_choice=False)
        self.assertIn('Start hyprwhspr in a terminal with: hyprwhspr', text)

    def test_restart_failure_is_not_reported_as_success(self):
        with patch.object(systemd, 'run_command', return_value=Mock(returncode=1)), \
                patch.object(systemd, 'log_success') as success:
            self.assertFalse(systemd.systemd_restart())
        success.assert_not_called()


if __name__ == '__main__':
    unittest.main()
