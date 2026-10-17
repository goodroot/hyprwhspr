"""Isolated coverage of the opt-in Hyprland ydotool backend."""

import subprocess
import threading
import types
import unittest
from unittest import mock

from tests.text_injector_helpers import ConfigStub, make_injector
from text_injector import InjectionOutcome, TextInjector


class HyprlandPasteBackendTests(unittest.TestCase):
    def setUp(self):
        self.injector = make_injector()
        self.injector.wtype_available = True
        self.injector.config_manager = ConfigStub({
            'hyprland_paste_backend': 'ydotool',
            'auto_submit': True,
            'append_trailing_space': False,
            'clipboard_settle_delay': 0,
        })
        self.mocks = {}
        patches = {
            '_is_hyprland_session': True,
            '_is_gnome_wayland_session': False,
            '_get_active_window_info': {'class': 'editor'},
            '_save_clipboard': b'old',
            '_copy_text_to_clipboard': True,
            '_restore_clipboard': None,
            '_send_shortcut_hyprland': True,
            '_send_paste_keys_wtype': True,
            '_run_ydotool': types.SimpleNamespace(returncode=0, stderr=b''),
        }
        for name, result in patches.items():
            self.mocks[name] = self.enterContext(mock.patch.object(
                self.injector, name, return_value=result))
        self.enterContext(mock.patch('text_injector.time.sleep'))
        self.log = self.enterContext(mock.patch('text_injector.log'))

    def assert_no_fallback(self):
        self.mocks['_send_shortcut_hyprland'].assert_not_called()
        self.mocks['_send_paste_keys_wtype'].assert_not_called()

    def test_paste_and_submit_use_only_ydotool(self):
        self.assertEqual(self.injector.inject_text('hello'), InjectionOutcome.INJECTED)
        self.assertEqual(self.mocks['_run_ydotool'].call_args_list, [
            mock.call(['key', '29:1'], timeout=1),
            mock.call(['key', '47:1', '47:0'], timeout=1),
            mock.call(['key', '29:0'], timeout=1),
            mock.call(['key', '28:1', '28:0'], timeout=1),
        ])
        self.assert_no_fallback()
        self.mocks['_restore_clipboard'].assert_called_once()

    def test_failures_leave_text_without_fallback_or_submit(self):
        for failure in ('missing', 'daemon', 'command', 'timeout', 'unsupported'):
            with self.subTest(failure=failure):
                self.injector.ydotool_available = failure != 'missing'
                self.injector.config_manager.settings['paste_mode'] = 'ctrl+unknown' if failure == 'unsupported' else 'ctrl'
                run = self.mocks['_run_ydotool']
                run.reset_mock()
                run.return_value = None if failure == 'daemon' else types.SimpleNamespace(returncode=1, stderr=b'failed')
                run.side_effect = subprocess.TimeoutExpired('ydotool', 1) if failure == 'timeout' else None
                # Even with no installed tools, this must be FAILED, not clipboard-only success.
                self.injector.wtype_available = False
                self.assertEqual(self.injector.inject_text('hello'), InjectionOutcome.FAILED)
                self.assertEqual(self.injector._last_text, 'hello')
                self.mocks['_copy_text_to_clipboard'].assert_called_with('hello')
                self.mocks['_restore_clipboard'].assert_not_called()
                self.assert_no_fallback()
                self.assertFalse(any('28:1' in call.args[0] for call in run.call_args_list))

    def test_auto_submit_failure_never_retries(self):
        for available, result in ((False, None), (True, None),
                                  (True, types.SimpleNamespace(returncode=1, stderr=b'failed'))):
            with self.subTest(available=available, result=result):
                self.injector.ydotool_available = available
                self.mocks['_run_ydotool'].reset_mock()
                self.mocks['_run_ydotool'].return_value = result
                self.injector._send_enter_if_auto_submit()
                self.assertEqual(self.mocks['_run_ydotool'].call_count, int(available))
                self.assert_no_fallback()
                self.assertTrue(any('Enter failed' in call.args[0] for call in self.log.call_args_list))

    def test_cancelled_submit_sends_nothing(self):
        cancelled = threading.Event()
        cancelled.set()
        self.injector._send_enter_if_auto_submit(cancelled)
        self.mocks['_run_ydotool'].assert_not_called()
        self.assert_no_fallback()

    def test_custom_chord_and_modifier_override(self):
        self.injector.config_manager.settings.update({
            'applications': {'editor': {'auto_paste': 'ctrl+y'}},
            'ydotool_modifier_overrides': {'ctrl': 'capslock'},
            'auto_submit': False,
        })
        self.assertTrue(self.injector._inject_via_clipboard_and_hotkey('hello'))
        self.assertEqual(self.mocks['_run_ydotool'].call_args_list, [
            mock.call(['key', '58:1'], timeout=1),
            mock.call(['key', '21:1', '21:0'], timeout=1),
            mock.call(['key', '58:0'], timeout=1),
        ])
        self.assert_no_fallback()

    def test_paste_keycode_override(self):
        self.injector.config_manager.settings.update({'paste_keycode_wev': 60, 'auto_submit': False})
        self.assertTrue(self.injector._inject_via_clipboard_and_hotkey('hello'))
        self.mocks['_run_ydotool'].assert_any_call(['key', '52:1', '52:0'], timeout=1)
        self.assert_no_fallback()

    def test_excluded_application_is_untouched(self):
        self.injector.config_manager.settings['applications'] = {'editor': {'auto_paste': False}}
        self.assertTrue(self.injector._inject_via_clipboard_and_hotkey('hello'))
        self.mocks['_copy_text_to_clipboard'].assert_not_called()
        self.mocks['_run_ydotool'].assert_not_called()
        self.assert_no_fallback()

    def test_replay_uses_selected_backend_without_enter(self):
        self.injector._last_text = 'hello'
        self.assertTrue(self.injector.recover_last('paste_last')[0])
        self.assertEqual(self.mocks['_run_ydotool'].call_count, 3)
        self.assert_no_fallback()

    def test_stream_failure_preserves_clipboard_and_skips_submit(self):
        session = self.injector.begin_stream()
        self.assertEqual(self.injector.inject_stream_chunk(session, 'hello', final=True), InjectionOutcome.INJECTED)
        self.mocks['_run_ydotool'].return_value = None
        self.assertEqual(self.injector.inject_stream_chunk(session, 'world', final=True), InjectionOutcome.FAILED)
        self.assertEqual(self.injector.end_stream(session), InjectionOutcome.FAILED)
        self.assertEqual(self.injector._last_text, 'hello world')
        self.mocks['_restore_clipboard'].assert_not_called()
        self.assertFalse(any('28:1' in call.args[0] for call in self.mocks['_run_ydotool'].call_args_list))
        self.assert_no_fallback()

    def test_stream_cancel_during_clipboard_settle_sends_no_keys(self):
        session = self.injector.begin_stream()
        self.mocks['_copy_text_to_clipboard'].side_effect = lambda text: (session.cancelled.set() or True)
        self.assertEqual(self.injector.inject_stream_chunk(session, 'hello', final=True), InjectionOutcome.FAILED)
        self.injector.end_stream(session, submit=False)
        self.mocks['_run_ydotool'].assert_not_called()
        self.assert_no_fallback()

    def test_default_and_invalid_values_use_native(self):
        for setting in ('auto', None, 'typo', []):
            with self.subTest(setting=setting):
                self.injector.config_manager.settings['hyprland_paste_backend'] = setting
                self.assertTrue(self.injector._inject_via_clipboard_and_hotkey('hello'))
                self.mocks['_send_shortcut_hyprland'].assert_called_with('enter')
                self.mocks['_run_ydotool'].assert_not_called()
        warnings = [call for call in self.log.call_args_list if 'Invalid hyprland_paste_backend' in call.args[0]]
        self.assertEqual(len(warnings), 1)

    def test_other_desktops_ignore_selection(self):
        self.mocks['_is_hyprland_session'].return_value = False
        self.injector.config_manager.settings['auto_submit'] = False
        self.assertTrue(self.injector._inject_via_clipboard_and_hotkey('hello'))
        self.mocks['_send_paste_keys_wtype'].assert_called_once_with('ctrl+v')
        self.mocks['_run_ydotool'].assert_not_called()


class HyprlandPasteStartupTests(unittest.TestCase):
    def test_missing_selected_backend_warns_once_at_init(self):
        for hyprland, backend, available in (
            (True, 'ydotool', False), (True, 'ydotool', True),
            (True, 'auto', False), (False, 'ydotool', False),
        ):
            with self.subTest(hyprland=hyprland, backend=backend, available=available):
                with (
                    mock.patch.object(TextInjector, '_check_ydotool', return_value=available),
                    mock.patch.object(TextInjector, '_is_hyprland_session', return_value=hyprland),
                    mock.patch.object(TextInjector, '_is_x11_session', return_value=False),
                    mock.patch('text_injector.shutil.which', return_value='/usr/bin/wtype'),
                    mock.patch('text_injector.YdotooldSession') as daemon,
                    mock.patch('text_injector.log') as log,
                ):
                    TextInjector(ConfigStub({'hyprland_paste_backend': backend}))
                warnings = [call for call in log.call_args_list if 'requires both ydotool and ydotoold' in call.args[0]]
                self.assertEqual(len(warnings), int(hyprland and backend == 'ydotool' and not available))
                daemon.return_value.ensure_running.assert_not_called()
