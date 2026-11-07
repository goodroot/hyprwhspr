"""Regression tests for Sway focus detection and XWayland paste dispatch.

All I/O is mocked: no audio, clipboard, keyboard or service changes.
"""
import json
import os
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib' / 'src'))
import text_injector as patched_injector


class Wire:
    def __init__(self, data): self.data = data
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def settimeout(self, value): pass
    def connect(self, path): pass
    def sendall(self, data): self.sent = data
    def recv(self, count):
        part, self.data = self.data[:min(count, 7)], self.data[min(count, 7):]
        return part


class FocusTests(unittest.TestCase):
    module = patched_injector

    def setUp(self):
        self.env = patch.dict(os.environ, SWAYSOCK='/fake/sway.sock')
        self.env.start(); self.addCleanup(self.env.stop)
        self.injector = self.module.TextInjector.__new__(self.module.TextInjector)
        self.injector.config_manager = None
        self.injector.wtype_available = True
        self.injector.ydotool_available = True
        self.injector.xdotool_available = True
        self.injector._is_hyprland_session = Mock(return_value=False)
        self.injector._is_x11_session = Mock(return_value=False)
        self.injector._is_gnome_wayland_session = Mock(return_value=False)
        self.logs = patch.object(self.module, 'log')
        self.logs.start(); self.addCleanup(self.logs.stop)

    def read_tree(self, tree):
        data = json.dumps(tree).encode()
        wire = Wire(b'i3-ipc'+struct.pack('<II', len(data), 4)+data)
        with patch.object(self.module.socket, 'socket', return_value=wire):
            result = self.injector._get_active_window_info()
        self.assertEqual(wire.sent, b'i3-ipc'+struct.pack('<II', 0, 4))
        return result

    def window(self, app='kitty', **extra):
        return dict(id=42, focused=True, app_id=app, name='Synthetic title', **extra)

    def test_kitty(self):
        info = self.read_tree({'nodes':[self.window()]})
        self.assertEqual(info['class'], 'kitty')
        self.assertEqual(self.injector._resolve_paste_chord(info)[0], 'ctrl+shift+v')

    def test_firefox(self):
        info = self.read_tree({'floating_nodes':[self.window('firefox')]})
        self.assertEqual(info['source'], 'sway')
        self.assertFalse(info['xwayland'])
        self.assertEqual(self.injector._resolve_paste_chord(info)[0], 'ctrl+v')

    def test_chatgpt(self):
        info = self.read_tree({'nodes':[self.window(None, window=123, window_properties={'class':'Chatgpt'})]})
        self.assertEqual(info['class'], 'Chatgpt')
        self.assertTrue(info['xwayland'])
        self.assertIn('chatgpt', self.injector.focused_window_identifiers(info))
        self.assertEqual(self.injector._resolve_paste_chord(info)[0], 'ctrl+v')

    def test_stale_xwayland_not_selected(self):
        info = self.read_tree({'nodes':[dict(self.window(None,window_properties={'class':'Chatgpt'}),focused=False), self.window()]})
        self.assertEqual(info['class'], 'kitty')

    def test_missing_socket_no_fallback(self):
        with patch.object(self.module.subprocess, 'run') as run:
            self.assertIsNone(self.injector._get_active_window_info())
            run.assert_not_called()

    def test_wire_errors(self):
        for data in [b'', b'i3-ipc'+struct.pack('<II', 9*1024*1024, 4),
                     b'bad-ip'+struct.pack('<II', 0, 4),
                     b'i3-ipc'+struct.pack('<II', 0, 2),
                     b'i3-ipc'+struct.pack('<II', 1, 4)+b'{']:
            with self.subTest(data=data[:14]), patch.object(self.module.socket,'socket',return_value=Wire(data)):
                self.assertIsNone(self.injector._get_sway_window_info())

    def test_timeout(self):
        with patch.object(self.module.socket,'socket',side_effect=TimeoutError):
            self.assertIsNone(self.injector._get_sway_window_info())

    def test_other_compositor_unchanged(self):
        with patch.dict(os.environ), patch.object(self.module.subprocess,'run',return_value=Mock(returncode=0,stdout='{"app_id":"kitty"}')):
            os.environ.pop('SWAYSOCK',None)
            self.assertEqual(self.injector._get_active_window_info()['source'],'niri')

    def paste(self, current, expected=None, xfocus='123', xresult=True):
        i = self.injector
        i._save_clipboard = Mock(return_value=None)
        i._copy_text_to_clipboard = Mock(return_value=True)
        i._get_sway_window_info = Mock(return_value=current)
        i._send_paste_keys_wtype = Mock(return_value=True)
        i._send_paste_keys_xdotool = Mock(return_value=xresult)
        i._send_paste_keys_slow = Mock(return_value=True)
        i._restore_clipboard = Mock()
        i._clear_stuck_modifiers = Mock()
        i._send_enter_if_auto_submit = Mock()
        with patch.object(self.module.time,'sleep'), patch.object(self.module.subprocess,'run',return_value=Mock(returncode=0,stdout=xfocus)):
            return i._paste_via_clipboard('Synthetic fixture only', 'ctrl+v', False, False, expected_window=expected or current)

    def info(self, xwayland=False):
        return dict(sway_id=42, app_id=None if xwayland else 'firefox',
                    **{'class':'Chatgpt' if xwayland else 'firefox'},
                    xwayland=xwayland, x11_window_id=123 if xwayland else None)

    def test_xwayland_transport(self):
        self.assertTrue(self.paste(self.info(True)))
        self.injector._send_paste_keys_xdotool.assert_called_once_with('ctrl+v')
        self.injector._send_paste_keys_wtype.assert_not_called()

    def test_firefox_transport(self):
        self.assertTrue(self.paste(self.info()))
        self.injector._send_paste_keys_wtype.assert_called_once_with('ctrl+v')
        self.injector._send_paste_keys_xdotool.assert_not_called()

    def test_focus_changed_before_dispatch(self):
        self.assertFalse(self.paste(dict(self.info(), sway_id=99), self.info()))
        self.injector._send_paste_keys_wtype.assert_not_called()
        self.injector._send_paste_keys_xdotool.assert_not_called()

    def test_ipc_lost_before_dispatch(self):
        self.assertFalse(self.paste(None, self.info()))
        self.injector._send_paste_keys_wtype.assert_not_called()

    def test_x_focus_mismatch(self):
        self.assertFalse(self.paste(self.info(True), xfocus='456'))
        self.injector._send_paste_keys_xdotool.assert_not_called()

    def test_xwayland_failure_no_retry(self):
        self.assertFalse(self.paste(self.info(True), xresult=False))
        self.injector._send_paste_keys_wtype.assert_not_called()
        self.injector._send_paste_keys_slow.assert_not_called()

    def test_failure_before_clipboard(self):
        i = self.injector
        i._get_active_window_info = Mock(return_value=None)
        i._copy_text_to_clipboard = Mock()
        self.assertFalse(i._inject_via_clipboard_and_hotkey('Synthetic fixture'))
        i._copy_text_to_clipboard.assert_not_called()

    def test_per_application_override(self):
        self.injector.config_manager = Mock()
        self.injector.config_manager.get_setting.side_effect = lambda key, default=None: {'applications':{'firefox':{'auto_paste':'ctrl+y'}}}.get(key,default)
        self.assertEqual(self.injector._resolve_paste_chord({'class':'firefox'}), ('ctrl+y', 'firefox'))


if __name__ == '__main__': unittest.main()
