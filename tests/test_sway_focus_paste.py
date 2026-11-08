"""Regression tests for Sway focus detection and XWayland paste dispatch.

All I/O is mocked: no audio, clipboard, keyboard or service changes.
"""
import json
import os
from pathlib import Path
import struct
import sys
import threading
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
        self.env = patch.dict(os.environ, {'SWAYSOCK': '/fake/sway.sock'}, clear=True)
        self.env.start(); self.addCleanup(self.env.stop)
        self.injector = self.module.TextInjector.__new__(self.module.TextInjector)
        self.injector._sway_enabled = True
        self.injector._sway_socket = '/fake/sway.sock'
        self.injector._sway_runtime_dir = '/nonexistent'
        self.injector.sway_xdotool_available = True
        self.injector.config_manager = None
        self.injector._clipboard_lock = threading.RLock()
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
        wire = Wire(b'i3-ipc'+struct.pack('=II', len(data), 4)+data)
        with patch.object(self.module.socket, 'socket', return_value=wire):
            result = self.injector._get_active_window_info()
        self.assertEqual(wire.sent, b'i3-ipc'+struct.pack('=II', 0, 4))
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
        for data in [b'', b'i3-ipc'+struct.pack('=II', 9*1024*1024, 4),
                     b'bad-ip'+struct.pack('=II', 0, 4),
                     b'i3-ipc'+struct.pack('=II', 0, 2),
                     b'i3-ipc'+struct.pack('=II', 1, 4)+b'{']:
            with self.subTest(data=data[:14]), patch.object(self.module.socket,'socket',return_value=Wire(data)):
                self.assertIsNone(self.injector._get_sway_window_info())

    def test_timeout(self):
        with patch.object(self.module.socket,'socket',side_effect=TimeoutError):
            self.assertIsNone(self.injector._get_sway_window_info())

    def test_other_compositor_unchanged(self):
        with patch.dict(os.environ), patch.object(self.module.subprocess,'run',return_value=Mock(returncode=0,stdout='{"app_id":"kitty"}')):
            self.injector._sway_enabled = False
            self.assertEqual(self.injector._get_active_window_info()['source'],'niri')

    def paste(self, current, expected=None, xfocus='123', xresult=True):
        i = self.injector
        i._save_clipboard = Mock(return_value=None)
        i._copy_text_to_clipboard = Mock(return_value=True)
        i._sway_request = Mock(return_value=[{'name': 'seat0', 'focus': current['sway_id']}] if current else None)
        i._send_paste_keys_wtype = Mock(return_value=True)
        i._send_paste_keys_xdotool = Mock(return_value=xresult)
        i._send_paste_keys_slow = Mock(return_value=True)
        i._restore_clipboard = Mock()
        i._clear_stuck_modifiers = Mock()
        i._send_enter_if_auto_submit = Mock()
        with patch.object(self.module.time,'sleep'), patch.object(self.module.subprocess,'run',return_value=Mock(returncode=0,stdout=xfocus)):
            return i._paste_via_clipboard('Synthetic fixture only', 'ctrl+v', False, False, expected_window=expected or current)

    def info(self, xwayland=False):
        return dict(source='sway', sway_socket='/fake/sway.sock', sway_id=42, app_id=None if xwayland else 'firefox',
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


    def test_views_without_app_identity_and_empty_workspace(self):
        for node, xwayland in (
            (self.window(None, pid=123, shell='xdg_shell'), False),
            (self.window(None, pid=123, shell='xwayland', window=55), True),
        ):
            with self.subTest(node=node):
                info = self.read_tree({'nodes': [node]})
                self.assertEqual(info['class'], '')
                self.assertEqual(info['xwayland'], xwayland)
                self.assertEqual(self.injector._resolve_paste_chord(info)[0], 'ctrl+v')
        self.assertIsNone(self.read_tree({'id': 42, 'type': 'workspace', 'focused': True}))

    def test_missing_xdotool_and_missing_window_id_are_logged(self):
        self.injector.ydotool_available = False
        for available, window_id, message in (
            (False, 123, 'needs xdotool or ydotool'),
            (True, None, 'window ID unknown'),
        ):
            with self.subTest(available=available):
                self.injector.sway_xdotool_available = available
                info = dict(self.info(True), x11_window_id=window_id)
                with patch.object(self.module, 'log') as log:
                    self.assertFalse(self.paste(info))
                    self.assertTrue(any(message in c.args[0] for c in log.call_args_list))
                self.injector._send_paste_keys_wtype.assert_not_called()
                self.injector._send_paste_keys_xdotool.assert_not_called()

    def test_xwayland_without_xdotool_pastes_with_ydotool(self):
        i = self.injector
        i.sway_xdotool_available = False
        self.assertTrue(self.paste(self.info(True)))
        i._send_paste_keys_slow.assert_called_once_with('ctrl+v')
        i._send_paste_keys_xdotool.assert_not_called()
        i._send_paste_keys_wtype.assert_not_called()

    def test_xwayland_ydotool_failure_no_retry(self):
        i = self.injector
        i.sway_xdotool_available = False
        i._send_paste_keys_slow = Mock(return_value=False)
        i._save_clipboard = Mock(return_value=None)
        i._copy_text_to_clipboard = Mock(return_value=True)
        i._sway_request = Mock(return_value=[{'name': 'seat0', 'focus': 42}])
        i._send_paste_keys_wtype = Mock(return_value=True)
        i._clear_stuck_modifiers = Mock()
        with patch.object(self.module.time, 'sleep'):
            self.assertFalse(i._paste_via_clipboard('Synthetic', 'ctrl+v', False, False, expected_window=self.info(True)))
        i._send_paste_keys_slow.assert_called_once()
        i._send_paste_keys_wtype.assert_not_called()

    def test_refused_paste_withdraws_text_from_clipboard(self):
        i = self.injector
        saved = b'previous clipboard'
        i._save_clipboard = Mock(return_value=saved)
        i._copy_text_to_clipboard = Mock(return_value=True)
        i._sway_request = Mock(return_value=[{'name': 'seat0', 'focus': 99}])
        i._restore_clipboard = Mock()
        with patch.object(self.module.time, 'sleep'):
            self.assertFalse(i._paste_via_clipboard('Synthetic', 'ctrl+v', False, False, expected_window=self.info()))
        i._restore_clipboard.assert_called_once_with(saved, injected=b'Synthetic', delay=0)

    def test_refused_paste_clears_clipboard_that_was_empty(self):
        i = self.injector
        for current, cleared in ((b'Synthetic', True), (b'copied since', False)):
            with self.subTest(current=current):
                i._save_clipboard = Mock(return_value=current)
                with patch.object(self.module.subprocess, 'run') as run:
                    i._withdraw_undelivered(None, 'Synthetic')
                self.assertEqual(run.called, cleared)
                if cleared:
                    self.assertEqual(run.call_args.args[0], ['wl-copy', '--clear'])

    def test_refused_stream_chunk_restores_stream_clipboard(self):
        i = self.injector
        stream = self.module._StreamSession()
        stream.saved_clipboard = b'before dictation'
        i._copy_text_to_clipboard = Mock(return_value=True)
        i._sway_request = Mock(return_value=None)
        i._restore_clipboard = Mock()
        with patch.object(self.module.time, 'sleep'):
            self.assertFalse(i._paste_via_clipboard('chunk', 'ctrl+v', False, False, stream=stream, expected_window=self.info()))
        i._restore_clipboard.assert_called_once_with(b'before dictation', injected=b'chunk', delay=0)

    def test_auto_submit_xwayland_without_xdotool_uses_ydotool(self):
        from tests.text_injector_helpers import ConfigStub
        i = self.injector
        i.config_manager = ConfigStub({'auto_submit': True})
        i.sway_xdotool_available = False
        i._run_ydotool = Mock(return_value=Mock(returncode=0))
        with patch.object(i, '_sway_request', return_value=[{'name': 'seat0', 'focus': 42}]), patch.object(self.module.subprocess, 'run') as run:
            i._send_enter_if_auto_submit(expected_window=self.info(True))
        i._run_ydotool.assert_called_once_with(['key', '28:1', '28:0'], timeout=1)
        run.assert_not_called()

    def test_named_seat_focus_matches(self):
        with patch.object(self.injector, '_sway_request', return_value=[{'name': 'main', 'focus': 42}]):
            self.assertTrue(self.injector._sway_focus_matches(self.info()))

    def test_native_clipboard_only_ignores_xwayland_tool(self):
        i = self.injector
        i.xdotool_available = i.wtype_available = i.ydotool_available = False
        i.sway_xdotool_available = True
        self.assertTrue(self.paste(self.info()))
        i._restore_clipboard.assert_not_called()
        i._send_enter_if_auto_submit.assert_not_called()
        i._send_paste_keys_xdotool.assert_not_called()

    def test_native_chord_validation_ignores_xwayland_tool(self):
        i = self.injector
        i.xdotool_available = i.ydotool_available = False
        self.assertFalse(i._chord_is_usable('ctrl+XF86AudioPlay', self.info()))
        self.assertTrue(i._chord_is_usable('ctrl+XF86AudioPlay', self.info(True)))

    def test_seat_focus_check_uses_small_reply_and_socket_identity(self):
        with patch.object(self.injector, '_sway_request', return_value=[{'name': 'seat0', 'focus': 42}]) as request:
            self.assertTrue(self.injector._sway_focus_matches(self.info()))
            request.assert_called_once_with(101)
            self.assertFalse(self.injector._sway_focus_matches(dict(self.info(), sway_socket='/old')))
            self.assertFalse(self.injector._sway_focus_matches(dict(self.info(), sway_id=43)))

    def test_socket_rediscovery_on_restart(self):
        data = json.dumps({'nodes': [self.window()]}).encode()
        wire = Wire(b'i3-ipc' + struct.pack('=II', len(data), 4) + data)
        wire.connect = Mock(side_effect=[FileNotFoundError(), None])
        with patch.object(self.module.socket, 'socket', return_value=wire), patch.object(self.injector, '_rediscover_sway_socket', return_value='/new/sway.sock'):
            info = self.injector._get_sway_window_info()
        self.assertEqual(info['sway_socket'], '/new/sway.sock')
        self.assertEqual(wire.connect.call_args_list[-1].args, ('/new/sway.sock',))

    def test_auto_submit_xwayland_uses_xdotool_return(self):
        from tests.text_injector_helpers import ConfigStub
        i = self.injector
        i.config_manager = ConfigStub({'auto_submit': True})
        i.ydotool_available = False
        with patch.object(i, '_sway_request', return_value=[{'name': 'seat0', 'focus': 42}]), patch.object(self.module.subprocess, 'run', return_value=Mock(returncode=0, stdout='123')) as run:
            i._send_enter_if_auto_submit(expected_window=self.info(True))
        self.assertEqual(run.call_args_list[-1].args[0], ['xdotool', 'key', '--clearmodifiers', 'Return'])
        self.assertFalse(any(c.args[0][0] == 'wtype' for c in run.call_args_list))

    def test_auto_submit_does_not_follow_changed_focus(self):
        from tests.text_injector_helpers import ConfigStub
        i = self.injector
        i.config_manager = ConfigStub({'auto_submit': True})
        with patch.object(i, '_sway_request', return_value=[{'name': 'seat0', 'focus': 99}]), patch.object(self.module.subprocess, 'run') as run:
            i._send_enter_if_auto_submit(expected_window=self.info(True))
        run.assert_not_called()


    def test_delivery_keeps_target_for_auto_submit_without_second_tree(self):
        from tests.text_injector_helpers import make_injector, ConfigStub
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                i = make_injector()
                i._sway_enabled = True
                i._sway_socket = '/fake/sway.sock'
                i.sway_xdotool_available = True
                i.config_manager = ConfigStub({'auto_submit': True, 'append_trailing_space': False})
                tree = {'nodes': [dict(id=42, focused=True, shell='xwayland', window=123,
                                      window_properties={'class': 'Chatgpt'})]}
                def request(kind):
                    return tree if kind == 4 else [{'name': 'seat0', 'focus': 42}]
                with (
                    patch.object(i, '_sway_request', side_effect=request) as ipc,
                    patch.object(i, '_save_clipboard', return_value=None),
                    patch.object(i, '_copy_text_to_clipboard', return_value=True),
                    patch.object(i, '_restore_clipboard'),
                    patch.object(i, '_send_paste_keys_xdotool', return_value=True) as send,
                    patch.object(patched_injector.subprocess, 'run', return_value=Mock(returncode=0, stdout='123')),
                    patch.object(patched_injector.time, 'sleep'),
                ):
                    if streaming:
                        session = i.begin_stream()
                        i.inject_stream_chunk(session, 'hello', final=True)
                        result = i.end_stream(session)
                    else:
                        result = i.inject_text('hello')
                    self.assertEqual(result, patched_injector.InjectionOutcome.INJECTED)
                    self.assertEqual([c.args[0] for c in send.call_args_list], ['ctrl+v', 'enter'])
                    self.assertEqual([c.args[0] for c in ipc.call_args_list], [4, 101, 101])

    def test_x11_focus_errors_are_logged_without_dispatch(self):
        i = self.injector
        cases = [Mock(returncode=1, stdout=''), Mock(returncode=0, stdout='bad'),
                 OSError('unavailable'), patched_injector.subprocess.TimeoutExpired('xdotool', 0.5)]
        for result in cases:
            with self.subTest(result=result), patch.object(patched_injector.subprocess, 'run', side_effect=result if isinstance(result, Exception) else None, return_value=result), patch.object(patched_injector, 'log') as log:
                self.assertFalse(i._sway_x11_focus_matches(self.info(True)))
                log.assert_called_once()


class SessionTests(unittest.TestCase):
    def test_other_desktops_ignore_stale_socket(self):
        for desktop in ('niri', 'Hyprland', 'GNOME', 'KDE'):
            with self.subTest(desktop=desktop), patch.dict(os.environ, {
                'XDG_CURRENT_DESKTOP': desktop, 'SWAYSOCK': '/old/socket',
                'XDG_SESSION_TYPE': 'wayland'}, clear=True):
                self.assertFalse(patched_injector.TextInjector.sway_session())

    def test_session_is_captured_and_tools_are_separate(self):
        with patch.dict(os.environ, {'XDG_CURRENT_DESKTOP': 'sway', 'SWAYSOCK': '/fake/socket'}, clear=True), patch.object(patched_injector.TextInjector, '_check_ydotool', return_value=False), patch.object(patched_injector.shutil, 'which', side_effect=lambda name: '/usr/bin/xdotool' if name == 'xdotool' else None), patch.object(patched_injector, 'YdotooldSession'), patch.object(patched_injector, 'log') as log:
            i = patched_injector.TextInjector()
            self.assertTrue(i._sway_enabled)
            self.assertTrue(i.sway_xdotool_available)
            self.assertFalse(i.xdotool_available)
            self.assertTrue(any('No injection backend found' in c.args[0] for c in log.call_args_list))
            os.environ['SWAYSOCK'] = '/changed'
            self.assertEqual(i._sway_socket, '/fake/socket')

    def test_gnome_token_on_sway_is_not_mutter(self):
        injector = patched_injector.TextInjector.__new__(patched_injector.TextInjector)
        injector._sway_enabled = True
        env = {'XDG_SESSION_TYPE': 'wayland', 'XDG_CURRENT_DESKTOP': 'sway:GNOME', 'SWAYSOCK': '/fake'}
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(injector._is_gnome_wayland_session())
            injector._sway_enabled = False
            self.assertTrue(injector._is_gnome_wayland_session())

    def test_rediscovery_refuses_ambiguous_instances(self):
        from tests.text_injector_helpers import make_injector
        i = make_injector()
        i._sway_runtime_dir = '/fake'
        for count in (0, 1, 2):
            paths = [Mock(name=f'socket{n}') for n in range(count)]
            for n, path in enumerate(paths):
                path.name = f'sway-ipc.1000.{n + 1}.sock'
                path.stat.return_value.st_mode = patched_injector.stat.S_IFSOCK
                path.__str__ = Mock(return_value=f'/fake/{path.name}')
            with patch.object(patched_injector.Path, 'glob', return_value=paths), patch.object(patched_injector.Path, 'exists', return_value=True):
                result = i._rediscover_sway_socket()
            self.assertEqual(result, str(paths[0]) if count == 1 else None)

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        from tests.text_injector_helpers import make_injector
        self.injector = make_injector()
        self.injector._sway_enabled = True
        self.injector._sway_socket = '/fake/sway.sock'
        self.injector.sway_xdotool_available = True
        for patcher in (
            patch.dict(os.environ, {'SWAYSOCK': '/fake/sway.sock'}, clear=True),
            patch.object(self.injector, '_get_active_window_info', return_value=None),
            patch.object(patched_injector, 'log'),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_unknown_focus_retention_respects_deny_rules(self):
        for applications, allowed in (
            ({}, True),
            ({'editor': {'auto_paste': 'ctrl+y'}}, True),
            ({'private': {'auto_paste': False}}, False),
        ):
            for previous in (None, 'older dictation'):
                with self.subTest(applications=applications, previous=previous):
                    i = self.injector
                    i.config_manager.settings['applications'] = applications
                    i._last_text = previous
                    with patch.object(i, '_copy_text_to_clipboard') as copy, patch.object(i, '_paste_via_clipboard') as paste:
                        self.assertEqual(i.inject_text('new dictation'), patched_injector.InjectionOutcome.FAILED)
                        self.assertEqual(i._last_text, 'new dictation ' if allowed else previous)
                        copy.assert_not_called()
                        paste.assert_not_called()

    def test_unknown_focus_stream_finalization_respects_deny_rules(self):
        for denied in (False, True):
            with self.subTest(denied=denied):
                i = self.injector
                i.config_manager.settings['applications'] = {'private': {'auto_paste': False}} if denied else {}
                i._last_text = 'older dictation'
                session = i.begin_stream()
                session.recovery_text = 'complete transcript'
                with patch.object(i, '_copy_text_to_clipboard') as copy, patch.object(i, '_paste_via_clipboard') as paste:
                    self.assertEqual(i.inject_stream_chunk(session, 'new dictation', final=True), patched_injector.InjectionOutcome.FAILED)
                    self.assertEqual(i.end_stream(session), patched_injector.InjectionOutcome.FAILED)
                    self.assertEqual(i._last_text, 'older dictation' if denied else 'complete transcript')
                    copy.assert_not_called()
                    paste.assert_not_called()

    def test_failed_repaste_does_not_undo_concurrent_clear(self):
        i = self.injector
        i._last_text = 'older dictation'
        def clear_during_lookup():
            i.recover_last('clear_last')
            return None
        i._get_active_window_info.side_effect = clear_during_lookup
        self.assertFalse(i.recover_last('paste_last')[0])
        self.assertIsNone(i._last_text)

    def test_cancel_during_delivery_focus_checks_does_not_dispatch(self):
        for xwayland in (False, True):
            with self.subTest(xwayland=xwayland):
                i = self.injector
                session = i.begin_stream()
                info = dict(source='sway', sway_socket='/fake/sway.sock', sway_id=42, app_id=None if xwayland else 'firefox',
                            xwayland=xwayland, x11_window_id=123 if xwayland else None)
                def focus(kind):
                    if not xwayland:
                        session.cancelled.set()
                    return [{'name': 'seat0', 'focus': 42}]
                def xfocus(*args, **kwargs):
                    session.cancelled.set()
                    return Mock(returncode=0, stdout='123')
                with (
                    patch.object(i, '_save_clipboard', return_value=None),
                    patch.object(i, '_copy_text_to_clipboard', return_value=True),
                    patch.object(i, '_sway_request', side_effect=focus),
                    patch.object(patched_injector.subprocess, 'run', side_effect=xfocus),
                    patch.object(patched_injector.time, 'sleep'),
                    patch.object(i, '_send_paste_keys_wtype') as wtype,
                    patch.object(i, '_send_paste_keys_xdotool') as xdotool,
                    patch.object(i, '_send_paste_keys_slow') as ydotool,
                ):
                    i.wtype_available = i.xdotool_available = True
                    self.assertFalse(i._paste_via_clipboard('new', 'ctrl+v', False, False, stream=session, expected_window=info))
                    wtype.assert_not_called()
                    xdotool.assert_not_called()
                    ydotool.assert_not_called()
                i.release_stream(session)


if __name__ == '__main__': unittest.main()
