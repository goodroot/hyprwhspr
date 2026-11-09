"""Sway IPC focus lookup: wire format, tree walk, seat check, rediscovery.

Socket I/O is faked except for the liveness probe, which binds real Unix
sockets in a temporary directory. No Sway, clipboard or input devices.
"""
import json
import os
from pathlib import Path
import socket
import struct
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib' / 'src'))
import sway_focus
from sway_focus import GET_SEATS, GET_TREE, IPC_UNAVAILABLE, NO_VIEW, SwayIPC, is_sway_session


class Wire:
    """A Sway socket that replies with fixed bytes, a few at a time."""

    def __init__(self, data):
        self.data = data
        self.sent = None
        self.connected = []

    def __enter__(self): return self
    def __exit__(self, *args): pass
    def settimeout(self, value): pass
    def connect(self, path): self.connected.append(path)
    def sendall(self, data): self.sent = data

    def recv(self, count):
        part, self.data = self.data[:min(count, 7)], self.data[min(count, 7):]
        return part


def reply(payload, kind=GET_TREE):
    data = json.dumps(payload).encode()
    return b'i3-ipc' + struct.pack('=II', len(data), kind) + data


def view(app='kitty', **extra):
    return dict(id=42, focused=True, app_id=app, name='Synthetic title', **extra)


class LookupTests(unittest.TestCase):
    def setUp(self):
        self.ipc = SwayIPC('/fake/sway.sock', '/nonexistent', uid=1000)

    def lookup(self, tree):
        wire = Wire(reply(tree))
        with patch.object(sway_focus.socket, 'socket', return_value=wire):
            result = self.ipc.lookup()
        self.assertEqual(wire.sent, b'i3-ipc' + struct.pack('=II', 0, GET_TREE))
        return result

    def test_native_view(self):
        info, reason = self.lookup({'floating_nodes': [view('firefox')]})
        self.assertIsNone(reason)
        self.assertEqual(info, {
            'class': 'firefox', 'app_id': 'firefox', 'title': 'Synthetic title',
            'source': 'sway', 'sway_id': 42, 'sway_socket': '/fake/sway.sock',
            'xwayland': False, 'x11_window_id': None,
        })

    def test_xwayland_view(self):
        info, _ = self.lookup({'nodes': [view(None, window=123, window_properties={'class': 'Chatgpt'})]})
        self.assertEqual((info['class'], info['xwayland'], info['x11_window_id']), ('Chatgpt', True, 123))

    def test_unfocused_xwayland_view_is_not_selected(self):
        stale = dict(view(None, window_properties={'class': 'Chatgpt'}), focused=False)
        info, _ = self.lookup({'nodes': [stale, view()]})
        self.assertEqual(info['class'], 'kitty')

    def test_views_without_app_identity(self):
        for node, xwayland in (
            (view(None, pid=123, shell='xdg_shell'), False),
            (view(None, pid=123, shell='xwayland', window=55), True),
        ):
            with self.subTest(node=node):
                info, _ = self.lookup({'nodes': [node]})
                self.assertEqual((info['class'], info['xwayland']), ('', xwayland))

    def test_focused_workspace_or_container_is_no_view(self):
        container = {'id': 7, 'type': 'con', 'focused': True, 'nodes': [dict(view(), focused=False)]}
        for tree in ({'id': 42, 'type': 'workspace', 'focused': True}, {'nodes': [container]}):
            with self.subTest(tree=tree):
                self.assertEqual(self.lookup(tree), (None, NO_VIEW))

    def test_malformed_tree_is_unavailable(self):
        for tree in ({'nodes': 'bad'}, {'nodes': [view(window_properties='bad')]},
                     {'nodes': [dict(view(), id='42')]}):
            with self.subTest(tree=tree):
                self.assertEqual(self.lookup(tree), (None, IPC_UNAVAILABLE))

    def test_wire_errors_are_unavailable(self):
        for data in (b'', b'i3-ipc' + struct.pack('=II', 9 * 1024 * 1024, GET_TREE),
                     b'bad-ip' + struct.pack('=II', 0, GET_TREE),
                     b'i3-ipc' + struct.pack('=II', 0, 2),
                     b'i3-ipc' + struct.pack('=II', 1, GET_TREE) + b'{'):
            with self.subTest(data=data[:14]), patch.object(sway_focus.socket, 'socket', return_value=Wire(data)):
                self.assertEqual(self.ipc.lookup(), (None, IPC_UNAVAILABLE))

    def test_timeout_is_unavailable(self):
        with patch.object(sway_focus.socket, 'socket', side_effect=TimeoutError):
            self.assertEqual(self.ipc.lookup(), (None, IPC_UNAVAILABLE))

    def test_no_socket_and_none_to_rediscover(self):
        ipc = SwayIPC(None, '/nonexistent', uid=1000)
        wire = Wire(b'')
        with patch.object(sway_focus.socket, 'socket', return_value=wire):
            self.assertEqual(ipc.lookup(), (None, IPC_UNAVAILABLE))
        self.assertEqual(wire.connected, [])

    def test_follows_a_restarted_sway(self):
        wire = Wire(reply({'nodes': [view()]}))
        wire.connect = Mock(side_effect=[FileNotFoundError(), None])
        with patch.object(sway_focus.socket, 'socket', return_value=wire), \
                patch.object(self.ipc, 'rediscover', return_value='/new/sway.sock'):
            info, _ = self.ipc.lookup()
        self.assertEqual(info['sway_socket'], '/new/sway.sock')
        self.assertEqual(self.ipc.socket_path, '/new/sway.sock')
        self.assertEqual(wire.connect.call_args_list[-1].args, ('/new/sway.sock',))


class FocusMatchTests(unittest.TestCase):
    def setUp(self):
        self.ipc = SwayIPC('/fake/sway.sock', '/nonexistent', uid=1000)
        self.target = {'sway_id': 42, 'sway_socket': '/fake/sway.sock'}

    def test_the_delivery_seat_must_hold_focus(self):
        keyboard = [{'type': 'keyboard'}]
        for seats in ([{'name': 'seat0', 'focus': 42}],
                      [{'name': 'main', 'focus': 42, 'devices': keyboard}],
                      [{'name': 'seat0', 'focus': 42, 'devices': []}]):
            with self.subTest(seats=seats), patch.object(self.ipc, 'request', return_value=seats) as request:
                self.assertTrue(self.ipc.focus_matches(self.target))
                request.assert_called_once_with(GET_SEATS)

    def test_another_seat_holding_the_target_does_not_count(self):
        keyboard = [{'type': 'keyboard'}]
        for seats in ([{'name': 'seat0', 'focus': 9, 'devices': keyboard},
                       {'name': 'seat1', 'focus': 42, 'devices': []}],
                      # An empty seat may still receive wtype's virtual keyboard.
                      [{'name': 'seat0', 'focus': 42, 'devices': keyboard},
                       {'name': 'seat1', 'focus': 9, 'devices': []}],
                      [{'name': 'seat0', 'focus': 9, 'devices': []},
                       {'name': 'main', 'focus': 42, 'devices': keyboard}],
                      # Two seats with input: the keys' seat is ambiguous.
                      [{'name': 'seat0', 'focus': 42, 'devices': keyboard},
                       {'name': 'seat1', 'focus': 42, 'devices': keyboard}],
                      [{'name': 'seat0', 'focus': 42}, {'name': 'seat1', 'focus': 42}]):
            with self.subTest(seats=seats), patch.object(self.ipc, 'request', return_value=seats):
                self.assertFalse(self.ipc.focus_matches(self.target))

    def test_moved_focus_or_other_instance_does_not_match(self):
        with patch.object(self.ipc, 'request', return_value=[{'name': 'seat0', 'focus': 42}]):
            self.assertFalse(self.ipc.focus_matches(dict(self.target, sway_id=43)))
            self.assertFalse(self.ipc.focus_matches(dict(self.target, sway_socket='/old')))
        for seats in (None, {'focus': 42}, ['bad'], [], [{'focus': 42}, 'bad']):
            with self.subTest(seats=seats), patch.object(self.ipc, 'request', return_value=seats):
                self.assertFalse(self.ipc.focus_matches(self.target))

    def test_no_target_never_queries(self):
        with patch.object(self.ipc, 'request') as request:
            self.assertFalse(self.ipc.focus_matches(None))
            self.assertFalse(self.ipc.focus_matches({'sway_id': '42'}))
        request.assert_not_called()


class RediscoveryTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.ipc = SwayIPC(None, self.dir.name, uid=1000)
        self.listeners = []

    def socket_file(self, pid, live):
        path = os.path.join(self.dir.name, f'sway-ipc.1000.{pid}.sock')
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(path)
        server.listen(1)
        if live:
            self.listeners.append(server)
            self.addCleanup(server.close)
        else:
            server.close()  # leaves the file behind, like a crashed Sway
        return path

    def test_single_live_socket_is_adopted(self):
        self.socket_file(100, live=False)
        live = self.socket_file(200, live=True)
        self.assertEqual(self.ipc.rediscover(), live)

    def test_ambiguous_or_absent_instances_are_refused(self):
        self.assertIsNone(self.ipc.rediscover())
        self.socket_file(100, live=True)
        self.socket_file(200, live=True)
        self.assertIsNone(self.ipc.rediscover())

    def test_other_users_and_plain_files_are_ignored(self):
        live = self.socket_file(100, live=True)
        Path(self.dir.name, 'sway-ipc.1000.300.sock').write_text('')
        Path(self.dir.name, 'sway-ipc.1001.400.sock').write_text('')
        self.assertEqual(self.ipc.rediscover(), live)

    def test_unreadable_runtime_dir(self):
        with patch.object(sway_focus.Path, 'glob', side_effect=PermissionError):
            self.assertIsNone(self.ipc.rediscover())


class SessionDetectionTests(unittest.TestCase):
    def test_desktop_identity_wins(self):
        for env, expected in (
            ({'XDG_CURRENT_DESKTOP': 'sway'}, True),
            ({'XDG_CURRENT_DESKTOP': 'SwayFX', 'NIRI_SOCKET': '/n'}, True),
            ({'XDG_SESSION_DESKTOP': 'sway:wlroots'}, True),
            ({'XDG_CURRENT_DESKTOP': 'niri', 'SWAYSOCK': '/s'}, False),
            ({'XDG_CURRENT_DESKTOP': 'sway', 'XDG_SESSION_TYPE': 'x11'}, False),
        ):
            with self.subTest(env=env), patch.object(SwayIPC, '_answers') as answers:
                self.assertEqual(is_sway_session(env), expected)
                answers.assert_not_called()

    def test_without_desktop_identity_a_live_swaysock_wins(self):
        for leftover in ({}, {'NIRI_SOCKET': '/n'}, {'HYPRLAND_INSTANCE_SIGNATURE': 'h'}):
            env = dict(leftover, SWAYSOCK='/s')
            with self.subTest(leftover=leftover):
                with patch.object(SwayIPC, '_answers', return_value=True):
                    self.assertTrue(is_sway_session(env))
                with patch.object(SwayIPC, '_answers', return_value=False):
                    # A dead socket alone still fails closed; leftovers decide otherwise.
                    self.assertEqual(is_sway_session(env), not leftover)

    def test_nothing_set_is_not_sway(self):
        self.assertFalse(is_sway_session({}))


if __name__ == '__main__':
    unittest.main()
