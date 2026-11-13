"""Sway delivery: transport choice, focus rechecks, recovery and live typing.

IPC lives in sway_focus (see test_sway_focus); here it is a stub. All I/O is
mocked: no audio, clipboard, keyboard or service changes.
"""
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib' / 'src'))
import text_injector as patched_injector
from sway_focus import GET_SEATS, GET_TREE, IPC_UNAVAILABLE, MULTI_SEAT, NO_VIEW, SwayIPC
from tests.text_injector_helpers import ConfigStub, make_injector

Outcome = patched_injector.InjectionOutcome


def target(xwayland=False, sway_id=42):
    return dict(source='sway', sway_socket='/fake/sway.sock', sway_id=sway_id,
                app_id=None if xwayland else 'firefox',
                **{'class': 'Chatgpt' if xwayland else 'firefox'},
                title='Synthetic title', xwayland=xwayland,
                x11_window_id=123 if xwayland else None)


def seats(focus=42):
    return [{'name': 'seat0', 'focus': focus}]


def sway_ipc(focus=42, window=None, reason=None):
    """A SwayIPC whose GET_SEATS reply is `focus` (None: IPC lost)."""
    ipc = SwayIPC('/fake/sway.sock', '/nonexistent', uid=1000)
    ipc.request = Mock(return_value=seats(focus) if focus is not None else None)
    ipc.lookup = Mock(return_value=(window, reason))
    return ipc


class DeliveryTests(unittest.TestCase):
    """_paste_via_clipboard and auto-submit with an already known target."""

    def setUp(self):
        i = self.injector = make_injector()
        i._sway = sway_ipc()
        i.sway_xdotool_available = True
        i.wtype_available = i.ydotool_available = True
        i._save_clipboard = Mock(return_value=None)
        i._copy_text_to_clipboard = Mock(return_value=True)
        i._send_paste_keys_wtype = Mock(return_value=True)
        i._send_paste_keys_xdotool = Mock(return_value=True)
        i._send_paste_keys_slow = Mock(return_value=True)
        i._restore_clipboard = Mock()
        i._clear_stuck_modifiers = Mock()
        i._send_enter_if_auto_submit = Mock()
        i._withdraw_undelivered = Mock()
        for patcher in (patch.dict(os.environ, {'SWAYSOCK': '/fake/sway.sock'}, clear=True),
                        patch.object(patched_injector, 'log'),
                        patch.object(patched_injector.time, 'sleep')):
            patcher.start()
            self.addCleanup(patcher.stop)

    def paste(self, expected, xfocus='123', stream=None):
        with patch.object(patched_injector.subprocess, 'run', return_value=Mock(returncode=0, stdout=xfocus)):
            return self.injector._paste_via_clipboard(
                'Synthetic fixture', 'ctrl+v', False, False, stream=stream, expected_window=expected)

    def senders(self):
        i = self.injector
        return {name: getattr(i, f'_send_paste_keys_{name}').called for name in ('wtype', 'xdotool', 'slow')}

    def test_native_view_uses_wtype(self):
        self.assertTrue(self.paste(target()))
        self.assertEqual(self.senders(), {'wtype': True, 'xdotool': False, 'slow': False})

    def test_empty_second_seat_refuses_delivery_and_withdraws_text(self):
        i = self.injector
        i._sway.request.return_value = [
            {'name': 'seat0', 'focus': 42, 'devices': [{'type': 'keyboard'}]},
            {'name': 'seat1', 'focus': 99, 'devices': []},
        ]
        i._save_clipboard.return_value = b'previous'
        self.assertFalse(self.paste(target()))
        self.assertFalse(any(self.senders().values()))
        i._withdraw_undelivered.assert_called_once_with(b'previous', 'Synthetic fixture')

    def test_wtype_failure_rechecks_focus_before_ydotool(self):
        i = self.injector
        i._send_paste_keys_wtype.return_value = False
        for name, later_focus, fallback in (('focus kept', 42, True), ('focus moved', 99, False)):
            with self.subTest(name):
                i._sway = sway_ipc()
                i._sway.request.side_effect = [seats(42), seats(later_focus)]
                i._send_paste_keys_slow.reset_mock()
                i._withdraw_undelivered.reset_mock()
                i._save_clipboard.return_value = b'previous'
                self.assertEqual(self.paste(target()), fallback)
                self.assertEqual(i._sway.request.call_count, 2)
                self.assertEqual(i._send_paste_keys_slow.called, fallback)
                self.assertEqual(i._withdraw_undelivered.called, not fallback)
                if not fallback:
                    i._withdraw_undelivered.assert_called_once_with(b'previous', 'Synthetic fixture')

    def test_wtype_failure_in_live_typing_stops_the_session(self):
        i = self.injector
        i._send_paste_keys_wtype.return_value = False
        i._sway.request.side_effect = [seats(42), seats(99)]
        stream = patched_injector._StreamSession()
        stream.saved_clipboard = b'before dictation'
        self.assertFalse(self.paste(target(), stream=stream))
        i._send_paste_keys_slow.assert_not_called()
        self.assertTrue(stream.focus_lost)

    def test_ydotool_first_needs_no_second_check(self):
        i = self.injector
        i.wtype_available = False
        self.assertTrue(self.paste(target()))
        i._send_paste_keys_slow.assert_called_once_with('ctrl+v')
        self.assertEqual(i._sway.request.call_count, 1)

    def test_xwayland_view_prefers_ydotool(self):
        # uinput follows keyboard focus; XTest would bypass a launcher holding it.
        with patch.object(patched_injector.subprocess, 'run') as run:
            self.assertTrue(self.injector._paste_via_clipboard(
                'Synthetic fixture', 'ctrl+v', False, False, expected_window=target(True)))
        self.injector._send_paste_keys_slow.assert_called_once_with('ctrl+v')
        self.assertEqual(self.senders(), {'wtype': False, 'xdotool': False, 'slow': True})
        run.assert_not_called()  # no X11 query needed

    def test_xwayland_without_ydotool_uses_xdotool(self):
        self.injector.ydotool_available = False
        self.assertTrue(self.paste(target(True)))
        self.injector._send_paste_keys_xdotool.assert_called_once_with('ctrl+v')
        self.assertEqual(self.senders(), {'wtype': False, 'xdotool': True, 'slow': False})

    def test_xdotool_route_needs_a_nameable_key(self):
        i = self.injector
        i.ydotool_available = False
        with patch.object(patched_injector.subprocess, 'run') as run:
            self.assertTrue(i._paste_via_clipboard(
                'Synthetic fixture', 'ctrl+bogus', False, False, expected_window=target(True)))
        self.assertFalse(any(self.senders().values()))
        run.assert_not_called()
        i._restore_clipboard.assert_not_called()

    def test_xwayland_transport_failure_is_not_retried(self):
        i = self.injector
        for xdotool in (True, False):
            with self.subTest(xdotool=xdotool):
                i.sway_xdotool_available = xdotool
                i._send_paste_keys_xdotool.return_value = i._send_paste_keys_slow.return_value = False
                i._send_paste_keys_wtype.reset_mock()
                self.assertFalse(self.paste(target(True)))
                i._send_paste_keys_wtype.assert_not_called()

    def test_xwayland_with_neither_tool_is_clipboard_only(self):
        i = self.injector
        i.sway_xdotool_available = i.ydotool_available = False
        stream = patched_injector._StreamSession()
        stream.saved_clipboard = None
        for session in (None, stream):
            with self.subTest(stream=session is not None):
                self.assertTrue(self.paste(target(True), stream=session))
                self.assertFalse(any(self.senders().values()))
                i._restore_clipboard.assert_not_called()
                i._send_enter_if_auto_submit.assert_not_called()
        self.assertTrue(stream.not_pasted)

    def test_refused_paste_sends_nothing_and_withdraws(self):
        cases = (
            ('focus moved', dict(focus=99), target()),
            ('IPC lost', dict(focus=None), target()),
            ('other Sway instance', dict(), dict(target(), sway_socket='/old')),
            ('X11 focus moved', dict(), target(True)),
            ('X11 window id missing', dict(), dict(target(True), x11_window_id=None)),
        )
        self.injector.ydotool_available = False  # X11 checks guard only xdotool
        for name, ipc, expected in cases:
            with self.subTest(name):
                i = self.injector
                i._sway = sway_ipc(**ipc)
                i._withdraw_undelivered.reset_mock()
                i._save_clipboard.return_value = b'previous'
                self.assertFalse(self.paste(expected, xfocus='456'))
                self.assertFalse(any(self.senders().values()))
                i._withdraw_undelivered.assert_called_once_with(b'previous', 'Synthetic fixture')

    def test_clipboard_only_live_typing_keeps_whole_dictation(self):
        i = self.injector
        i.sway_xdotool_available = i.ydotool_available = False
        stream = patched_injector._StreamSession()
        stream.saved_clipboard = None
        for chunk in ('hello there', ' general kenobi'):
            with patch.object(patched_injector.subprocess, 'run'):
                self.assertTrue(i._paste_via_clipboard(chunk, 'ctrl+v', False, False, stream=stream, expected_window=target(True)))
        self.assertEqual(i._copy_text_to_clipboard.call_args.args[0], 'hello there general kenobi')
        self.assertTrue(stream.not_pasted)

    def test_refused_stream_chunk_withdraws_to_stream_clipboard(self):
        stream = patched_injector._StreamSession()
        stream.saved_clipboard = b'before dictation'
        self.injector._sway = sway_ipc(focus=None)
        self.assertFalse(self.paste(target(), stream=stream))
        self.injector._withdraw_undelivered.assert_called_once_with(b'before dictation', 'Synthetic fixture')
        self.assertTrue(stream.focus_lost)

    def test_native_clipboard_only_ignores_xwayland_tool(self):
        i = self.injector
        i.wtype_available = i.ydotool_available = False
        self.assertTrue(self.paste(target()))
        i._restore_clipboard.assert_not_called()
        i._send_enter_if_auto_submit.assert_not_called()
        i._send_paste_keys_xdotool.assert_not_called()

    def test_native_chord_validation_ignores_xwayland_tool(self):
        i = self.injector
        i.ydotool_available = False
        self.assertFalse(i._chord_is_usable('ctrl+XF86AudioPlay', target()))
        i.wtype_available = False
        self.assertFalse(i._chord_is_usable('ctrl+XF86AudioPlay', target(True)))
        self.assertTrue(i._chord_is_usable('ctrl+tab', target(True)))


class AutoSubmitTests(unittest.TestCase):
    def setUp(self):
        i = self.injector = make_injector()
        i.config_manager = ConfigStub({'auto_submit': True})
        i._sway = sway_ipc()
        i.sway_xdotool_available = True
        i.wtype_available = True
        i._run_ydotool = Mock(return_value=Mock(returncode=0))
        logger = patch.object(patched_injector, 'log')
        logger.start()
        self.addCleanup(logger.stop)

    def submit(self, expected, xfocus='123', cancel_event=None):
        with patch.object(patched_injector.subprocess, 'run', return_value=Mock(returncode=0, stdout=xfocus)) as run:
            self.injector._send_enter_if_auto_submit(cancel_event=cancel_event, expected_window=expected)
        return [c.args[0] for c in run.call_args_list], self.injector._run_ydotool.called

    def test_xwayland_enter_prefers_ydotool(self):
        commands, _ = self.submit(target(True))
        self.injector._run_ydotool.assert_called_once_with(['key', '28:1', '28:0'], timeout=1)
        self.assertEqual(commands, [])

    def test_xwayland_enter_without_ydotool_uses_xdotool(self):
        self.injector.ydotool_available = False
        commands, ydotool = self.submit(target(True))
        self.assertEqual(commands[-1][:3], ['xdotool', 'key', '--clearmodifiers'])
        self.assertEqual(commands[-1][3], 'Return')
        self.assertFalse(ydotool)
        self.assertFalse(any(c[0] == 'wtype' for c in commands))

    def test_xwayland_enter_with_neither_tool_is_skipped(self):
        self.injector.sway_xdotool_available = self.injector.ydotool_available = False
        self.assertEqual(self.submit(target(True)), ([], False))

    def test_enter_never_follows_moved_focus(self):
        for name, ipc, expected, ydotool_available in (
                ('Sway focus', dict(focus=99), target(True), True),
                ('X11 focus', dict(), target(True), False),
                ('no target', dict(), None, True)):
            with self.subTest(name):
                self.injector._sway = sway_ipc(**ipc)
                self.injector.ydotool_available = ydotool_available
                commands, ydotool = self.submit(expected, xfocus='456')
                self.assertNotIn(['xdotool', 'key', '--clearmodifiers', 'Return'], commands)
                self.assertFalse(ydotool)

    def test_cancel_during_focus_check_sends_nothing(self):
        cancel = threading.Event()
        self.injector._sway.request.side_effect = lambda kind: (cancel.set(), seats())[1]
        commands, ydotool = self.submit(target(), cancel_event=cancel)
        self.assertEqual((commands, ydotool), ([], False))


class InjectTests(unittest.TestCase):
    """Whole dictations through inject_text and live typing."""

    def setUp(self):
        i = self.injector = make_injector()
        i.config_manager = ConfigStub({'auto_submit': True, 'append_trailing_space': False})
        i._sway = SwayIPC('/fake/sway.sock', '/nonexistent', uid=1000)
        i.sway_xdotool_available = True
        i.ydotool_available = False  # observe keys through the xdotool stub
        self.focus = 42
        self.tree = {'nodes': [dict(id=42, focused=True, shell='xwayland', window=123,
                                    window_properties={'class': 'Chatgpt'})]}
        i._sway.request = Mock(side_effect=lambda kind: self.tree if kind == GET_TREE else seats(self.focus))
        self.sent = []
        def xdotool(chord):
            self.sent.append(chord)
            return True
        for patcher in (
            patch.dict(os.environ, {'SWAYSOCK': '/fake/sway.sock'}, clear=True),
            patch.object(i, '_save_clipboard', return_value=None),
            patch.object(i, '_copy_text_to_clipboard', return_value=True),
            patch.object(i, '_restore_clipboard'),
            patch.object(i, '_withdraw_undelivered'),
            patch.object(i, '_send_paste_keys_xdotool', side_effect=xdotool),
            patch.object(patched_injector.subprocess, 'run', return_value=Mock(returncode=0, stdout='123')),
            patch.object(patched_injector.time, 'sleep'),
            patch.object(patched_injector, 'log'),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def requests(self):
        return [c.args[0] for c in self.injector._sway.request.call_args_list]

    def test_one_shot_reads_tree_once_and_rechecks_before_paste_and_enter(self):
        self.assertEqual(self.injector.inject_text('hello'), Outcome.INJECTED)
        self.assertEqual(self.sent, ['ctrl+v', 'enter'])
        self.assertEqual(self.requests(), [GET_SEATS, GET_TREE, GET_SEATS, GET_SEATS])

    def test_live_typing_reads_tree_once_per_dictation(self):
        i = self.injector
        session = i.begin_stream()
        for chunk in ('one', 'two', 'three'):
            self.assertEqual(i.inject_stream_chunk(session, chunk, final=True), Outcome.INJECTED)
        self.assertEqual(i.end_stream(session), Outcome.INJECTED)
        self.assertEqual(self.requests().count(GET_TREE), 1)
        self.assertEqual(self.sent, ['ctrl+v', 'ctrl+v', 'ctrl+v', 'enter'])

    def test_live_typing_stops_for_good_when_focus_moves(self):
        i = self.injector
        i.config_manager.settings['append_trailing_space'] = True
        session = i.begin_stream()
        self.assertEqual(i.inject_stream_chunk(session, 'one', final=True), Outcome.INJECTED)
        self.focus = 99
        self.assertEqual(i.inject_stream_chunk(session, 'two', final=True), Outcome.FAILED)
        self.focus = 42  # back on the original window: still nothing more
        self.assertEqual(i.inject_stream_chunk(session, 'three', final=True), Outcome.FAILED)
        self.assertEqual(i.end_stream(session), Outcome.FAILED)
        self.assertEqual(self.sent, ['ctrl+v'])  # no hole-filling chunk, space or Enter
        self.assertEqual(i._last_text, 'one two three')
        i.release_stream(session)

    def test_held_words_reach_recovery_after_focus_moves(self):
        # "new" may begin "new line", so it is held for the next chunk.
        i = self.injector
        session = i.begin_stream()
        self.assertEqual(i.inject_stream_chunk(session, 'one', final=True), Outcome.INJECTED)
        self.focus = 99
        i.inject_stream_chunk(session, 'two new', final=False)
        self.assertEqual(session.held, ['new'])
        self.assertEqual(i.end_stream(session), Outcome.FAILED)
        self.assertEqual(i._last_text, 'one two new')
        i.release_stream(session)

    def test_unknown_focus_logs_why(self):
        for tree, focus, reason, message in (
            ({'id': 42, 'type': 'workspace'}, 42, NO_VIEW, 'No Sway window has focus'),
            (None, 42, IPC_UNAVAILABLE, 'Sway did not answer'),
            (self.tree, None, MULTI_SEAT, 'more than one seat'),
        ):
            with self.subTest(reason=reason):
                self.tree = tree
                if focus is None:
                    self.injector._sway.request.side_effect = lambda kind: (
                        self.tree if kind == GET_TREE else seats(42) + seats(42))
                with patch.object(patched_injector, 'log') as log:
                    self.assertEqual(self.injector.inject_text('hello'), Outcome.FAILED)
                self.assertTrue(any(message in c.args[0] for c in log.call_args_list))
                self.injector._copy_text_to_clipboard.assert_not_called()
                self.assertEqual(self.sent, [])


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.injector = make_injector()
        self.injector._sway = sway_ipc(window=None, reason=IPC_UNAVAILABLE)
        for patcher in (
            patch.dict(os.environ, {'SWAYSOCK': '/fake/sway.sock'}, clear=True),
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
                        self.assertEqual(i.inject_text('new dictation'), Outcome.FAILED)
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
                    self.assertEqual(i.inject_stream_chunk(session, 'new dictation', final=True), Outcome.FAILED)
                    self.assertEqual(i.end_stream(session), Outcome.FAILED)
                    self.assertEqual(i._last_text, 'older dictation' if denied else 'complete transcript')
                    copy.assert_not_called()
                    paste.assert_not_called()
                i.release_stream(session)

    def test_failed_repaste_does_not_undo_concurrent_clear(self):
        i = self.injector
        i._last_text = 'older dictation'
        def clear_during_lookup():
            i.recover_last('clear_last')
            return None, IPC_UNAVAILABLE
        i._sway.lookup.side_effect = clear_during_lookup
        self.assertFalse(i.recover_last('paste_last')[0])
        self.assertIsNone(i._last_text)

    def test_cancel_during_delivery_focus_checks_does_not_dispatch(self):
        for xwayland in (False, True):
            with self.subTest(xwayland=xwayland):
                i = self.injector
                i.sway_xdotool_available = True
                i.ydotool_available = False  # cancel lands inside the X11 query
                i._sway = sway_ipc()
                session = i.begin_stream()
                def focus(kind):
                    if not xwayland:
                        session.cancelled.set()
                    return seats()
                def xfocus(*args, **kwargs):
                    session.cancelled.set()
                    return Mock(returncode=0, stdout='123')
                i._sway.request.side_effect = focus
                with (
                    patch.object(i, '_save_clipboard', return_value=None),
                    patch.object(i, '_copy_text_to_clipboard', return_value=True),
                    patch.object(patched_injector.subprocess, 'run', side_effect=xfocus),
                    patch.object(patched_injector.time, 'sleep'),
                    patch.object(i, '_send_paste_keys_wtype') as wtype,
                    patch.object(i, '_send_paste_keys_xdotool') as xdotool,
                    patch.object(i, '_send_paste_keys_slow') as ydotool,
                ):
                    i.wtype_available = True
                    self.assertFalse(i._paste_via_clipboard('new', 'ctrl+v', False, False, stream=session, expected_window=target(xwayland)))
                    wtype.assert_not_called()
                    xdotool.assert_not_called()
                    ydotool.assert_not_called()
                i.release_stream(session)


class WithdrawTests(unittest.TestCase):
    def setUp(self):
        self.injector = make_injector()

    def test_restores_saved_clipboard_unless_user_copied_since(self):
        with patch.object(self.injector, '_restore_clipboard') as restore:
            self.injector._withdraw_undelivered(b'previous', 'Synthetic')
        restore.assert_called_once_with(b'previous', injected=b'Synthetic', delay=0)

    def test_clears_a_clipboard_that_was_empty(self):
        for current, cleared in ((b'Synthetic', True), (b'copied since', False)):
            with self.subTest(current=current), \
                    patch.object(self.injector, '_save_clipboard', return_value=current), \
                    patch.object(patched_injector.subprocess, 'run') as run:
                self.injector._withdraw_undelivered(None, 'Synthetic')
            self.assertEqual(run.called, cleared)
            if cleared:
                self.assertEqual(run.call_args.args[0], ['wl-copy', '--clear'])

    def test_falls_back_to_the_text_clipboard_writer(self):
        for failure in (Mock(returncode=1), OSError('no wl-copy')):
            clipboard = Mock()
            with self.subTest(failure=failure), \
                    patch.object(self.injector, '_save_clipboard', return_value=b'Synthetic'), \
                    patch.object(patched_injector, 'pyperclip', clipboard), \
                    patch.object(patched_injector.subprocess, 'run',
                                 side_effect=failure if isinstance(failure, Exception) else None,
                                 return_value=failure):
                self.injector._withdraw_undelivered(None, 'Synthetic')
            clipboard.copy.assert_called_once_with('')


class LookupRoutingTests(unittest.TestCase):
    def test_sway_never_falls_back_to_other_compositors(self):
        i = make_injector()
        i._sway = sway_ipc(window=None, reason=IPC_UNAVAILABLE)
        with patch.object(patched_injector.subprocess, 'run') as run:
            self.assertIsNone(i._get_active_window_info())
        run.assert_not_called()

    def test_other_compositors_unchanged(self):
        i = make_injector()
        with patch.object(patched_injector.subprocess, 'run', return_value=Mock(returncode=0, stdout='{"app_id":"kitty"}')):
            self.assertEqual(i._get_active_window_info()['source'], 'niri')

    def test_chords_resolve_from_sway_identity(self):
        i = make_injector()
        self.assertEqual(i._resolve_paste_chord({'class': 'kitty', 'source': 'sway'})[0], 'ctrl+shift+v')
        self.assertEqual(i._resolve_paste_chord(target())[0], 'ctrl+v')
        self.assertIn('chatgpt', i.focused_window_identifiers(target(True)))
        i.config_manager = ConfigStub({'applications': {'firefox': {'auto_paste': 'ctrl+y'}}})
        self.assertEqual(i._resolve_paste_chord({'class': 'firefox'}), ('ctrl+y', 'firefox'))

    def test_only_a_matching_x11_answer_confirms_focus(self):
        i = make_injector()
        cases = [Mock(returncode=1, stdout=''), Mock(returncode=0, stdout='bad'),
                 Mock(returncode=0, stdout='456'),
                 OSError('unavailable'), patched_injector.subprocess.TimeoutExpired('xdotool', 0.5)]
        for result in cases:
            with self.subTest(result=result), \
                    patch.object(patched_injector.subprocess, 'run',
                                 side_effect=result if isinstance(result, Exception) else None,
                                 return_value=result):
                self.assertFalse(i._sway_x11_focused(target(True)))
        with patch.object(patched_injector.subprocess, 'run', return_value=Mock(returncode=0, stdout='123')):
            self.assertTrue(i._sway_x11_focused(target(True)))
        self.assertFalse(i._sway_x11_focused(dict(target(True), x11_window_id=None)))


class SessionTests(unittest.TestCase):
    def test_other_desktops_ignore_stale_socket(self):
        for desktop in ('niri', 'Hyprland', 'GNOME', 'KDE'):
            with self.subTest(desktop=desktop), patch.dict(os.environ, {
                    'XDG_CURRENT_DESKTOP': desktop, 'SWAYSOCK': '/old/socket',
                    'XDG_SESSION_TYPE': 'wayland'}, clear=True):
                self.assertFalse(patched_injector.TextInjector.sway_session())

    def test_session_is_captured_and_tools_are_separate(self):
        with (
            patch.dict(os.environ, {'XDG_CURRENT_DESKTOP': 'sway', 'SWAYSOCK': '/fake/socket', 'DISPLAY': ':0'}, clear=True),
            patch.object(patched_injector.TextInjector, '_check_ydotool', return_value=False),
            patch.object(patched_injector.shutil, 'which', side_effect=lambda name: '/usr/bin/xdotool' if name == 'xdotool' else None),
            patch.object(patched_injector, 'YdotooldSession'),
            patch.object(patched_injector, 'log') as log,
        ):
            i = patched_injector.TextInjector()
            self.assertIsInstance(i._sway, SwayIPC)
            self.assertTrue(i.sway_xdotool_available)
            self.assertFalse(i.xdotool_available)
            self.assertTrue(any('No injection backend found' in c.args[0] for c in log.call_args_list))
            os.environ['SWAYSOCK'] = '/changed'
            self.assertEqual(i._sway.socket_path, '/fake/socket')

    def test_xdotool_needs_an_x_display(self):
        with (
            patch.dict(os.environ, {'XDG_CURRENT_DESKTOP': 'sway', 'SWAYSOCK': '/fake/socket'}, clear=True),
            patch.object(patched_injector.TextInjector, '_check_ydotool', return_value=True),
            patch.object(patched_injector.shutil, 'which', return_value='/usr/bin/xdotool'),
            patch.object(patched_injector, 'YdotooldSession'),
            patch.object(patched_injector, 'log'),
        ):
            i = patched_injector.TextInjector()
        self.assertIsInstance(i._sway, SwayIPC)
        self.assertFalse(i.sway_xdotool_available)

    def test_non_sway_session_has_no_ipc(self):
        with (
            patch.dict(os.environ, {'XDG_CURRENT_DESKTOP': 'niri', 'SWAYSOCK': '/old/socket'}, clear=True),
            patch.object(patched_injector.TextInjector, '_check_ydotool', return_value=False),
            patch.object(patched_injector.shutil, 'which', return_value='/usr/bin/xdotool'),
            patch.object(patched_injector, 'YdotooldSession'),
            patch.object(patched_injector, 'log'),
        ):
            i = patched_injector.TextInjector()
        self.assertIsNone(i._sway)
        self.assertFalse(i.sway_xdotool_available)

    def test_desktop_tokens_on_sway_do_not_pick_other_paths(self):
        i = make_injector()
        i._sway = sway_ipc()
        env = {'XDG_SESSION_TYPE': 'wayland', 'XDG_CURRENT_DESKTOP': 'sway:GNOME',
               'SWAYSOCK': '/fake', 'HYPRLAND_INSTANCE_SIGNATURE': 'leftover'}
        with patch.dict(os.environ, env, clear=True), \
                patch.object(patched_injector.shutil, 'which', return_value='/usr/bin/hyprctl'):
            self.assertFalse(i._is_gnome_wayland_session())
            self.assertFalse(i._is_hyprland_session())
            i._sway = None
            self.assertTrue(i._is_gnome_wayland_session())
            self.assertTrue(i._is_hyprland_session())

    def test_leftover_hyprland_signature_does_not_route_sway_paste(self):
        i = make_injector()
        i._sway = sway_ipc()
        i.wtype_available = True
        i.config_manager = ConfigStub({'hyprland_paste_backend': 'ydotool'})
        with (
            patch.dict(os.environ, {'XDG_CURRENT_DESKTOP': 'sway', 'HYPRLAND_INSTANCE_SIGNATURE': 'leftover'}, clear=True),
            patch.object(patched_injector.shutil, 'which', return_value='/usr/bin/hyprctl'),
            patch.object(i, '_save_clipboard', return_value=None),
            patch.object(i, '_copy_text_to_clipboard', return_value=True),
            patch.object(i, '_restore_clipboard'),
            patch.object(i, '_clear_stuck_modifiers'),
            patch.object(i, '_send_shortcut_hyprland') as hyprctl,
            patch.object(i, '_send_paste_keys_slow') as ydotool,
            patch.object(i, '_send_paste_keys_wtype', return_value=True) as wtype,
            patch.object(patched_injector.time, 'sleep'),
        ):
            self.assertTrue(i._paste_via_clipboard('hello', 'ctrl+v', False, False, expected_window=target()))
        hyprctl.assert_not_called()
        ydotool.assert_not_called()
        wtype.assert_called_once_with('ctrl+v')


if __name__ == '__main__':
    unittest.main()
