import threading
import unittest
from contextlib import contextmanager
from unittest import mock

from tests.text_injector_helpers import make_injector, ConfigStub
from text_injector import InjectionOutcome, TextInjector


class DictationRecoveryTests(unittest.TestCase):
    def setUp(self):
        for name, value in (
            ('_get_active_window_info', {}),
            ('_is_gnome_wayland_session', False),
            ('_resolve_paste_chord', ('ctrl+v', None)),
        ):
            patcher = mock.patch.object(TextInjector, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    @contextmanager
    def _clipboard_session(self, injector):
        clipboard = {'text': b'original'}
        callbacks = []

        def write(data):
            clipboard['text'] = data
            return True, None

        injector.xdotool_available = True
        with (
            mock.patch.object(injector, '_is_x11_session', return_value=True),
            mock.patch.object(injector, '_save_clipboard', side_effect=lambda: clipboard['text']),
            mock.patch.object(injector, '_try_wl_copy', side_effect=write),
            mock.patch.object(injector, '_send_paste_keys_xdotool', return_value=True),
            mock.patch('text_injector.time.sleep'),
            mock.patch('text_injector.threading.Thread', side_effect=lambda target, **kw:
                       mock.Mock(start=lambda: callbacks.append(target))),
        ):
            yield clipboard, callbacks

    def test_disabled_dictation_is_not_recoverable_and_preserves_pending_restore(self):
        injector = make_injector()
        with self._clipboard_session(injector) as (clipboard, callbacks):
            injector.inject_text('allowed')
            with mock.patch.object(injector, '_resolve_paste_chord', return_value=(False, 'password-app')):
                injector.inject_text('secret passphrase')
            self.assertEqual(injector._last_text, 'allowed ')
            self.assertEqual(clipboard['text'], b'allowed ')
            callbacks[0]()
            self.assertEqual(clipboard['text'], b'original')
            self.assertTrue(injector.recover_last('copy_last')[0])
            self.assertEqual(clipboard['text'], b'allowed ')

        fresh = make_injector()
        with mock.patch.object(fresh, '_resolve_paste_chord', return_value=(False, 'password-app')):
            fresh.inject_text('secret passphrase')
        self.assertFalse(fresh.recover_last('copy_last')[0])

    def test_failed_writes_and_prewrite_exceptions_preserve_pending_restore(self):
        for failure in ('native-and-fallback', 'prewrite', 'copy-last'):
            with self.subTest(failure=failure):
                injector = make_injector()
                with self._clipboard_session(injector) as (clipboard, callbacks):
                    injector.inject_text('allowed')
                    with (
                        mock.patch.object(injector, '_try_wl_copy', return_value=(False, 'unavailable')),
                        mock.patch('text_injector.pyperclip.copy', side_effect=RuntimeError('unavailable')),
                    ):
                        if failure == 'copy-last':
                            self.assertFalse(injector.recover_last('copy_last')[0])
                        elif failure == 'prewrite':
                            with mock.patch.object(injector, '_save_clipboard', side_effect=RuntimeError('unavailable')):
                                self.assertEqual(injector.inject_text('next'), InjectionOutcome.FAILED)
                        else:
                            self.assertEqual(injector.inject_text('next'), InjectionOutcome.FAILED)
                    callbacks[0]()
                    self.assertEqual(clipboard['text'], b'original')

    def test_direct_typing_preserves_pending_clipboard_restore(self):
        injector = make_injector()
        with self._clipboard_session(injector) as (clipboard, callbacks):
            injector.inject_text('allowed')
            with (
                mock.patch.object(injector, '_is_gnome_wayland_session', return_value=True),
                mock.patch.object(injector, '_layout_is_type_safe', return_value=True),
                mock.patch.object(injector, '_clear_stuck_modifiers'),
                mock.patch.object(injector, '_type_text_ydotool', return_value=True),
            ):
                injector.inject_text('typed')
            callbacks[0]()
            self.assertEqual(clipboard['text'], b'original')

    def test_recovery_waits_for_restore_without_reporting_delivery_busy(self):
        injector = make_injector()
        injector._last_text = 'retained'
        restoring = threading.Event()
        release = threading.Event()
        recovering = threading.Event()
        results = []
        writes = []
        delivery_lock = injector._delivery_lock

        class ObservedDeliveryLock:
            def __enter__(self):
                delivery_lock.acquire()

            def __exit__(self, *args):
                delivery_lock.release()

            def acquire(self, **kwargs):
                acquired = delivery_lock.acquire(**kwargs)
                recovering.set()
                return acquired

            def release(self):
                delivery_lock.release()

        injector._delivery_lock = ObservedDeliveryLock()

        def read():
            restoring.set()
            if not release.wait(2):
                raise RuntimeError('test did not release restore')
            return b'retained'

        def write(data):
            writes.append(data)
            return True, None

        def recover():
            results.append(injector.recover_last('copy_last'))

        with (
            mock.patch.object(injector, '_save_clipboard', side_effect=read),
            mock.patch.object(injector, '_try_wl_copy', side_effect=write),
            mock.patch('text_injector.time.sleep'),
        ):
            injector._restore_clipboard(b'original', injected=b'retained')
            worker = threading.Thread(target=recover)
            try:
                self.assertTrue(restoring.wait(2))
                worker.start()
                self.assertTrue(recovering.wait(2))
            finally:
                release.set()
                if worker.ident is not None:
                    worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertTrue(results[0][0], results)
            self.assertEqual(writes, [b'original', b'retained'])

    def test_retains_final_unicode_text_and_does_not_repeat_processing(self):
        injector = make_injector()
        injector.config_manager = ConfigStub({'post_transcription_hook': 'transform'})
        with (
            mock.patch.object(injector, '_preprocess_text', return_value='clean') as preprocess,
            mock.patch('text_injector.subprocess.run', return_value=mock.Mock(
                returncode=0, stdout='Café 東京', stderr='')) as hook,
            mock.patch.object(injector, '_paste_via_clipboard', return_value=False) as paste,
            mock.patch.object(injector, '_copy_text_to_clipboard', return_value=True) as copy,
        ):
            self.assertEqual(injector.inject_text('raw'), InjectionOutcome.FAILED)
            self.assertTrue(injector.recover_last('copy_last')[0])
            self.assertFalse(injector.recover_last('paste_last')[0])
            preprocess.assert_called_once_with('raw')
            hook.assert_called_once()
            copy.assert_called_once_with('Café 東京')
            paste.assert_has_calls([mock.call('Café 東京', 'ctrl+v', False, True),
                                   mock.call('Café 東京', 'ctrl+v', False, False)])

    def test_replacement_empty_filtered_and_consumed_results(self):
        injector = make_injector()
        with mock.patch.object(injector, '_paste_via_clipboard', return_value=True):
            injector.inject_text('first')
            injector.inject_text('second')
            self.assertEqual(injector._last_text, 'second ')
            injector.inject_text(' ')
            with mock.patch.object(injector, '_preprocess_text', return_value=''):
                injector.inject_text('filtered')
            injector.config_manager = ConfigStub({'post_transcription_hook': 'consume'})
            with mock.patch('text_injector.subprocess.run', return_value=mock.Mock(returncode=77)):
                self.assertEqual(injector.inject_text('consumed'), InjectionOutcome.CONSUMED)
            self.assertEqual(injector._last_text, 'second ')
        self.assertTrue(injector.recover_last('clear_last')[0])
        self.assertTrue(injector.recover_last('clear_last')[0])
        self.assertFalse(injector.recover_last('copy_last')[0])
        self.assertFalse(injector.recover_last('paste_last')[0])
        self.assertIsNone(make_injector()._last_text)

    def test_busy_recovery_is_rejected_and_clear_does_not_cancel_delivery(self):
        injector = make_injector()
        started = threading.Event()
        release = threading.Event()

        def deliver(text, *args):
            started.set()
            return release.wait(2)

        with mock.patch.object(injector, '_paste_via_clipboard', side_effect=deliver):
            worker = threading.Thread(target=injector.inject_text, args=('latest',))
            worker.start()
            try:
                self.assertTrue(started.wait(2))
                self.assertIn('busy', injector.recover_last('copy_last')[1])
                self.assertIn('busy', injector.recover_last('paste_last')[1])
                self.assertTrue(injector.recover_last('clear_last')[0])
            finally:
                release.set()
                worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertIsNone(injector._last_text)

    def test_copy_cancels_pending_restore_of_identical_text(self):
        injector = make_injector()
        injector._last_text = 'hello'
        callbacks = []
        with (
            mock.patch('text_injector.threading.Thread', side_effect=lambda target, **kw:
                       mock.Mock(start=lambda: callbacks.append(target))),
            mock.patch('text_injector.time.sleep'),
            mock.patch.object(injector, '_try_wl_copy', return_value=(True, None)) as write,
        ):
            injector._restore_clipboard(b'old', injected=b'hello')
            self.assertTrue(injector.recover_last('copy_last')[0])
            callbacks[0]()
            write.assert_called_once_with(b'hello')

    def test_recovery_releases_lock_after_tool_failure(self):
        injector = make_injector()
        injector._last_text = 'private text'
        with mock.patch.object(injector, '_copy_text_to_clipboard', side_effect=RuntimeError('private text')):
            ok, message = injector.recover_last('copy_last')
        self.assertFalse(ok)
        self.assertNotIn('private text', message)
        with mock.patch.object(injector, '_copy_text_to_clipboard', return_value=True):
            self.assertTrue(injector.recover_last('copy_last')[0])

    def test_repaste_respects_disabled_app(self):
        injector = make_injector()
        injector._last_text = 'hello'
        with (
            mock.patch.object(injector, '_get_active_window_info', return_value={}),
            mock.patch.object(injector, '_is_gnome_wayland_session', return_value=False),
            mock.patch.object(injector, '_resolve_paste_chord', return_value=(False, 'disabled-app')),
            mock.patch.object(injector, '_copy_text_to_clipboard') as copy,
            mock.patch.object(injector, '_send_enter_if_auto_submit') as enter,
        ):
            self.assertTrue(injector.recover_last('paste_last')[0])
        copy.assert_not_called()
        enter.assert_not_called()

    def test_repaste_never_auto_submits_clipboard_or_direct_type(self):
        for direct in (False, True):
            with self.subTest(direct=direct):
                injector = make_injector()
                injector.config_manager = ConfigStub({'auto_submit': True})
                injector._last_text = 'hello '
                with (
                    mock.patch.object(injector, '_get_active_window_info', return_value={}),
                    mock.patch.object(injector, '_is_gnome_wayland_session', return_value=direct),
                    mock.patch.object(injector, '_resolve_paste_chord', return_value=('ctrl+v', None)),
                    mock.patch.object(injector, '_layout_is_type_safe', return_value=True),
                    mock.patch.object(injector, '_clear_stuck_modifiers'),
                    mock.patch.object(injector, '_type_text_ydotool', return_value=True) as type_text,
                    mock.patch.object(injector, '_save_clipboard', return_value=b'old'),
                    mock.patch.object(injector, '_copy_text_to_clipboard', return_value=True),
                    mock.patch.object(injector, '_send_paste_keys_slow', return_value=True),
                    mock.patch.object(injector, '_restore_clipboard'),
                    mock.patch.object(injector, '_send_enter_if_auto_submit') as enter,
                    mock.patch('text_injector.time.sleep'),
                    mock.patch('text_injector.subprocess.run', return_value=mock.Mock(returncode=0, stdout=b'')),
                ):
                    self.assertTrue(injector.recover_last('paste_last')[0])
                    self.assertEqual(type_text.called, direct)
                    enter.assert_not_called()


if __name__ == '__main__':
    unittest.main()
