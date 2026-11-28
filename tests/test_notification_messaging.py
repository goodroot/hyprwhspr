"""Notification messaging: levels, one alert per subject, recovery episodes, honest results."""
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "lib" / "src"))

import desktop_notify  # noqa: E402
from desktop_notify import NotificationSlots  # noqa: E402
from longform_controller import LongFormController  # noqa: E402
from mic_osd.notification_presenter import NotificationPresenter  # noqa: E402
from text_injector import InjectionOutcome  # noqa: E402

from tests.test_suspend_resume_recovery import _import_main_isolated, patch_app_global  # noqa: E402
from tests.text_injector_helpers import make_injector  # noqa: E402


class HeldTimer:
    """A timer that fires only when the test says so."""

    created = []

    def __init__(self, interval, function, args=()):
        self.interval, self.function, self.args = interval, function, args
        self.cancelled = False
        HeldTimer.created.append(self)

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True

    def fire(self):
        self.function(*self.args)


class LevelCase(unittest.TestCase):
    """Sets notification_level for one test and records what reaches the desktop."""

    def setUp(self):
        self.level = 'all'
        desktop_notify.set_level_provider(lambda: self.level)
        self.addCleanup(desktop_notify.set_level_provider, None)
        HeldTimer.created = []
        ids = iter(range(100, 1000))
        # A replaced notification keeps its id, as on a real server
        send = mock.patch.object(
            desktop_notify, 'send_notification_with_id',
            side_effect=lambda *a, **k: k.get('replaces_id') or next(ids))
        self.send = send.start()
        self.addCleanup(send.stop)
        close = mock.patch.object(desktop_notify, 'close_notification')
        self.close = close.start()
        self.addCleanup(close.stop)

    def sent(self):
        return [call.args[1] for call in self.send.call_args_list]


class NotificationLevelTests(LevelCase):
    def test_each_level_admits_its_kinds(self):
        expected = {
            'all': {'lifecycle': True, 'result': True, 'action': True},
            'action_required': {'lifecycle': False, 'result': True, 'action': True},
            'off': {'lifecycle': False, 'result': False, 'action': False},
        }
        for level, kinds in expected.items():
            self.level = level
            for kind, admitted in kinds.items():
                with self.subTest(level=level, kind=kind):
                    self.assertEqual(desktop_notify.allowed(kind), admitted)

    def test_unknown_level_falls_back_to_all(self):
        self.level = 'loud'
        self.assertTrue(desktop_notify.allowed('lifecycle'))

    def test_notify_and_slots_obey_the_level(self):
        slots = NotificationSlots(timer_factory=HeldTimer)
        with mock.patch.object(desktop_notify.shutil, 'which', return_value='/usr/bin/notify-send'):
            for level, shown in (('all', 2), ('action_required', 0), ('off', 0)):
                with self.subTest(level=level):
                    self.level = level
                    self.send.reset_mock()
                    desktop_notify.notify('hyprwhspr', 'loaded', kind='lifecycle')
                    slots.show('model', 'loaded', kind='lifecycle')
                    self.assertEqual(self.send.call_count, shown)
            self.level = 'action_required'
            self.send.reset_mock()
            desktop_notify.notify('hyprwhspr', 'broken')            # a problem by default
            self.assertTrue(slots.show('model', 'done', kind='result'))
            self.assertEqual(self.send.call_count, 2)

    def test_recording_status_ignores_the_level(self):
        # mic_osd_enabled alone decides the status bubble
        self.level = 'off'
        presenter = NotificationPresenter(active_timeout_ms=5000)
        with mock.patch.object(NotificationPresenter, 'is_available', return_value=True), \
             mock.patch('mic_osd.notification_presenter.send_notification_with_id',
                        return_value=7) as send:
            presenter.show()
            presenter.set_state('processing')
        self.assertEqual(send.call_count, 2)


class NotificationSlotTests(LevelCase):
    def setUp(self):
        super().setUp()
        self.slots = NotificationSlots(timer_factory=HeldTimer)

    def test_a_new_message_replaces_the_slots_last(self):
        self.slots.show('model', 'loading', kind='lifecycle')
        self.slots.show('model', 'ready', kind='lifecycle')
        first, second = self.send.call_args_list
        self.assertIsNone(first.kwargs['replaces_id'])
        self.assertEqual(second.kwargs['replaces_id'], 100)

    def test_slots_do_not_replace_each_other(self):
        self.slots.show('model', 'loading')
        self.slots.show('recovery', 'recovering')
        self.assertIsNone(self.send.call_args.kwargs['replaces_id'])

    def test_an_old_messages_cleanup_cannot_close_its_replacement(self):
        self.slots.show('model', 'loading', kind='lifecycle')
        stale = HeldTimer.created[-1]
        self.slots.show('model', 'failed', retained=True)
        self.assertTrue(stale.cancelled)
        stale.fire()                                   # a timer that had already started running
        self.close.assert_not_called()
        self.assertTrue(self.slots.active('model'))

    def test_a_temporary_message_closes_itself(self):
        self.slots.show('model', 'ready', kind='lifecycle', timeout_ms=5000)
        HeldTimer.created[-1].fire()
        self.close.assert_called_once_with(100)
        self.assertFalse(self.slots.active('model'))

    def test_a_retained_message_stays_until_cleared(self):
        self.slots.show('delivery', 'not delivered', retained=True)
        sent = self.send.call_args.kwargs
        self.assertEqual((sent['urgency'], sent['timeout_ms'], sent['transient']), ('critical', 0, False))
        self.assertEqual(HeldTimer.created, [])        # nothing is scheduled to close it
        self.slots.clear('delivery')
        self.close.assert_called_once_with(100)
        self.slots.clear('delivery')                   # already gone
        self.close.assert_called_once_with(100)

    def test_a_hidden_message_still_retires_what_it_replaces(self):
        self.slots.show('recovery', 'not recovered', retained=True)
        self.level = 'action_required'
        self.assertFalse(self.slots.show('recovery', 'recovered', kind='lifecycle'))
        self.close.assert_called_once_with(100)
        self.assertEqual(self.sent(), ['not recovered'])


class PresenterDetailTests(unittest.TestCase):
    def setUp(self):
        self.presenter = NotificationPresenter(active_timeout_ms=5000)
        available = mock.patch.object(NotificationPresenter, 'is_available', return_value=True)
        available.start()
        self.addCleanup(available.stop)
        send = mock.patch('mic_osd.notification_presenter.send_notification_with_id', return_value=7)
        self.send = send.start()
        self.addCleanup(send.stop)
        close = mock.patch('mic_osd.notification_presenter.close_notification')
        self.close = close.start()
        self.addCleanup(close.stop)

    def last(self):
        call = self.send.call_args
        return call.args[1], call.kwargs['timeout_ms'], call.kwargs['replaces_id']

    def test_a_wait_rewords_the_bubble_and_gives_it_back(self):
        self.presenter.show()
        self.assertTrue(self.presenter.set_detail('loading_recording'))
        self.assertEqual(self.last(), ('Model loading — keep speaking', 5000, 7))
        self.presenter.clear_detail()
        self.assertEqual(self.last(), ('🎤 Recording…', 5000, 7))

    def test_a_wait_under_processing_stays_until_replaced(self):
        self.presenter.show()
        self.presenter.set_state('processing')
        self.presenter.set_detail('loading_captured')
        self.assertEqual(self.last(), ('Model loading — audio captured', 0, 7))
        self.presenter.clear_detail()
        self.assertEqual(self.last(), ('Transcribing…', 0, 7))
        self.presenter.set_state('success')            # every processing state ends in a result
        self.assertEqual(self.last()[1], NotificationPresenter._TRANSIENT_TIMEOUT_MS)

    def test_a_wait_needs_a_bubble_to_update(self):
        self.assertFalse(self.presenter.set_detail('loading_recording'))
        self.send.assert_not_called()

    def test_results_name_what_happened(self):
        for detail, text in (('consumed', 'Handled by hook'),
                             ('clipboard', 'Text copied — paste to insert'),
                             ('empty', 'No speech recognized — try recording again'),
                             ('done', '✓ Done')):
            with self.subTest(detail=detail):
                self.assertTrue(self.presenter.set_detail(detail))
                body, timeout, _ = self.last()
                self.assertEqual(body, text)
                self.assertNotIn('Inserted', body)
                # What the user must act on stays as long as an alert would
                held = detail in ('clipboard', 'empty')
                self.assertEqual(timeout, 5000 if held else NotificationPresenter._TRANSIENT_TIMEOUT_MS)

    def test_clearing_a_wait_leaves_a_result_alone(self):
        self.presenter.set_detail('clipboard')
        self.send.reset_mock()
        self.presenter.clear_detail()                  # a late wake thread
        self.send.assert_not_called()


class AppCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()
        cls.app_cls = cls.main.hyprwhsprApp

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name)
        for name, value in (('MODEL_UNLOADED_FILE', self.temp / 'model_unloaded'),
                            ('RECOVERY_RESULT_FILE', self.temp / 'recovery_result'),
                            ('RECOVERY_REQUESTED_FILE', self.temp / 'recovery_requested'),
                            ('MIC_ZERO_VOLUME_FILE', self.temp / 'mic_zero_volume'),
                            ('VISUALIZER_STATE_FILE', self.temp / 'visualizer_state'),
                            ('log', mock.Mock())):
            patcher = patch_app_global(self.app_cls, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _app(self, **settings):
        app = self.app_cls.__new__(self.app_cls)
        app.config = SimpleNamespace(
            get_setting=lambda key, default=None: settings.get(key, default),
            get_hallucination_markers=lambda: ['Thank you.'])
        app._recording_lock = threading.Lock()
        app._recording_finalizing = threading.Event()
        app.is_recording = app.is_processing = False
        app._file_transcription_active = app._longform_active = False
        app._model_operation_active = app._model_initializing = False
        app._backend_init_failed = False
        app._notify_when_ready = False
        app._model_idle_since = None
        app._model_idle_unloaded = app._model_waking = False
        app._model_wake_done = threading.Event()
        app._model_wake_done.set()
        app._mic_osd_runner = None
        app._cancel_pending_hide = False
        app._cancel_pending_hide_lock = threading.Lock()
        app._mic_state_lock = threading.Lock()
        app._mic_disconnected = False
        app._hotplug_lock = threading.Lock()
        app._last_hotplug_add_time = app._last_hotplug_remove_time = float('-inf')
        app._startup_time, app._startup_grace_period = 0.0, 0.0
        app._recovery_result_lock = threading.Lock()
        app._background_recovery_needed = threading.Event()
        app._background_recovery_stop = threading.Event()
        app._background_recovery_thread = None
        app.recovery_attempted = threading.Event()
        app._notify_user = mock.Mock()
        app._clear_alert = mock.Mock()
        app.whisper_manager = SimpleNamespace(
            is_ready=lambda: True,
            active_backend_is_local=lambda: True,
            unload_model=mock.Mock(return_value=True),
            reload_model=mock.Mock(return_value=True),
            reinitialize_after_resume=mock.Mock(return_value=True),
            _model_manually_unloaded=False,
        )
        app.audio_capture = mock.Mock()
        app.audio_capture.recovery_lock = threading.Lock()
        app.audio_capture.recovery_in_progress = False
        app.audio_capture.recover_audio_capture.return_value = True
        app.audio_manager = mock.Mock()
        return app

    @staticmethod
    def alerts(app):
        """(message, kind, slot, retained) of every alert, in order."""
        return [(c.args[1], c.kwargs.get('kind', 'action'), c.kwargs.get('slot'),
                 c.kwargs.get('retained', False)) for c in app._notify_user.call_args_list]

    @classmethod
    def messages(cls, app):
        return [alert[0] for alert in cls.alerts(app)]


class HeadlessVisualizerTests(AppCase):
    def test_notification_results_update_bar_state_without_rewording_the_bubble(self):
        app = self._app()
        presenter = NotificationPresenter()
        app._mic_osd_runner = presenter
        state = self.temp / 'visualizer_state'
        thread = mock.Mock()
        with (mock.patch.object(presenter, 'is_available', return_value=True),
              mock.patch.object(presenter, '_send') as send,
              mock.patch.object(presenter, '_close'),
              patch_app_global(self.app_cls, 'threading', SimpleNamespace(Thread=thread)),
              patch_app_global(self.app_cls, 'time', SimpleNamespace(sleep=lambda _: None))):
            app._show_mic_osd()
            self.assertEqual(state.read_text(), 'recording')
            presenter._nid = 1
            app._set_visualizer_state('processing')
            self.assertEqual(state.read_text(), 'processing')
            send.reset_mock()
            app._show_result_and_hide(True, detail='clipboard')
            self.assertEqual(state.read_text(), 'success')
            send.assert_called_once_with('Text copied — paste to insert', 5000)
            thread.call_args.kwargs['target']()
            self.assertFalse(state.exists())

    def test_unavailable_presenter_and_failed_hide_still_publish_and_clear(self):
        app = self._app()
        app._mic_osd_runner = mock.Mock()
        app._mic_osd_runner.is_available.return_value = False
        app._mic_osd_runner.hide.side_effect = RuntimeError('presenter unavailable')
        state = self.temp / 'new-runtime' / 'visualizer_state'
        with patch_app_global(self.app_cls, 'VISUALIZER_STATE_FILE', state):
            app._show_mic_osd()
            self.assertEqual(state.read_text(), 'recording')
            app._hide_mic_osd()
            self.assertFalse(state.exists())

    def test_visualizer_state_has_the_same_lifecycle_without_an_overlay(self):
        app = self._app(mic_osd_enabled=False)
        state = self.temp / 'visualizer_state'
        app._show_mic_osd()
        self.assertEqual(state.read_text(), 'recording')
        app._set_visualizer_state('processing')
        self.assertEqual(state.read_text(), 'processing')
        thread = mock.Mock()
        with (patch_app_global(self.app_cls, 'threading', SimpleNamespace(Thread=thread)),
              patch_app_global(self.app_cls, 'time', SimpleNamespace(sleep=lambda _: None))):
            app._show_result_and_hide(True)
            self.assertEqual(state.read_text(), 'success')
            thread.call_args.kwargs['target']()
        self.assertFalse(state.exists())


class NotifyUserTests(AppCase, LevelCase):
    """_notify_user itself: the path every service alert takes."""

    def setUp(self):
        AppCase.setUp(self)
        LevelCase.setUp(self)

    def _real_app(self, **settings):
        app = self._app(**settings)
        del app._notify_user, app._clear_alert
        app._notification_slots = NotificationSlots(timer_factory=HeldTimer)
        return app

    def test_levels_are_independent_of_the_mic_osd(self):
        for osd in (True, False):
            for level, expected in (('all', ['unloaded', 'model ready', 'failed']),
                                    ('action_required', ['model ready', 'failed']),
                                    ('off', [])):
                with self.subTest(mic_osd_enabled=osd, level=level):
                    self.level = level
                    self.send.reset_mock()
                    app = self._real_app(mic_osd_enabled=osd)
                    app._notify_user('hyprwhspr', 'unloaded', kind='lifecycle', slot='model')
                    app._notify_user('hyprwhspr', 'model ready', kind='result', slot='model')
                    app._notify_user('hyprwhspr', 'failed', kind='action', slot='model', retained=True)
                    self.assertEqual(self.sent(), expected)

    def test_configured_timeout_applies_to_temporary_alerts(self):
        app = self._real_app(notification_timeout_ms=9000)
        app._notify_user('hyprwhspr', 'ready', kind='result', slot='model')
        self.assertEqual(self.send.call_args.kwargs['timeout_ms'], 9000)

    def test_mic_failure_is_one_replaced_alert_cleared_by_the_next_start(self):
        app = self._real_app()
        app._error_log_lock = threading.Lock()
        app._last_mic_error_log_time, app._last_mic_error_message = 0.0, None
        app._notify_zero_volume('Microphone not responding')
        app._notify_zero_volume('Microphone unavailable')
        self.assertEqual(self.send.call_args.kwargs['replaces_id'], 100)
        app._clear_zero_volume_signal()
        self.close.assert_called_once_with(100)


class ModelLifecycleTests(AppCase):
    def test_model_repair_detail_is_retained_until_loading_succeeds(self):
        app = self._app()
        repair = "Speech model 'base' is damaged. Run: hyprwhspr model download base"
        app.whisper_manager.model_load_error = repair
        app._report_backend_load_failed()
        self.assertEqual(self.alerts(app), [(repair, 'action', 'model', True)])
        app.whisper_manager.model_load_error = None
        app._announce_backend_ready(True)
        app._clear_alert.assert_called_once_with('model')
        self.assertIsNone(app._model_failure_message)
        self.assertEqual(self.messages(app)[-1], 'Model ready')

    def _missing_modules(self, *names):
        patcher = patch_app_global(self.app_cls, 'missing_imports', mock.Mock(return_value=names))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_missing_modules_are_the_only_startup_banner(self):
        app = self._app()
        app.whisper_manager.model_load_error = "Speech model 'base' is not downloaded yet."
        self._missing_modules('pywhispercpp')
        app._report_missing_dependencies()
        app._report_backend_load_failed()
        self.assertEqual(len(self.alerts(app)), 1)
        self.assertIn('Missing Python modules: pywhispercpp', self.messages(app)[0])

    def test_model_alerts_resume_once_modules_are_installed(self):
        app = self._app()
        repair = "Speech model 'base' is not downloaded yet."
        app.whisper_manager.model_load_error = repair
        self._missing_modules('pywhispercpp')
        app._report_missing_dependencies()
        self._missing_modules()
        app._report_backend_load_failed()
        self.assertEqual(self.alerts(app)[-1], (repair, 'action', 'model', True))

    def test_manual_reload_failure_uses_the_same_repair_detail(self):
        app = self._app()
        repair = "Speech model is missing. Run: hyprwhspr model download small.en"
        app.whisper_manager.model_load_error = repair
        app.whisper_manager.reload_model.return_value = False
        app._run_model_operation('reload')
        self.assertEqual(self.alerts(app)[-1], (repair, 'action', 'model', True))

    def test_automatic_and_manual_unload_say_what_happens_next(self):
        app = self._app()
        app._handle_model_operation('unload', automatic=True)
        app._handle_model_operation('unload')
        self.assertEqual(self.alerts(app), [
            ('Model unloaded after inactivity — reloads on next recording', 'lifecycle', 'model', False),
            ('Model unloaded — run: hyprwhspr model reload', 'result', 'model', False),
        ])

    def test_manual_reload_reports_ready_only_when_it_loaded(self):
        app = self._app()
        app.whisper_manager.is_ready = lambda: False
        app._handle_model_operation('reload')
        self.assertEqual(self.alerts(app), [('Model ready', 'result', 'model', False)])

        app = self._app()
        app.whisper_manager.is_ready = lambda: False
        app.whisper_manager.reload_model.return_value = False
        app._handle_model_operation('reload')
        (message, kind, slot, retained), = self.alerts(app)
        self.assertIn('Model failed to load', message)
        self.assertNotIn('ready', message.lower())
        self.assertEqual((kind, slot, retained), ('action', 'model', True))

    def test_one_alert_follows_a_wake_from_loading_to_ready(self):
        # An overlay (or no indicator) has no text: the model slot carries it
        app = self._app()
        app._announce_model_loading('recording')
        app._announce_model_loading('captured')
        app._model_operation_active = True
        app._run_model_operation('reload', quiet=True)
        self.assertEqual(self.alerts(app), [
            ('Model loading — keep speaking', 'lifecycle', 'model', False),
            ('Model loading — audio captured', 'lifecycle', 'model', False),
            ('Model ready', 'lifecycle', 'model', False),
        ])

    def test_a_quiet_wake_nobody_was_told_about_stays_quiet(self):
        app = self._app()
        app._model_operation_active = True
        app._run_model_operation('reload', quiet=True)     # a file transcription's wake
        app._notify_user.assert_not_called()

    def test_the_status_bubble_carries_the_wait_instead_of_an_alert(self):
        app = self._app()
        app._mic_osd_runner = mock.Mock()
        app._mic_osd_runner.set_detail.return_value = True
        app._announce_model_loading('recording')
        app._model_operation_active = True
        app._run_model_operation('reload', quiet=True)
        app._mic_osd_runner.set_detail.assert_called_once_with('loading_recording')
        app._mic_osd_runner.clear_detail.assert_called_once_with()
        app._notify_user.assert_not_called()

    def test_a_finished_recording_waiting_on_the_load_says_its_audio_is_safe(self):
        app = self._app()
        app._model_idle_unloaded = True
        app._model_wake_done.clear()
        app._MODEL_WAKE_TIMEOUT = 0.01
        self.assertFalse(app._await_model_wake(announce=True))
        self.assertEqual(self.messages(app), ['Model loading — audio captured'])

        app._notify_user.reset_mock()
        self.assertFalse(app._await_model_wake())           # a file request: the CLI is its feedback
        app._notify_user.assert_not_called()

    def test_a_quiet_retry_that_loads_retires_the_failure_alert(self):
        app = self._app()
        app._model_idle_unloaded = True
        app.whisper_manager.reload_model.return_value = False
        app._run_model_operation('reload', quiet=True)      # the wake fails
        self.assertTrue(self.alerts(app)[-1][3])
        app._clear_alert.assert_not_called()

        app.whisper_manager.reload_model.return_value = True
        app._run_model_operation('reload', quiet=True)      # the next press retries it
        app._clear_alert.assert_called_once_with('model')
        self.assertEqual(len(self.alerts(app)), 1)          # and nothing else is said

        app._run_model_operation('reload', quiet=True)      # a later wake has nothing to retire
        app._clear_alert.assert_called_once_with('model')

    def test_a_wait_notice_put_up_after_the_load_finished_is_retired(self):
        # Overlay: the alert went out after "ready" had nothing to replace
        app = self._app()
        app._announce_model_loading('recording')
        app._settle_model_wait()
        self.assertEqual(self.messages(app), ['Model loading — keep speaking', 'Model ready'])
        self.assertFalse(app._model_loading_alerted)

        # Bubble: the wait wording is dropped
        app = self._app()
        app._mic_osd_runner = mock.Mock()
        app._mic_osd_runner.set_detail.return_value = True
        app._announce_model_loading('recording')
        app._settle_model_wait()
        app._mic_osd_runner.clear_detail.assert_called_once_with()
        app._notify_user.assert_not_called()

    def test_a_wait_notice_put_up_after_the_load_failed_gives_the_failure_back(self):
        app = self._app()
        app._model_idle_unloaded = True
        app.whisper_manager.reload_model.return_value = False
        app._run_model_operation('reload', quiet=True)
        failure = self.alerts(app)[-1]
        app._announce_model_loading('recording')            # replaces it in the model slot
        app._settle_model_wait()
        self.assertEqual(self.alerts(app)[-1], failure)
        self.assertNotIn('Model ready', self.messages(app))

    def test_a_finished_recording_settles_a_wait_the_load_outran(self):
        app = self._app()
        app._model_wake_done = mock.Mock()
        app._model_wake_done.is_set.side_effect = [False, True]   # ends as the notice goes up
        app._model_wake_done.wait.return_value = True
        self.assertTrue(app._await_model_wake(announce=True))
        self.assertEqual(self.messages(app), ['Model loading — audio captured', 'Model ready'])

    def test_a_quick_startup_retry_retires_the_failure_nobody_waits_on(self):
        app = self._app()
        app._backend_init_lock = threading.Lock()
        app.whisper_manager.initialize = mock.Mock(return_value=True)
        app._report_backend_load_failed()
        thread = mock.Mock()
        with patch_app_global(self.app_cls, 'threading', SimpleNamespace(Thread=thread)):
            app._start_backend_init_background()
        thread.call_args.kwargs['target']()
        app._clear_alert.assert_called_once_with('model')
        self.assertEqual(len(self.alerts(app)), 1)

    def test_ready_after_a_refused_press_is_a_result(self):
        for requested, kind in ((True, 'result'), (False, 'lifecycle')):
            with self.subTest(requested=requested):
                app = self._app()
                app._announce_backend_ready(requested)
                self.assertEqual(self.alerts(app), [('Model ready', kind, 'model', False)])


class KeptAudioTests(AppCase):
    def test_saved_files_have_distinct_alerts_not_cleared_by_success(self):
        app = self._app()
        app.audio_capture.save_audio_to_wav.return_value = True
        with patch_app_global(self.app_cls, 'DEBUG_RECORDINGS_DIR', self.temp / 'recordings'):
            app._keep_unsent_audio([0.1])
            app._keep_unsent_audio([0.2])
        slots = [call.kwargs['slot'] for call in app._notify_user.call_args_list]
        self.assertEqual(len(set(slots)), 2)
        app._mark_delivery_healthy()
        cleared = [call.args[0] for call in app._clear_alert.call_args_list]
        self.assertFalse(set(slots).intersection(cleared))

    def test_kept_audio_names_the_command_that_recovers_it(self):
        app = self._app()
        app.audio_capture.save_audio_to_wav.return_value = True
        with patch_app_global(self.app_cls, 'DEBUG_RECORDINGS_DIR', self.temp / 'recordings'):
            app._keep_unsent_audio([0.1])
        (message, kind, slot, retained), = self.alerts(app)
        self.assertTrue(message.startswith('Transcription failed — audio kept\nRun: hyprwhspr transcribe '))
        self.assertIn(str(self.temp / 'recordings' / 'unsent'), message)
        self.assertTrue(slot.startswith('audio:'))
        self.assertEqual((kind, retained), ('action', True))

    def test_lost_audio_keeps_its_explicit_warning(self):
        app = self._app()
        app.audio_capture.save_audio_to_wav.return_value = False
        with patch_app_global(self.app_cls, 'DEBUG_RECORDINGS_DIR', self.temp / 'recordings'):
            app._keep_unsent_audio([0.1])
        self.assertEqual(self.alerts(app), [('Model failed to load — dictation lost', 'action', None, True)])


class RecoveryEpisodeTests(AppCase):
    """Recovery is announced by the service; no Waybar script is involved."""

    def test_resume_initial_success_is_silent(self):
        app = self._app()
        app._resync_shortcut_keyboards = mock.Mock()
        clock = SimpleNamespace(sleep=lambda s: None, time=lambda: 1.0)
        with patch_app_global(self.app_cls, 'time', clock):
            app._on_system_resume()
        app._notify_user.assert_not_called()
        self.assertIsNone(app._recovery_announced())

    def test_resume_failed_initial_attempt_announces_the_failed_component(self):
        for audio_ok, local, message in (
                (False, True, 'Recovering microphone…'),
                (True, True, 'Reloading model…'),
                (True, False, 'Reconnecting to transcription service…')):
            with self.subTest(message=message):
                app = self._app()
                app._resync_shortcut_keyboards = mock.Mock()
                app.audio_capture.recover_audio_capture.return_value = audio_ok
                app.whisper_manager.reinitialize_after_resume.return_value = False
                app.whisper_manager.active_backend_is_local = lambda: local
                app._recovery_episode_guard = threading.Lock()
                clock = SimpleNamespace(sleep=lambda s: None, time=lambda: 1.0)
                with (patch_app_global(self.app_cls, 'time', clock),
                      patch_app_global(self.app_cls, 'threading', mock.Mock())):
                    app._on_system_resume()
                self.assertEqual(self.messages(app), [message])

    def test_an_attempt_from_a_finished_episode_cannot_report_on_the_next(self):
        app = self._app()
        old = app._recovery_start('microphone')
        app._recovery_confirmed(token=old)                  # healed quietly
        new = app._recovery_start('microphone')
        app._recovery_begin('microphone', token=new)
        app._recovery_confirmed(token=old)                  # a straggler from the first
        app._recovery_exhausted('microphone', token=old)
        self.assertEqual(self.messages(app), ['Recovering microphone…'])
        app._recovery_confirmed(token=new)
        self.assertEqual(self.messages(app)[-1], 'Microphone recovered')

    def test_a_new_attempt_joins_the_announced_episode(self):
        # A replug while the resume retries are announced: its success is theirs
        app = self._app()
        resume = app._recovery_start('microphone')
        app._recovery_begin('microphone', token=resume)
        hotplug = app._recovery_start('microphone')
        self.assertIs(hotplug, resume)
        app._recovery_confirmed(token=hotplug)
        self.assertEqual(self.messages(app), ['Recovering microphone…', 'Microphone recovered'])

    def test_a_new_attempt_keeps_a_retained_failure_until_it_succeeds(self):
        app = self._app()
        app._recovery_exhausted('microphone')
        token = app._recovery_start('microphone')
        app._clear_alert.assert_not_called()                # still unresolved
        app._recovery_confirmed(token=token)
        self.assertEqual(self.alerts(app)[-1], ('Microphone recovered', 'lifecycle', 'recovery', False))

    def test_a_failed_hotplug_leaves_the_outcome_to_the_running_retries(self):
        app = self._app()
        app.audio_capture.recover_audio_capture.return_value = False
        resume = app._recovery_start('microphone')
        app._recovery_begin('microphone', token=resume)
        app._background_recovery_needed.set()
        app._background_recovery_thread = SimpleNamespace(is_alive=lambda: True)
        clock = SimpleNamespace(monotonic=lambda: 100.0, sleep=lambda s: None, time=lambda: 1.0)
        with patch_app_global(self.app_cls, 'time', clock):
            app._on_audio_device_added({'ID_MODEL': 'Mic'})
            self.assertEqual(self.messages(app), ['Recovering microphone…'])   # announced once
            app._background_recovery_retry(episode=resume)
        self.assertEqual(self.alerts(app)[-1], (
            'Microphone not recovered — replug it, then record again', 'action', 'recovery', True))

    def test_the_realtime_link_and_the_microphone_recover_independently(self):
        app = self._app()
        mic = app._recovery_start('microphone')
        app._recovery_begin('microphone', token=mic)
        app._on_realtime_connection('reconnecting')
        app._on_realtime_connection('retrying')
        app._recovery_exhausted('microphone', token=mic)    # still the microphone's to report
        app._on_realtime_connection('reconnected')
        self.assertEqual(self.alerts(app), [
            ('Recovering microphone…', 'lifecycle', 'recovery', False),
            ('Reconnecting to transcription service…', 'lifecycle', 'connection', False),
            ('Microphone not recovered — replug it, then record again', 'action', 'recovery', True),
            ('Transcription service reconnected', 'lifecycle', 'connection', False),
        ])
        self.assertEqual(app._recovery_announced(), 'microphone')

    def test_the_command_a_failed_model_recovery_names_resolves_it(self):
        for ready in (False, True):                         # a reload, or "already loaded"
            with self.subTest(ready=ready):
                app = self._app()
                app._recovery_exhausted('model')
                app.whisper_manager.is_ready = lambda: ready
                app._handle_model_operation('reload')
                app._clear_alert.assert_called_with('recovery')
                self.assertIsNone(app._recovery_announced())
                self.assertEqual(self.alerts(app)[-1][2], 'model')

    def test_a_model_reload_leaves_a_microphone_failure_up(self):
        app = self._app()
        app._recovery_exhausted('microphone')
        app.whisper_manager.is_ready = lambda: False
        app._handle_model_operation('reload')
        app._clear_alert.assert_not_called()
        self.assertEqual(app._recovery_announced(), 'microphone')

    def test_hidden_recovery_has_no_unsolicited_success_after_level_change(self):
        app = self._app()
        with mock.patch.object(desktop_notify, 'allowed', return_value=False):
            token = app._recovery_start('microphone')
            app._recovery_begin('microphone', token=token)
        app._notify_user.reset_mock()
        app._recovery_confirmed(token=token)
        app._notify_user.assert_not_called()
        app._clear_alert.assert_called_with('recovery')

    def test_first_realtime_reconnect_success_is_silent(self):
        app = self._app()
        app._on_realtime_connection('reconnecting')
        app._on_realtime_connection('reconnected')
        app._notify_user.assert_not_called()

    def test_a_blocked_press_gets_a_result_when_recovery_finishes(self):
        app = self._app()
        token = app._recovery_start('service', track='link')
        app._recovery_begin('service', kind='result', repeat=True, track='link')
        app._recovery_confirmed(token=token, track='link')
        self.assertEqual(self.alerts(app)[-1], (
            'Transcription service reconnected', 'result', 'connection', False))

    def test_an_episode_is_announced_once_however_often_it_retries(self):
        app = self._app()
        for _ in range(3):
            app._recovery_begin('microphone')
        self.assertEqual(self.alerts(app), [('Recovering microphone…', 'lifecycle', 'recovery', False)])

    def test_a_refused_press_is_answered_every_time(self):
        app = self._app()
        app._recovery_begin('service', kind='result', repeat=True)
        app._recovery_begin('service', kind='result', repeat=True)
        self.assertEqual(self.alerts(app),
                         [('Reconnecting to transcription service…', 'result', 'recovery', False)] * 2)

    def test_success_is_reported_only_for_an_announced_episode(self):
        app = self._app()
        app._recovery_confirmed()                          # nothing was announced
        app._notify_user.assert_not_called()

        app._recovery_begin('microphone')
        app._recovery_confirmed()
        app._recovery_confirmed()                          # the episode is over
        self.assertEqual(self.messages(app), ['Recovering microphone…', 'Microphone recovered'])

    def test_exhaustion_is_retained_and_cleared_by_confirmed_recovery(self):
        app = self._app()
        app._recovery_begin('microphone')
        app._recovery_exhausted('microphone')
        self.assertEqual(self.alerts(app)[-1], (
            'Microphone not recovered — replug it, then record again', 'action', 'recovery', True))
        app._recovery_confirmed()                          # same slot: replaces the retained alert
        self.assertEqual(self.alerts(app)[-1], ('Microphone recovered', 'lifecycle', 'recovery', False))

    def test_each_failed_component_gets_its_own_next_step(self):
        expected = {'microphone': 'replug', 'model': 'hyprwhspr model reload', 'service': 'check network'}
        for component, step in expected.items():
            with self.subTest(component=component):
                app = self._app()
                app._recovery_exhausted(component)
                message, _, slot, retained = self.alerts(app)[-1]
                self.assertIn(step, message)
                self.assertEqual((slot, retained), ('recovery', True))

    def test_a_disconnect_that_healed_unannounced_closes_quietly(self):
        app = self._app()
        with patch_app_global(self.app_cls, 'time', SimpleNamespace(monotonic=lambda: 100.0)):
            app._on_audio_device_removed({'ID_MODEL': 'Mic'})
        self.assertEqual(self.alerts(app), [
            ('Microphone disconnected — reconnect it to record', 'action', 'recovery', False)])
        app._recovery_confirmed()
        self.assertEqual(len(self.alerts(app)), 1)         # no unsolicited "recovered"
        app._clear_alert.assert_called_with('recovery')

    def test_hotplug_initial_success_is_silent(self):
        app = self._app()
        clock = SimpleNamespace(monotonic=lambda: 100.0, sleep=lambda s: None, time=lambda: 1.0)
        with patch_app_global(self.app_cls, 'time', clock):
            app._on_audio_device_added({'ID_MODEL': 'Mic'})
        app._notify_user.assert_not_called()

    def test_audio_only_recovery_does_not_confirm_a_failed_backend(self):
        for component in ('model', 'service'):
            for event in ('hotplug', 'pulse_restart'):
                with self.subTest(component=component, event=event):
                    app = self._app()
                    app._recovery_exhausted(component)
                    app._notify_user.reset_mock()
                    app._end_orphaned_recording = mock.Mock()
                    clock = SimpleNamespace(monotonic=lambda: 100.0, sleep=lambda s: None,
                                            time=lambda: 1.0)
                    with patch_app_global(self.app_cls, 'time', clock):
                        if event == 'hotplug':
                            app._on_audio_device_added({'ID_MODEL': 'Mic'})
                        else:
                            app._on_pulse_server_restarted()
                    app._notify_user.assert_not_called()
                    app._clear_alert.assert_not_called()
                    self.assertEqual(app._recovery_announced(), component)

    def test_hotplug_failure_with_nothing_retrying_is_final(self):
        app = self._app()
        app.audio_capture.recover_audio_capture.return_value = False
        clock = SimpleNamespace(monotonic=lambda: 100.0, sleep=lambda s: None, time=lambda: 1.0)
        with patch_app_global(self.app_cls, 'time', clock):
            app._on_audio_device_added({'ID_MODEL': 'Mic'})
        self.assertEqual(self.alerts(app)[-1][::3], (
            'Microphone not recovered — replug it, then record again', True))

    def test_background_retries_stay_quiet_until_the_outcome(self):
        clock = SimpleNamespace(sleep=lambda s: None, time=lambda: 1.0, monotonic=lambda: 1.0)
        cases = (
            # (audio recovers, backend recovers, backend is local, final alert)
            (False, True, True, 'Microphone not recovered — replug it, then record again'),
            (True, False, True, 'Model not recovered — run: hyprwhspr model reload'),
            (True, False, False, 'Transcription service unreachable — check network, then record again'),
        )
        for audio_ok, backend_ok, local, final in cases:
            with self.subTest(final=final):
                app = self._app()
                app.audio_capture.recover_audio_capture.return_value = audio_ok
                app.whisper_manager.reinitialize_after_resume.return_value = backend_ok
                app.whisper_manager.active_backend_is_local = lambda local=local: local
                app._recovery_begin('microphone')              # the resume announced it
                app._background_recovery_needed.set()
                with patch_app_global(self.app_cls, 'time', clock):
                    app._background_recovery_retry()
                self.assertEqual(app.audio_capture.recover_audio_capture.call_count, 6)
                self.assertEqual(self.messages(app), ['Recovering microphone…', final])
                self.assertTrue(self.alerts(app)[-1][3])

    def test_background_retry_success_confirms_the_episode(self):
        app = self._app()
        app._recovery_begin('microphone')
        app._background_recovery_needed.set()
        clock = SimpleNamespace(sleep=lambda s: None, time=lambda: 1.0, monotonic=lambda: 1.0)
        with patch_app_global(self.app_cls, 'time', clock):
            app._background_recovery_retry()
        self.assertEqual(self.messages(app), ['Recovering microphone…', 'Microphone recovered'])

    def test_a_dropped_realtime_connection_is_one_episode(self):
        app = self._app()
        for state in ('reconnecting', 'retrying', 'reconnected'):
            app._on_realtime_connection(state)
        app._on_realtime_connection('reconnecting')
        app._on_realtime_connection('retrying')
        app._on_realtime_connection('failed')
        self.assertEqual(self.messages(app), [
            'Reconnecting to transcription service…', 'Transcription service reconnected',
            'Reconnecting to transcription service…',
            'Transcription service unreachable — check network, then record again'])

    def test_service_readiness_closes_a_service_episode(self):
        app = self._app()
        app._recovery_begin('service', kind='result', repeat=True, track='link')   # a refused press
        app._announce_backend_ready(True)
        self.assertEqual(self.alerts(app)[-1], (
            'Transcription service reconnected', 'result', 'connection', False))

    def test_a_landed_dictation_resolves_what_was_reported(self):
        app = self._app()
        app._recovery_exhausted('microphone')
        app._mark_delivery_healthy()
        app._clear_alert.assert_any_call('delivery')
        self.assertEqual(self.messages(app)[-1], 'Microphone recovered')

    def test_a_landed_dictation_resolves_a_failed_link_too(self):
        app = self._app()
        app._on_realtime_connection('reconnecting')
        app._on_realtime_connection('failed')
        app._mark_delivery_healthy()
        self.assertEqual(self.alerts(app)[-1], (
            'Transcription service reconnected', 'lifecycle', 'connection', False))


class ReconnectListenerTests(unittest.TestCase):
    def test_elevenlabs_announces_only_after_first_failed_attempt(self):
        from elevenlabs_realtime_client import ElevenLabsRealtimeClient
        for results, expected in (([True], ['reconnecting', 'reconnected']),
                                  ([False, True], ['reconnecting', 'retrying', 'reconnected']),
                                  ([False], ['reconnecting', 'failed'])):
            with self.subTest(results=results):
                client = ElevenLabsRealtimeClient.__new__(ElevenLabsRealtimeClient)
                client._reconnect_lock = threading.Lock()
                client._reconnecting = False
                client._should_stay_alive = True
                client.reconnect_attempts = 0
                client.max_reconnect_attempts = len(results)
                client.reconnect_delays = [0]
                client._connect_internal = mock.Mock(side_effect=results)
                client.connection_listener = mock.Mock()
                client._attempt_reconnect()
                self.assertEqual([c.args[0] for c in client.connection_listener.call_args_list], expected)

    def test_a_reconnect_loop_reports_once_per_outcome(self):
        import realtime_base
        client = realtime_base.WebSocketRealtimeClientBase.__new__(realtime_base.WebSocketRealtimeClientBase)
        client._reconnect_lock = threading.Lock()
        client._stop_event = mock.Mock()
        client._stop_event.is_set.return_value = False
        client._stop_event.wait.return_value = False
        client.lock = threading.Lock()
        client._closed, client.receiver_running = False, True
        client.max_reconnect_attempts, client.reconnect_delays = 3, [0]
        client._log = mock.Mock()
        for results, expected in (([True], ['reconnecting', 'reconnected']),
                                  ([False, False, True], ['reconnecting', 'retrying', 'reconnected']),
                                  ([False, False, False], ['reconnecting', 'retrying', 'failed'])):
            with self.subTest(expected=expected):
                client.reconnect_attempts = 0
                client._connect_internal = mock.Mock(side_effect=results)
                client.connection_listener = states = mock.Mock()
                client._attempt_reconnect()
                self.assertEqual([c.args[0] for c in states.call_args_list], expected)


class DeliveryOutcomeTests(unittest.TestCase):
    def test_clipboard_only_delivery_is_its_own_outcome(self):
        injector = make_injector()

        def leave(text, **kwargs):
            injector._leave_on_clipboard(text)
            return True

        with mock.patch.object(injector, '_inject_via_clipboard_and_hotkey', side_effect=leave):
            self.assertEqual(injector.inject_text('hello'), InjectionOutcome.CLIPBOARD)
        with mock.patch.object(injector, '_inject_via_clipboard_and_hotkey', return_value=True):
            self.assertEqual(injector.inject_text('hello'), InjectionOutcome.INJECTED)   # the note is per delivery
        with mock.patch.object(injector, '_inject_via_clipboard_and_hotkey', return_value=False):
            self.assertEqual(injector.inject_text('hello'), InjectionOutcome.FAILED)


class ResultTests(AppCase):
    def _process_app(self, transcript, outcome=None):
        app = self._app()
        app._live_typing = None
        app._current_language_override = None
        app.audio_capture = SimpleNamespace(sample_rate=16000)
        app._recording_control_server = mock.Mock()
        app._recording_control_server.is_trace_capture.return_value = False
        app._recording_control_server.has_capture_subscriber.return_value = False
        app._notify_capture = mock.Mock()
        app._show_result_and_hide = mock.Mock()
        app._clear_mic_osd_preview_text = mock.Mock()
        app.text_injector = mock.Mock()
        app.text_injector.inject_text.return_value = outcome
        app.whisper_manager.transcribe_audio = mock.Mock(return_value=transcript)
        return app

    def test_each_outcome_is_shown_as_what_it_was(self):
        cases = (
            ('words', InjectionOutcome.INJECTED, mock.call(True)),
            ('words', InjectionOutcome.CONSUMED, mock.call(True, detail='consumed')),
            ('words', InjectionOutcome.CLIPBOARD, mock.call(True, detail='clipboard')),
            ('words', InjectionOutcome.SKIPPED, mock.call(True, detail='done')),
            ('words', InjectionOutcome.FAILED, mock.call(False)),
            ('', None, mock.call(False, detail='empty')),
            ('Thank you.', None, mock.call(False, detail='empty')),      # a hallucination marker
        )
        for transcript, outcome, shown in cases:
            with self.subTest(transcript=transcript, outcome=outcome):
                app = self._process_app(transcript, outcome)
                app._process_audio([0.0])
                self.assertEqual(app._show_result_and_hide.call_args_list,
                                 [mock.call(*shown.args, **shown.kwargs, generation=0)])   # processing always ends
                self.assertFalse(app.is_processing)

    def test_a_recording_dropped_behind_a_busy_transcription_ends_its_processing(self):
        # Its stop put "processing" up; the older result is stale and won't end it
        app = self._process_app('words')
        app.is_processing = True
        app._process_audio([0.0])
        app._show_result_and_hide.assert_called_once_with(False)
        app.whisper_manager.transcribe_audio.assert_not_called()
        self.assertTrue(app.is_processing)                  # still the first one's

    def test_processing_ends_when_transcription_raises(self):
        app = self._process_app('words')
        app.whisper_manager.transcribe_audio.side_effect = RuntimeError('boom')
        app._process_audio([0.0])
        app._show_result_and_hide.assert_called_once_with(False, generation=0)

    def test_a_failed_load_ends_processing_and_keeps_the_audio(self):
        app = self._process_app('words')
        app._model_idle_unloaded = True
        app._keep_unsent_audio = mock.Mock()
        app._process_audio([0.0])
        app._keep_unsent_audio.assert_called_once_with([0.0])
        app._show_result_and_hide.assert_called_once_with(False, generation=0)

    def test_failed_delivery_is_a_retained_alert_of_its_own(self):
        app = self._process_app('words', InjectionOutcome.FAILED)
        self.assertEqual(app._inject_text('words'), InjectionOutcome.FAILED)
        self.assertEqual(self.alerts(app), [(
            'Text not delivered — run: hyprwhspr record paste-last or hyprwhspr record copy-last',
            'action', 'delivery', True)])

    def test_text_handed_to_a_capture_client_is_not_called_inserted(self):
        app = self._process_app('words')
        app._recording_control_server.has_capture_subscriber.return_value = True
        self.assertEqual(app._inject_text('words'), InjectionOutcome.CAPTURED)

    def test_hook_and_clipboard_results_never_claim_insertion(self):
        # With a text bubble: the bubble says it and no alert competes
        for detail in ('consumed', 'clipboard', 'empty'):
            with self.subTest(detail=detail, bubble=True):
                app = self._app()
                app._mic_osd_runner = mock.Mock()
                app._mic_osd_runner.set_detail.return_value = True
                with patch_app_global(self.app_cls, 'threading', mock.Mock()):
                    app._show_result_and_hide(detail != 'empty', detail=detail)
                app._mic_osd_runner.set_detail.assert_called_once_with(detail)
                app._mic_osd_runner.set_state.assert_not_called()   # never "Inserted" first
                app._notify_user.assert_not_called()

        # With an overlay or no indicator: what the user must act on is an alert
        expected = {'consumed': [], 'clipboard': ['Text copied — paste to insert'],
                    'empty': ['No speech recognized — try recording again']}
        for detail, messages in expected.items():
            with self.subTest(detail=detail, bubble=False):
                app = self._app()
                with patch_app_global(self.app_cls, 'threading', mock.Mock()):
                    app._show_result_and_hide(detail != 'empty', detail=detail)
                self.assertEqual(self.messages(app), messages)
                self.assertEqual({a[1:3] for a in self.alerts(app)} - {('result', 'result')}, set())

    def test_the_bubble_holds_what_the_user_must_act_on(self):
        # Hiding at the overlay's fade would close the bubble mid-sentence
        for detail, held in (('clipboard', 3.0), ('empty', 3.0), ('consumed', 1.25), (None, 1.25)):
            with self.subTest(detail=detail):
                app = self._app(notification_timeout_ms=3000)
                app._mic_osd_runner = mock.Mock()
                app._mic_osd_runner.set_detail.return_value = True
                slept = []
                thread = mock.Mock()
                with (patch_app_global(self.app_cls, 'threading', SimpleNamespace(Thread=thread)),
                      patch_app_global(self.app_cls, 'time', SimpleNamespace(sleep=slept.append))):
                    app._show_result_and_hide(True, detail=detail)
                    thread.call_args.kwargs['target']()
                self.assertEqual(slept, [held])
                app._mic_osd_runner.hide.assert_called_once_with()

    def test_an_overlay_keeps_its_fade_and_gets_the_alert(self):
        app = self._app(notification_timeout_ms=3000)
        app._mic_osd_runner = mock.Mock(spec=['set_state', 'hide', 'clear_state', 'clear_preview_text'])
        slept = []
        thread = mock.Mock()
        with (patch_app_global(self.app_cls, 'threading', SimpleNamespace(Thread=thread)),
              patch_app_global(self.app_cls, 'time', SimpleNamespace(sleep=slept.append))):
            app._show_result_and_hide(True, detail='clipboard')
            thread.call_args.kwargs['target']()
        self.assertEqual(slept, [1.25])
        app._mic_osd_runner.set_state.assert_called_once_with('success')
        self.assertEqual(self.messages(app), ['Text copied — paste to insert'])

    def _run_hide(self, app, during):
        """Call _show_result_and_hide, run `during` while its hide sleeps, then let it finish."""
        asleep, wake = threading.Event(), threading.Event()

        def sleep(_):
            asleep.set()
            wake.wait(timeout=5)

        with patch_app_global(self.app_cls, 'time', SimpleNamespace(sleep=sleep)):
            before = set(threading.enumerate())
            app._show_result_and_hide(True)
            hider, = set(threading.enumerate()) - before
            self.assertTrue(asleep.wait(timeout=5))
            during()
            wake.set()
            hider.join(timeout=5)

    def test_result_cleanup_hides_its_own_indicator(self):
        app = self._app()
        app._mic_osd_runner = mock.Mock()
        self._run_hide(app, lambda: None)
        app._mic_osd_runner.hide.assert_called_once_with()

    def test_result_cleanup_cannot_close_a_newer_recordings_indicator(self):
        app = self._app()
        app._mic_osd_runner = mock.Mock()

        def newer_recording_starts_and_ends():
            app._show_mic_osd()
            # Its own result re-arms the hide flag; the generation still differs
            with app._cancel_pending_hide_lock:
                app._cancel_pending_hide = False

        self._run_hide(app, newer_recording_starts_and_ends)
        app._mic_osd_runner.hide.assert_not_called()
        self.assertEqual((self.temp / 'visualizer_state').read_text(), 'recording')

    def test_a_late_result_leaves_a_live_recording_alone(self):
        app = self._app()
        app._mic_osd_runner = mock.Mock()
        app.is_recording = True                             # the next recording is under way
        app._show_result_and_hide(True, detail='clipboard')
        app._mic_osd_runner.set_state.assert_not_called()
        app._mic_osd_runner.set_detail.assert_not_called()
        self.assertEqual(self.messages(app), ['Text copied — paste to insert'])

    def test_old_result_cannot_restyle_newer_processing_indicator(self):
        app = self._app()
        app._mic_osd_runner = mock.Mock()
        app._show_mic_osd()
        old = app._osd_generation
        app._show_mic_osd()
        app.is_recording = False
        app._mic_osd_runner.reset_mock()
        app._show_result_and_hide(True, generation=old)
        app._mic_osd_runner.set_state.assert_not_called()
        app._mic_osd_runner.hide.assert_not_called()

    def test_new_recording_waits_until_old_result_update_finishes(self):
        app = self._app()
        app._mic_osd_runner = mock.Mock()
        entered, release, started = threading.Event(), threading.Event(), threading.Event()

        def set_state(state):
            if state == 'success':
                entered.set()
                release.wait(2)

        app._mic_osd_runner.set_state.side_effect = set_state
        result = threading.Thread(target=app._show_result_and_hide, args=(True,))
        result.start()
        self.assertTrue(entered.wait(2))

        def start():
            started.set()
            app._show_mic_osd()

        recording = threading.Thread(target=start)
        recording.start()
        self.assertTrue(started.wait(2))
        release.set()
        result.join(2)
        recording.join(2)
        self.assertFalse(result.is_alive() or recording.is_alive())
        self.assertEqual(app._mic_osd_runner.set_state.call_args.args, ('recording',))


class LongFormAlertTests(unittest.TestCase):
    def test_result_keeps_the_indicator_generation_from_before_transcription(self):
        controller = self._controller()
        generation = [3]
        controller.result_generation = lambda: generation[0]

        def transcribe(*args, **kwargs):
            generation[0] = 4
            return 'hello'

        controller.whisper_manager.transcribe_audio.side_effect = transcribe
        controller.submit()
        controller.show_result_and_hide.assert_called_once_with(True, generation=3)

    def _controller(self, transcript='hello', outcome=InjectionOutcome.INJECTED):
        audio = np.array([0.1, 0.2], dtype=np.float32)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        controller = LongFormController(
            config=SimpleNamespace(get_setting=lambda key, default=None: 0),
            audio_capture=SimpleNamespace(
                sample_rate=16000, stop_recording=mock.Mock(),
                resume_recording=mock.Mock(return_value=False)),
            audio_manager=mock.Mock(),
            whisper_manager=SimpleNamespace(transcribe_audio=mock.Mock(return_value=transcript)),
            inject_text=mock.Mock(return_value=outcome),
            notify_capture=mock.Mock(),
            set_visualizer_state=mock.Mock(),
            show_mic_osd=mock.Mock(),
            hide_mic_osd=mock.Mock(),
            show_result_and_hide=mock.Mock(),
            write_recording_status=mock.Mock(),
            set_processing=mock.Mock(),
            hallucination_markers={'Thank you.'},
            notify_problem=mock.Mock(),
            clear_problem=mock.Mock(),
            state_file=Path(temp.name) / 'longform_state',
        )
        controller.segment_manager = SimpleNamespace(
            concatenate_all=mock.Mock(return_value=audio),
            concatenate_readable=mock.Mock(return_value=audio),
            has_segments=mock.Mock(return_value=True),
            clear_session=mock.Mock())
        controller.state = 'PAUSED'
        return controller

    def test_failures_say_the_audio_is_kept_and_how_to_retry(self):
        cases = (('', 'No speech recognized'), ('Thank you.', 'No speech recognized'),
                 (RuntimeError('boom'), 'Transcription failed'))
        for transcript, opening in cases:
            with self.subTest(transcript=transcript):
                controller = self._controller()
                transcribe = controller.whisper_manager.transcribe_audio
                if isinstance(transcript, Exception):
                    transcribe.side_effect = transcript
                else:
                    transcribe.return_value = transcript
                controller.submit()
                self.assertEqual(controller.state, 'ERROR')
                self.assertIsNotNone(controller.error_audio)            # retry has something to send
                controller.notify_problem.assert_called_once_with(
                    f'{opening} — audio kept, submit to retry or cancel to discard')
                # Processing ended in the error state, not a lingering banner
                self.assertEqual(controller.set_visualizer_state.call_args.args, ('error',))

    def test_a_failed_paste_leaves_the_alert_to_text_delivery(self):
        controller = self._controller(outcome=InjectionOutcome.FAILED)
        controller.submit()
        self.assertEqual(controller.state, 'ERROR')
        controller.notify_problem.assert_not_called()

    def test_a_successful_retry_clears_the_alert(self):
        controller = self._controller(transcript='')
        controller.submit()
        controller.whisper_manager.transcribe_audio.return_value = 'hello'
        controller.submit(retry=True)
        self.assertEqual(controller.state, 'IDLE')
        controller.clear_problem.assert_called_once_with()
        controller.show_result_and_hide.assert_called_once_with(True)

    def test_cancel_clears_the_alert(self):
        controller = self._controller(transcript='')
        controller.submit()
        controller.cancel()
        controller.clear_problem.assert_called_once_with()

    def test_a_result_that_was_not_a_paste_is_worded_as_such(self):
        for outcome, detail in ((InjectionOutcome.CLIPBOARD, 'clipboard'),
                                (InjectionOutcome.CONSUMED, 'consumed')):
            with self.subTest(outcome=outcome):
                controller = self._controller(outcome=outcome)
                controller.submit()
                controller.show_result_and_hide.assert_called_once_with(True, detail=detail)

    def test_a_failed_resume_and_a_failed_save_are_reported(self):
        controller = self._controller()
        controller.resume_recording()
        controller.notify_problem.assert_called_once_with(
            'Recording not resumed — audio kept, submit to retry or cancel to discard')
        controller.notify_problem.reset_mock()
        controller.persistence_failed(np.array([0.3], dtype=np.float32))
        controller.notify_problem.assert_called_once_with(
            'Recording not saved to disk — audio kept, submit to retry or cancel to discard')


class WaybarScriptTests(unittest.TestCase):
    def test_the_waybar_script_leaves_recovery_announcements_to_the_service(self):
        script = (ROOT / 'config' / 'hyprland' / 'hyprwhspr-tray.sh').read_text()
        start = script.index('check_recovery_result() {')
        end = script.index('\n}\n', start)
        self.assertNotIn('show_notification', script[start:end])
        self.assertIn('systemctl --user restart hyprwhspr.service', script[start:end])   # behaviour kept
        for phrase in ('recovered successfully', 'Recovery failed', 'Recovery successful'):
            self.assertNotIn(phrase, script)


class ConfigurationTests(unittest.TestCase):
    def test_default_and_schema_agree_on_the_levels(self):
        import json
        from config_manager import build_default_config
        schema = json.loads((ROOT / 'share' / 'config.schema.json').read_text())
        prop = schema['properties']['notification_level']
        self.assertEqual(tuple(prop['enum']), desktop_notify.LEVELS)
        self.assertEqual(prop['default'], 'all')
        defaults = build_default_config()
        self.assertEqual(defaults['notification_level'], 'all')
        self.assertEqual(defaults['notification_timeout_ms'], 5000)


if __name__ == '__main__':
    unittest.main()
