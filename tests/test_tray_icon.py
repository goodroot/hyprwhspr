"""System tray icon: state mapping, menu building and menu actions."""
import contextlib
import io
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import tray_icon  # noqa: E402
from tests.test_suspend_resume_recovery import _import_main_isolated, patch_app_global  # noqa: E402


class MenuTests(unittest.TestCase):
    def _labels(self, state, service_controls=True):
        return {item_id: props for item_id, props in
                tray_icon.menu_items(state, '', service_controls)}

    def test_record_item_follows_recording_state(self):
        ready, recording = self._labels('ready'), self._labels('recording')
        self.assertEqual(ready[tray_icon.ITEM_START]['label'], 'Start recording')
        self.assertNotIn(tray_icon.ITEM_STOP, ready)
        self.assertEqual(recording[tray_icon.ITEM_STOP]['label'], 'Stop recording')
        self.assertNotIn(tray_icon.ITEM_START, recording)

    def test_paused_offers_resume(self):
        items = self._labels('paused')
        self.assertEqual(items[tray_icon.ITEM_START]['label'], 'Resume recording')
        self.assertTrue(items[tray_icon.ITEM_START]['enabled'])
        self.assertFalse(items[tray_icon.ITEM_UNLOAD]['enabled'])

    def test_model_item_follows_unloaded_state(self):
        self.assertEqual(self._labels('ready')[tray_icon.ITEM_UNLOAD]['label'], 'Unload model')
        self.assertEqual(self._labels('unloaded')[tray_icon.ITEM_LOAD]['label'], 'Load model')
        self.assertNotIn(tray_icon.ITEM_UNLOAD, self._labels('unloaded'))

    def test_each_item_maps_to_one_action(self):
        # The action comes from the id alone, never from the current state
        self.assertEqual(tray_icon.menu_action(tray_icon.ITEM_START), 'start')
        self.assertEqual(tray_icon.menu_action(tray_icon.ITEM_STOP), 'stop')
        self.assertEqual(tray_icon.menu_action(tray_icon.ITEM_UNLOAD), 'model_unload')
        self.assertEqual(tray_icon.menu_action(tray_icon.ITEM_LOAD), 'model_reload')
        self.assertIsNone(tray_icon.menu_action(tray_icon.ITEM_STATUS))

    def test_actions_are_disabled_while_busy(self):
        for state in ('loading', 'processing'):
            items = self._labels(state)
            self.assertFalse(items[tray_icon.ITEM_START]['enabled'])
            self.assertFalse(items[tray_icon.ITEM_UNLOAD]['enabled'])

    def test_model_cannot_be_unloaded_mid_recording(self):
        self.assertFalse(self._labels('recording')[tray_icon.ITEM_UNLOAD]['enabled'])

    def test_service_items_only_with_service_controls(self):
        self.assertIn(tray_icon.ITEM_STOP_SERVICE, self._labels('ready', service_controls=True))
        self.assertNotIn(tray_icon.ITEM_RESTART_SERVICE, self._labels('ready', service_controls=False))
        self.assertNotIn(tray_icon.ITEM_STOP_SERVICE, self._labels('ready', service_controls=False))
        self.assertEqual(tray_icon.menu_action(tray_icon.ITEM_RESTART_SERVICE), 'restart')
        self.assertEqual(tray_icon.menu_action(tray_icon.ITEM_STOP_SERVICE), 'stop_service')

    def test_status_item_shows_detail(self):
        items = tray_icon.menu_items('error', 'Microphone disconnected', False)
        self.assertEqual(items[0][1]['label'], 'hyprwhspr: Microphone disconnected')
        self.assertFalse(items[0][1]['enabled'])

    def test_every_state_has_an_icon(self):
        for state in ('ready', 'recording', 'paused', 'processing', 'loading', 'unloaded', 'error'):
            self.assertTrue(tray_icon.STATES[state][0])


class TrayIconWithoutGioTests(unittest.TestCase):
    def test_start_fails_cleanly_without_pygobject(self):
        tray = tray_icon.TrayIcon(get_state=lambda: ('ready', ''), on_action=lambda a: None)
        with mock.patch.object(tray_icon, 'GIO_AVAILABLE', False), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(tray.start())

    def test_broken_state_provider_reports_error(self):
        def boom():
            raise RuntimeError('gone')
        tray = tray_icon.TrayIcon(get_state=boom, on_action=lambda a: None)
        state, detail = tray._read_state()
        self.assertEqual(state, 'error')
        self.assertIn('gone', detail)

    def test_unknown_state_reports_error(self):
        tray = tray_icon.TrayIcon(get_state=lambda: ('weird', 'x'), on_action=lambda a: None)
        self.assertEqual(tray._read_state(), ('error', 'x'))

    def test_stop_during_slow_connect_keeps_the_icon_off(self):
        # start() gives up after its wait; a connect that finishes later must not publish
        tray = tray_icon.TrayIcon(get_state=lambda: ('ready', ''), on_action=lambda a: None)
        fake_glib = mock.Mock()
        # The connect finishes only once start() has given up and called stop()
        tray._connect = mock.Mock(side_effect=lambda: tray._stopping.wait(5))
        tray._disconnect = mock.Mock()
        with mock.patch.object(tray_icon, 'GIO_AVAILABLE', True), \
             mock.patch.object(tray_icon, 'GLib', fake_glib, create=True), \
             mock.patch.object(tray._ready, 'wait', return_value=False):
            self.assertFalse(tray.start())
        tray._disconnect.assert_called_once()
        fake_glib.MainLoop.new.return_value.run.assert_not_called()
        self.assertFalse(tray._started)


class _Invocation:
    def __init__(self):
        self.value = None
        self.error = None

    def return_value(self, value):
        self.value = value

    def return_dbus_error(self, name, message):
        self.error = (name, message)


@unittest.skipUnless(tray_icon.GIO_AVAILABLE, "PyGObject not installed")
class MenuCallTests(unittest.TestCase):
    def _tray(self, state='ready'):
        self.actions = []
        done = threading.Event()

        def on_action(action):
            self.actions.append(action)
            done.set()
        tray = tray_icon.TrayIcon(get_state=lambda: (state, ''), on_action=on_action,
                                  service_controls=True)
        tray._state = state
        return tray, done

    def _call(self, tray, method, signature, args):
        invocation = _Invocation()
        params = tray_icon.GLib.Variant(signature, args)
        tray._on_menu_call(None, None, tray_icon.MENU_PATH, tray_icon.MENU_IFACE,
                           method, params, invocation)
        return invocation

    def test_get_layout_lists_every_item(self):
        tray, _ = self._tray()
        invocation = self._call(tray, 'GetLayout', '(iias)', (0, -1, []))
        revision, (root_id, _props, children) = invocation.value.unpack()
        self.assertEqual(root_id, 0)
        self.assertEqual([child[0] for child in children],
                         [item_id for item_id, _ in tray._items()])

    def test_click_runs_action_off_the_bus_thread(self):
        tray, done = self._tray('ready')
        data = tray_icon.GLib.Variant('s', '')
        invocation = self._call(tray, 'Event', '(isvu)',
                                (tray_icon.ITEM_START, 'clicked', data, 0))
        self.assertIsNone(invocation.error)
        self.assertTrue(done.wait(2))
        self.assertEqual(self.actions, ['start'])

    def test_stale_stop_click_never_starts_a_recording(self):
        # Silence auto-stop flipped the state while "Stop recording" was on screen
        tray, done = self._tray('ready')
        data = tray_icon.GLib.Variant('s', '')
        self._call(tray, 'Event', '(isvu)', (tray_icon.ITEM_STOP, 'clicked', data, 0))
        self.assertTrue(done.wait(2))
        self.assertEqual(self.actions, ['stop'])

    def test_hover_does_not_run_action(self):
        tray, done = self._tray('ready')
        data = tray_icon.GLib.Variant('s', '')
        self._call(tray, 'Event', '(isvu)', (tray_icon.ITEM_START, 'hovered', data, 0))
        self.assertFalse(done.wait(0.2))

    def test_icon_property_follows_state(self):
        tray, _ = self._tray('recording')
        value = tray._get_item_property(None, None, tray_icon.ITEM_PATH,
                                        tray_icon.ITEM_IFACE, 'IconName')
        self.assertEqual(value.unpack(), tray_icon.STATES['recording'][0])


class AppTrayStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        self.app._model_initializing = False
        self.app._model_operation_active = False
        self.app._backend_init_failed = False
        self.app._mic_state_lock = threading.Lock()
        self.app._mic_disconnected = False
        for name in ('RECORDING_STATUS_FILE', 'MODEL_UNLOADED_FILE', 'MIC_ZERO_VOLUME_FILE'):
            patch = patch_app_global(self.app.__class__, name, self.dir / name)
            patch.start()
            self.addCleanup(patch.stop)

    def _touch(self, name, age=0):
        path = self.dir / name
        path.touch()
        if age:
            stamp = time.time() - age
            os.utime(path, (stamp, stamp))

    def test_ready_by_default(self):
        self.assertEqual(self.app._tray_state(), ('ready', ''))

    def test_recording(self):
        self._touch('RECORDING_STATUS_FILE')
        self.assertEqual(self.app._tray_state()[0], 'recording')

    def test_unloaded(self):
        self._touch('MODEL_UNLOADED_FILE')
        self.assertEqual(self.app._tray_state()[0], 'unloaded')

    def test_loading_wins_while_a_model_operation_runs(self):
        self._touch('MODEL_UNLOADED_FILE')
        self.app._model_operation_active = True
        self.assertEqual(self.app._tray_state()[0], 'loading')

    def test_failed_init_is_an_error(self):
        self.app._backend_init_failed = True
        self.assertEqual(self.app._tray_state(), ('error', 'Model failed to load'))

    def test_disconnected_mic_is_an_error(self):
        self.app._mic_disconnected = True
        self.assertEqual(self.app._tray_state(), ('error', 'Microphone disconnected'))

    def test_zero_volume_signal_expires_after_a_minute(self):
        self._touch('MIC_ZERO_VOLUME_FILE')
        self.assertEqual(self.app._tray_state()[0], 'error')
        self._touch('MIC_ZERO_VOLUME_FILE', age=61)
        self.assertEqual(self.app._tray_state()[0], 'ready')

    def test_long_form_states(self):
        # Long-form never writes the recording file; its own state drives the icon
        self.app._longform = types.SimpleNamespace(state='IDLE')
        expected = {'IDLE': 'ready', 'RECORDING': 'recording', 'PAUSED': 'paused',
                    'PROCESSING': 'processing', 'ERROR': 'error'}
        for longform_state, tray_state in expected.items():
            self.app._longform.state = longform_state
            self.assertEqual(self.app._tray_state()[0], tray_state, longform_state)


class ServiceUnitDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def _detect(self, cgroup):
        cgroup_file = mock.Mock()
        cgroup_file.read_text.return_value = cgroup
        real_path = Path

        def path(p):
            return cgroup_file if p == '/proc/self/cgroup' else real_path(p)
        with patch_app_global(self.main.hyprwhsprApp, 'Path', path):
            return self.main.hyprwhsprApp._running_as_service_unit()

    def test_own_unit(self):
        self.assertTrue(self._detect(
            '0::/user.slice/user-1000.slice/user@1000.service/app.slice/hyprwhspr.service\n'))

    def test_inherited_invocation_id_from_compositor_unit(self):
        # e.g. hyprwhspr launched from Hyprland's exec-once under uwsm
        self.assertFalse(self._detect(
            '0::/user.slice/user-1000.slice/user@1000.service/session.slice/'
            'wayland-wm@hyprland.desktop.service\n'))


class AppTrayActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def _app(self):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app._handle_control_command = mock.Mock()
        return app

    def test_recording_and_model_actions_use_control_commands(self):
        app = self._app()
        for action in ('start', 'stop', 'model_unload', 'model_reload'):
            app._on_tray_action(action)
        self.assertEqual([c.args[0] for c in app._handle_control_command.call_args_list],
                         ['start', 'stop', 'model_unload', 'model_reload'])

    def test_service_actions_queue_a_systemd_job(self):
        app = self._app()
        popen = mock.Mock()
        fake_subprocess = types.SimpleNamespace(Popen=popen, DEVNULL=-3)
        with patch_app_global(app.__class__, 'subprocess', fake_subprocess), \
             contextlib.redirect_stdout(io.StringIO()):
            app._on_tray_action('restart')
            app._on_tray_action('stop_service')
        self.assertEqual([c.args[0] for c in popen.call_args_list], [
            ['systemctl', '--user', '--no-block', 'restart', 'hyprwhspr.service'],
            ['systemctl', '--user', '--no-block', 'stop', 'hyprwhspr.service'],
        ])
        app._handle_control_command.assert_not_called()

    def test_off_by_default(self):
        app = self._app()
        app.config = types.SimpleNamespace(get_setting=lambda key, default=None: default)
        with mock.patch.object(tray_icon.TrayIcon, 'start') as start:
            app._setup_tray_icon()
        start.assert_not_called()
        self.assertIsNone(app.tray_icon)

    def test_disabled_in_config_starts_nothing(self):
        app = self._app()
        app.config = types.SimpleNamespace(get_setting=lambda key, default=None: False)
        with mock.patch.object(tray_icon.TrayIcon, 'start') as start:
            app._setup_tray_icon()
        start.assert_not_called()
        self.assertIsNone(app.tray_icon)



if __name__ == '__main__':
    unittest.main()
