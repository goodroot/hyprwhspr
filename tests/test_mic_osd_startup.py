"""Service presenter selection without starting hardware or desktop services."""

import os
import sys
import types
import unittest
from unittest import mock

from tests.test_suspend_resume_recovery import _import_main_isolated, patch_app_global
from mic_osd.runner import MicOSDRunner


class MicOSDStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_class = _import_main_isolated().hyprwhsprApp

    def _initialize(self, *, wayland=False, gtk3=True, layer_shell=False,
                    dependencies=True, started=True, notifications=True, enabled=True):
        app = object.__new__(self.app_class)
        settings = {'mic_osd_enabled': enabled, 'mic_osd_style': 'pill',
                    'notification_timeout_ms': 4200}
        app.config = mock.Mock()
        app.config.get_setting.side_effect = lambda key, default=None: settings.get(key, default)
        app.audio_capture = types.SimpleNamespace(get_viz_frame=mock.Mock())
        env = {'DISPLAY': ':0', 'WAYLAND_DISPLAY': 'wayland-1' if wayland else ''}
        runner = mock.Mock()
        runner.is_available.return_value = dependencies
        runner.layer_shell_active.return_value = layer_shell
        runner.x11_available.side_effect = lambda: MicOSDRunner.x11_session() and gtk3
        runner.return_value._ensure_daemon.return_value = started
        presenter = mock.Mock()
        presenter.return_value.is_available.return_value = notifications
        with mock.patch.dict(os.environ, env), \
                mock.patch.dict(sys.modules, {'mic_osd': types.SimpleNamespace(
                    MicOSDRunner=runner, NotificationPresenter=presenter)}), \
                patch_app_global(self.app_class, 'log', mock.Mock()), \
                patch_app_global(self.app_class, '_looks_like_wlroots_session', lambda: False):
            app._init_mic_osd()
        return app, runner, presenter

    def test_supported_wayland_starts_overlay_with_configured_style_and_feed(self):
        app, runner, presenter = self._initialize(wayland=True, layer_shell=True)
        self.assertIs(app._mic_osd_runner, runner.return_value)
        runner.assert_called_once_with(level_source=app.audio_capture.get_viz_frame, style='pill')
        runner.return_value._ensure_daemon.assert_called_once_with()
        presenter.assert_not_called()

    def test_unsupported_wayland_with_xwayland_uses_notifications(self):
        app, runner, presenter = self._initialize(wayland=True)
        self.assertIs(app._mic_osd_runner, presenter.return_value)
        presenter.assert_called_once_with(active_timeout_ms=4200)
        runner.assert_not_called()

    def test_x11_with_gtk3_starts_overlay(self):
        app, runner, presenter = self._initialize()
        self.assertIs(app._mic_osd_runner, runner.return_value)
        runner.return_value._ensure_daemon.assert_called_once_with()
        presenter.assert_not_called()

    def test_x11_without_gtk3_uses_notifications(self):
        app, runner, presenter = self._initialize(gtk3=False)
        self.assertIs(app._mic_osd_runner, presenter.return_value)
        runner.assert_not_called()

    def test_failed_daemon_start_falls_back_to_notifications(self):
        app, runner, presenter = self._initialize(started=False)
        runner.return_value._ensure_daemon.assert_called_once_with()
        self.assertIs(app._mic_osd_runner, presenter.return_value)

    def test_no_available_presenter_leaves_indicator_unset(self):
        app, runner, presenter = self._initialize(gtk3=False, notifications=False)
        self.assertIsNone(app._mic_osd_runner)
        runner.assert_not_called()
        presenter.return_value.is_available.assert_called_once_with()

    def test_disabled_osd_does_not_probe_or_start_presenters(self):
        app, runner, presenter = self._initialize(enabled=False)
        self.assertIsNone(app._mic_osd_runner)
        self.assertEqual(runner.mock_calls, [])
        self.assertEqual(presenter.mock_calls, [])
