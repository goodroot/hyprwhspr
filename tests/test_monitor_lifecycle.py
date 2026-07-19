import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import pulse_monitor
import suspend_monitor


class PulseMonitorLifecycleTests(unittest.TestCase):
    def test_start_replaces_resources_left_by_crashed_thread(self):
        monitor = pulse_monitor.PulseAudioMonitor()
        old_pulse = mock.Mock()
        dead_thread = mock.Mock()
        dead_thread.is_alive.return_value = False
        monitor._pulse = old_pulse
        monitor._monitor_thread = dead_thread
        monitor._running = True
        new_pulse = mock.Mock()
        new_pulse.server_info.return_value.default_source_name = "mic"
        fake_pulsectl = types.SimpleNamespace(Pulse=mock.Mock(return_value=new_pulse))

        with (
            mock.patch.object(pulse_monitor, "PULSECTL_AVAILABLE", True),
            mock.patch.object(pulse_monitor, "pulsectl", fake_pulsectl),
            mock.patch("pulse_monitor.threading.Thread") as thread_type,
        ):
            self.assertTrue(monitor.start())

        old_pulse.close.assert_called_once_with()
        thread_type.return_value.start.assert_called_once_with()
        self.assertIs(monitor._pulse, new_pulse)

    def test_start_refuses_to_overlap_thread_that_is_still_stopping(self):
        monitor = pulse_monitor.PulseAudioMonitor()
        monitor._monitor_thread = mock.Mock()
        monitor._monitor_thread.is_alive.return_value = True
        fake_pulsectl = types.SimpleNamespace(Pulse=mock.Mock())

        with (
            mock.patch.object(pulse_monitor, "PULSECTL_AVAILABLE", True),
            mock.patch.object(pulse_monitor, "pulsectl", fake_pulsectl),
        ):
            self.assertFalse(monitor.start())

        fake_pulsectl.Pulse.assert_not_called()

    def test_reconnect_stops_retrying_when_shutdown_begins(self):
        monitor = pulse_monitor.PulseAudioMonitor()
        monitor._running = True
        monitor._pulse = mock.Mock()
        monitor._stop_event.wait = mock.Mock(side_effect=lambda timeout: monitor._stop_event.set() or True)
        fake_pulsectl = types.SimpleNamespace(Pulse=mock.Mock(side_effect=RuntimeError("offline")))

        with mock.patch.object(pulse_monitor, "pulsectl", fake_pulsectl):
            self.assertFalse(monitor._reconnect())

        fake_pulsectl.Pulse.assert_called_once_with("hyprwhspr-monitor")
        monitor._stop_event.wait.assert_called_once_with(2)

    def test_reconnect_does_not_open_connection_after_stop(self):
        monitor = pulse_monitor.PulseAudioMonitor()
        monitor._running = False
        monitor._stop_event.set()
        fake_pulsectl = types.SimpleNamespace(Pulse=mock.Mock())

        with mock.patch.object(pulse_monitor, "pulsectl", fake_pulsectl):
            self.assertFalse(monitor._reconnect())

        fake_pulsectl.Pulse.assert_not_called()


class SuspendMonitorLifecycleTests(unittest.TestCase):
    def test_start_removes_subscription_left_by_crashed_thread(self):
        monitor = suspend_monitor.SuspendMonitor()
        old_match = mock.Mock()
        dead_thread = mock.Mock()
        dead_thread.is_alive.return_value = False
        monitor._signal_match = old_match
        monitor._bus = mock.Mock()
        monitor._thread = dead_thread
        monitor._running = True
        new_match = mock.Mock()
        bus = mock.Mock()
        bus.add_signal_receiver.return_value = new_match
        fake_dbus = types.SimpleNamespace(SystemBus=mock.Mock(return_value=bus))
        loop = mock.Mock()
        fake_glib = types.SimpleNamespace(MainLoop=mock.Mock(return_value=loop))

        with (
            mock.patch.object(suspend_monitor, "DBUS_AVAILABLE", True),
            mock.patch.object(suspend_monitor, "DBusGMainLoop"),
            mock.patch.object(suspend_monitor, "dbus", fake_dbus),
            mock.patch.object(suspend_monitor, "GLib", fake_glib),
            mock.patch("suspend_monitor.threading.Thread") as thread_type,
        ):
            self.assertTrue(monitor.start())

        old_match.remove.assert_called_once_with()
        thread_type.return_value.start.assert_called_once_with()
        self.assertIs(monitor._signal_match, new_match)

    def test_start_refuses_to_overlap_thread_that_is_still_stopping(self):
        monitor = suspend_monitor.SuspendMonitor()
        monitor._thread = mock.Mock()
        monitor._thread.is_alive.return_value = True

        with mock.patch.object(suspend_monitor, "DBUS_AVAILABLE", True):
            self.assertFalse(monitor.start())

    def test_stop_unregisters_signal_and_late_signal_is_ignored(self):
        on_suspend = mock.Mock()
        monitor = suspend_monitor.SuspendMonitor(on_suspend_callback=on_suspend)
        signal_match = mock.Mock()
        loop = mock.Mock()
        thread = mock.Mock()
        thread.is_alive.side_effect = [True, False]
        monitor._signal_match = signal_match
        monitor._bus = mock.Mock()
        monitor._loop = loop
        monitor._thread = thread
        monitor._running = True

        monitor.stop()
        monitor._handle_sleep_signal(True)

        loop.quit.assert_called_once_with()
        thread.join.assert_called_once_with(timeout=2.0)
        signal_match.remove.assert_called_once_with()
        self.assertIsNone(monitor._signal_match)
        self.assertIsNone(monitor._bus)
        self.assertIsNone(monitor._loop)
        self.assertIsNone(monitor._thread)
        on_suspend.assert_not_called()

    def test_failed_start_unregisters_partial_signal_subscription(self):
        monitor = suspend_monitor.SuspendMonitor()
        signal_match = mock.Mock()
        bus = mock.Mock()
        bus.add_signal_receiver.return_value = signal_match
        fake_dbus = types.SimpleNamespace(SystemBus=mock.Mock(return_value=bus))
        fake_glib = types.SimpleNamespace(MainLoop=mock.Mock(side_effect=RuntimeError("no loop")))

        with (
            mock.patch.object(suspend_monitor, "DBUS_AVAILABLE", True),
            mock.patch.object(suspend_monitor, "DBusGMainLoop"),
            mock.patch.object(suspend_monitor, "dbus", fake_dbus),
            mock.patch.object(suspend_monitor, "GLib", fake_glib),
        ):
            self.assertFalse(monitor.start())

        signal_match.remove.assert_called_once_with()
        self.assertIsNone(monitor._signal_match)
        self.assertIsNone(monitor._bus)


if __name__ == "__main__":
    unittest.main()
