import importlib
import sys
import threading
import types
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))


def _stub_if_missing(name, **attrs):
    """Install a stub module only if the real one cannot be imported."""
    try:
        importlib.import_module(name)
    except Exception:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


_stub_if_missing(
    "evdev",
    InputDevice=object,
    list_devices=lambda: [],
    categorize=lambda *a, **k: None,
    ecodes=types.SimpleNamespace(),
    UInput=object,
)

import global_shortcuts  # noqa: E402


def _dev(path):
    return types.SimpleNamespace(path=path)


class ResyncDevicesTests(unittest.TestCase):
    """Cover resync_devices orchestration: guards, path dedup, counting."""

    def _make(self, *, is_running=True, grab_keys=True, devices_grabbed=True,
              devices=()):
        """Build a GlobalShortcuts without running __init__'s device discovery."""
        gs = global_shortcuts.GlobalShortcuts.__new__(
            global_shortcuts.GlobalShortcuts)
        gs.is_running = is_running
        gs.stop_event = threading.Event()
        gs.grab_keys = grab_keys
        gs.devices_grabbed = devices_grabbed
        gs.devices = [_dev(p) for p in devices]
        gs._device_lock = threading.Lock()
        # Attributes the finalizer (stop()) touches, so GC of these bare
        # __new__-built objects doesn't print spurious teardown errors.
        gs.keyboard_monitor = None
        gs.listener_thread = None
        gs.uinput = None
        gs.pressed_keys = set()
        gs.suppressed_keys = set()
        gs.combination_active = False
        # Fake the validated add path: simulate a successful attach by appending
        # a device with that path, and record which paths we were asked to add.
        gs._attempted = []

        def _fake_add(path):
            gs._attempted.append(path)
            gs.devices.append(_dev(path))

        gs._try_hotplug_add = _fake_add
        return gs

    def test_attaches_only_unmonitored_paths(self):
        gs = self._make(devices=["/dev/input/event1"])
        gs_evdev = ["/dev/input/event1", "/dev/input/event5"]
        with mock.patch.object(global_shortcuts.evdev,
                                        "list_devices",
                                        return_value=gs_evdev):
            attached = gs.resync_devices("test")
        self.assertEqual(attached, 1)
        self.assertEqual(gs._attempted, ["/dev/input/event5"])

    def test_noop_when_all_paths_known(self):
        gs = self._make(devices=["/dev/input/event1", "/dev/input/event2"])
        with mock.patch.object(
                global_shortcuts.evdev, "list_devices",
                return_value=["/dev/input/event1", "/dev/input/event2"]):
            attached = gs.resync_devices("test")
        self.assertEqual(attached, 0)
        self.assertEqual(gs._attempted, [])

    def test_skips_when_not_running(self):
        gs = self._make(is_running=False)
        with mock.patch.object(global_shortcuts.evdev,
                                        "list_devices",
                                        return_value=["/dev/input/event5"]):
            self.assertEqual(gs.resync_devices("test"), 0)
        self.assertEqual(gs._attempted, [])

    def test_skips_before_initial_grab_pass(self):
        gs = self._make(grab_keys=True, devices_grabbed=False)
        with mock.patch.object(global_shortcuts.evdev,
                                        "list_devices",
                                        return_value=["/dev/input/event5"]):
            self.assertEqual(gs.resync_devices("test"), 0)
        self.assertEqual(gs._attempted, [])

    def test_clears_stale_held_key_state(self):
        # A key held when the device dropped leaves stale state; resync clears it.
        gs = self._make(devices=["/dev/input/event1"])
        gs.pressed_keys = {30, 56}
        gs.suppressed_keys = {30}
        gs.combination_active = True
        with mock.patch.object(global_shortcuts.evdev, "list_devices",
                                        return_value=["/dev/input/event1"]):
            gs.resync_devices("test")
        self.assertEqual(gs.pressed_keys, set())
        self.assertEqual(gs.suppressed_keys, set())
        self.assertFalse(gs.combination_active)

    def test_runs_without_grab_when_grab_disabled(self):
        # grab_keys=False should still scan (the grab gate doesn't apply).
        gs = self._make(grab_keys=False, devices_grabbed=False)
        with mock.patch.object(global_shortcuts.evdev,
                                        "list_devices",
                                        return_value=["/dev/input/event5"]):
            self.assertEqual(gs.resync_devices("test"), 1)
        self.assertEqual(gs._attempted, ["/dev/input/event5"])


if __name__ == "__main__":
    unittest.main()
