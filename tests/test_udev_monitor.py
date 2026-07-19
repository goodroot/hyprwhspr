import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


SRC = Path(__file__).resolve().parents[1] / "lib" / "src"
sys.path.insert(0, str(SRC))

from udev_monitor import SerializedUdevMonitor


class FakeContext:
    pass


class FakeMonitor:
    instances = []

    def __init__(self):
        self.filters = []
        self.__class__.instances.append(self)

    def filter_by(self, subsystem=None):
        self.filters.append(subsystem)

    @classmethod
    def from_netlink(cls, context):
        return cls()


class FakeObserver:
    instances = []
    fail_start = False

    def __init__(self, monitor, callback):
        self.monitor = monitor
        self.callback = callback
        self.stopped = False
        self.__class__.instances.append(self)

    def start(self):
        if self.fail_start:
            raise RuntimeError("observer start")

    def stop(self):
        self.stopped = True


class FakePyudev(types.ModuleType):
    def __init__(self):
        super().__init__("pyudev")
        self.Context = FakeContext
        self.Monitor = FakeMonitor
        self.MonitorObserver = FakeObserver


class FakeDevice:
    def __init__(self, device_node):
        self.device_node = device_node


class StuckWorker:
    def __init__(self, *args, **kwargs):
        self.alive = False

    def start(self):
        self.alive = True

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return self.alive


class UdevMonitorTests(unittest.TestCase):
    def setUp(self):
        FakeMonitor.instances = []
        FakeObserver.instances = []
        FakeObserver.fail_start = False
        self.pyudev = FakePyudev()

    def test_keyboard_adapter_filters_and_normalizes_event_nodes(self):
        saved = sys.modules.get("pyudev")
        sys.modules["pyudev"] = self.pyudev
        try:
            import keyboard_monitor
            module = importlib.reload(keyboard_monitor)
            seen = []
            monitor = module.KeyboardMonitor(
                on_add=lambda node: seen.append(("add", node)),
                on_remove=lambda node: seen.append(("remove", node)),
            )
            self.assertTrue(monitor.start())
            observer = FakeObserver.instances[-1]
            observer.callback("add", FakeDevice(None))
            observer.callback("add", FakeDevice("/dev/input/mouse0"))
            observer.callback("change", FakeDevice("/dev/input/event1"))
            observer.callback("add", FakeDevice("/dev/input/event1"))
            observer.callback("remove", FakeDevice("/dev/input/event1"))
            monitor.stop()
        finally:
            if saved is None:
                sys.modules.pop("pyudev", None)
            else:
                sys.modules["pyudev"] = saved

        self.assertEqual(FakeMonitor.instances[-1].filters, ["input"])
        self.assertEqual(seen, [
            ("add", "/dev/input/event1"),
            ("remove", "/dev/input/event1"),
        ])

    def test_restart_uses_fresh_queue_and_stopped_callback_is_ignored(self):
        seen = []
        monitor = SerializedUdevMonitor(
            self.pyudev, "sound", lambda action, device: (action, device),
            lambda action, device: seen.append((action, device)), "TEST",
        )
        self.assertTrue(monitor.start())
        first_queue = monitor._event_queue
        first_observer = FakeObserver.instances[-1]
        monitor.stop()
        first_observer.callback("add", "late")

        self.assertTrue(monitor.start())
        self.assertIsNot(monitor._event_queue, first_queue)
        FakeObserver.instances[-1].callback("add", "current")
        monitor.stop()

        self.assertEqual(seen, [("add", "current")])

    def test_observer_start_failure_cleans_partial_resources(self):
        FakeObserver.fail_start = True
        monitor = SerializedUdevMonitor(
            self.pyudev, "sound", lambda action, device: (action, device),
            lambda action, device: None, "TEST",
        )

        self.assertFalse(monitor.start())

        self.assertTrue(FakeObserver.instances[-1].stopped)
        self.assertIsNone(monitor.observer)
        self.assertIsNone(monitor.monitor)
        self.assertIsNone(monitor.context)
        self.assertIsNone(monitor._worker)

    def test_timed_out_worker_blocks_overlapping_restart(self):
        monitor = SerializedUdevMonitor(
            self.pyudev, "sound", lambda action, device: (action, device),
            lambda action, device: None, "TEST",
        )
        with mock.patch("udev_monitor.threading.Thread", StuckWorker):
            self.assertTrue(monitor.start())
            worker = monitor._worker
            monitor.stop()
            self.assertIs(monitor._worker, worker)
            self.assertFalse(monitor.start())


if __name__ == "__main__":
    unittest.main()
