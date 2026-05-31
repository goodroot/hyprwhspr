import importlib
import sys
import types
import unittest
import queue
from pathlib import Path


class FakeContext:
    pass


class FakeMonitor:
    def __init__(self):
        self.filters = []

    def filter_by(self, subsystem=None):
        self.filters.append(subsystem)

    @classmethod
    def from_netlink(cls, context):
        return cls()


class FakeObserver:
    instances = []

    def __init__(self, monitor, callback):
        self.monitor = monitor
        self.callback = callback
        self.started = False
        self.stopped = False
        self.__class__.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class FakePyudev(types.ModuleType):
    def __init__(self):
        super().__init__("pyudev")
        self.Context = FakeContext
        self.Monitor = FakeMonitor
        self.MonitorObserver = FakeObserver


class DeviceMonitorTests(unittest.TestCase):
    def _load_device_monitor(self):
        self._saved_device_monitor = sys.modules.get("device_monitor")
        self._saved_pyudev = sys.modules.get("pyudev")
        FakeObserver.instances = []
        sys.modules["pyudev"] = FakePyudev()
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "src"))
        import device_monitor
        return importlib.reload(device_monitor)

    def tearDown(self):
        if hasattr(self, "_saved_pyudev"):
            if self._saved_pyudev is None:
                sys.modules.pop("pyudev", None)
            else:
                sys.modules["pyudev"] = self._saved_pyudev
        if hasattr(self, "_saved_device_monitor"):
            if self._saved_device_monitor is None:
                sys.modules.pop("device_monitor", None)
            else:
                sys.modules["device_monitor"] = self._saved_device_monitor

    def test_events_are_queued_and_dispatched_serially(self):
        module = self._load_device_monitor()
        seen = []
        monitor = module.DeviceMonitor(
            on_audio_add=lambda device: seen.append(("add", device)),
            on_audio_remove=lambda device: seen.append(("remove", device)),
        )

        self.assertTrue(monitor.start())
        observer = FakeObserver.instances[-1]
        observer.callback("add", "mic-a")
        observer.callback("remove", "mic-a")
        # stop() enqueues the sentinel behind the events above and joins the
        # worker, so FIFO ordering guarantees both callbacks have run here.
        monitor.stop()

        self.assertEqual(seen, [("add", "mic-a"), ("remove", "mic-a")])
        self.assertTrue(observer.stopped)
        self.assertIsNone(monitor._worker)

    def test_stop_shuts_down_worker_without_observer(self):
        module = self._load_device_monitor()
        monitor = module.DeviceMonitor()
        self.assertTrue(monitor.start())
        worker = monitor._worker

        monitor.stop()

        self.assertFalse(worker.is_alive())
        self.assertIsNone(monitor._worker)

    def test_stop_before_start_does_not_leave_stale_sentinel(self):
        module = self._load_device_monitor()
        seen = []
        monitor = module.DeviceMonitor(on_audio_add=lambda device: seen.append(device))

        monitor.stop()
        with self.assertRaises(queue.Empty):
            monitor._event_queue.get_nowait()

        self.assertTrue(monitor.start())
        FakeObserver.instances[-1].callback("add", "mic-a")
        monitor.stop()

        self.assertEqual(seen, ["mic-a"])


if __name__ == "__main__":
    unittest.main()
