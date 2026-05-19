"""
Keyboard hotplug monitor for hyprwhspr.
Uses pyudev to detect when input (keyboard) devices are plugged or unplugged,
so newly attached keyboards (e.g. a USB keyboard via a dock) can be grabbed
without restarting the service.
"""

import queue
import threading

try:
    import pyudev
    PYUDEV_AVAILABLE = True
except ImportError:
    PYUDEV_AVAILABLE = False


class KeyboardMonitor:
    """Monitor for input-subsystem hotplug events on /dev/input/event* nodes."""

    def __init__(self, on_add=None, on_remove=None):
        self.on_add = on_add
        self.on_remove = on_remove
        self.observer = None
        self.monitor = None
        self.context = None
        self.is_running = False
        self._event_queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None

        if not PYUDEV_AVAILABLE:
            print("[KEYBOARD_MONITOR] pyudev not available, keyboard hotplug disabled")

    def _dispatch_loop(self):
        while True:
            item = self._event_queue.get()
            if item is None:
                break
            action, devnode = item
            try:
                if action == 'add' and self.on_add:
                    self.on_add(devnode)
                elif action == 'remove' and self.on_remove:
                    self.on_remove(devnode)
            except Exception as e:
                print(f"[KEYBOARD_MONITOR] Error handling event: {e}")

    def start(self) -> bool:
        """Start monitoring. Returns True on success."""
        if not PYUDEV_AVAILABLE:
            return False
        if self.is_running:
            return True

        try:
            self.context = pyudev.Context()
            self.monitor = pyudev.Monitor.from_netlink(self.context)
            self.monitor.filter_by(subsystem='input')

            def handle_event(action, device):
                try:
                    devnode = device.device_node
                    # Only /dev/input/eventN nodes are actionable; parent
                    # input class devices without a node also fire events.
                    if not devnode or not devnode.startswith('/dev/input/event'):
                        return
                    if action in ('add', 'remove'):
                        self._event_queue.put((action, devnode))
                except Exception as e:
                    print(f"[KEYBOARD_MONITOR] Error handling event: {e}")

            self.observer = pyudev.MonitorObserver(self.monitor, handle_event)
            self.observer.start()
            # Start worker after observer so a failed observer leaves no stranded thread.
            self._worker = threading.Thread(target=self._dispatch_loop, daemon=True)
            self._worker.start()
            self.is_running = True
            print("[KEYBOARD_MONITOR] Started monitoring for keyboard hotplug events")
            return True

        except Exception as e:
            print(f"[KEYBOARD_MONITOR] Failed to start: {e}")
            if self.observer:
                try:
                    self.observer.stop()
                except Exception:
                    pass
                self.observer = None
            if self._worker is not None:
                self._event_queue.put(None)
                self._worker.join(timeout=2.0)
                self._worker = None
            self.monitor = None
            self.context = None
            self.is_running = False
            return False

    def stop(self):
        """Stop monitoring."""
        if not self.is_running:
            return
        if self.observer:
            try:
                self.observer.stop()
            except Exception as e:
                print(f"[KEYBOARD_MONITOR] Error stopping observer: {e}")
            finally:
                self.observer = None
                self.monitor = None
                self.context = None
                self.is_running = False
                print("[KEYBOARD_MONITOR] Stopped monitoring")
        self._event_queue.put(None)
        if self._worker:
            self._worker.join(timeout=2.0)
            self._worker = None
