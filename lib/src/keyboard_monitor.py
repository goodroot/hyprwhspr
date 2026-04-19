"""
Keyboard hotplug monitor for hyprwhspr.
Uses pyudev to detect when input (keyboard) devices are plugged or unplugged,
so newly attached keyboards (e.g. a USB keyboard via a dock) can be grabbed
without restarting the service.
"""

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

        if not PYUDEV_AVAILABLE:
            print("[KEYBOARD_MONITOR] pyudev not available, keyboard hotplug disabled")

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
                    # Only care about the /dev/input/eventN nodes.
                    # Parent input class devices without a device node also
                    # fire events; they're not useful here.
                    if not devnode or not devnode.startswith('/dev/input/event'):
                        return

                    if action == 'add' and self.on_add:
                        threading.Thread(
                            target=self.on_add,
                            args=(devnode,),
                            daemon=True,
                        ).start()
                    elif action == 'remove' and self.on_remove:
                        threading.Thread(
                            target=self.on_remove,
                            args=(devnode,),
                            daemon=True,
                        ).start()
                except Exception as e:
                    print(f"[KEYBOARD_MONITOR] Error handling event: {e}")

            self.observer = pyudev.MonitorObserver(self.monitor, handle_event)
            self.observer.start()
            self.is_running = True
            print("[KEYBOARD_MONITOR] Started monitoring for keyboard hotplug events")
            return True

        except Exception as e:
            print(f"[KEYBOARD_MONITOR] Failed to start: {e}")
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
