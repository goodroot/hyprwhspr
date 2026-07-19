"""
Keyboard hotplug monitor for hyprwhspr.
Uses pyudev to detect when input (keyboard) devices are plugged or unplugged,
so newly attached keyboards (e.g. a USB keyboard via a dock) can be grabbed
without restarting the service.
"""

try:
    import pyudev
    PYUDEV_AVAILABLE = True
except ImportError:
    pyudev = None
    PYUDEV_AVAILABLE = False

try:
    from .udev_monitor import SerializedUdevMonitor
except ImportError:
    from udev_monitor import SerializedUdevMonitor


class KeyboardMonitor(SerializedUdevMonitor):
    """Monitor for input-subsystem hotplug events on /dev/input/event* nodes."""

    def __init__(self, on_add=None, on_remove=None):
        self.on_add = on_add
        self.on_remove = on_remove
        super().__init__(
            pyudev,
            subsystem='input',
            normalize_event=self._normalize_event,
            dispatch_event=self._dispatch_event,
            log_prefix='KEYBOARD_MONITOR',
        )

        if not PYUDEV_AVAILABLE:
            print("[KEYBOARD_MONITOR] pyudev not available, keyboard hotplug disabled")

    @staticmethod
    def _normalize_event(action, device):
        devnode = device.device_node
        if action not in ('add', 'remove'):
            return None
        if not devnode or not devnode.startswith('/dev/input/event'):
            return None
        return action, devnode

    def _dispatch_event(self, action, devnode):
        if action == 'add' and self.on_add:
            self.on_add(devnode)
        elif action == 'remove' and self.on_remove:
            self.on_remove(devnode)

    def start(self) -> bool:
        """Start monitoring. Returns True on success."""
        if super().start():
            print("[KEYBOARD_MONITOR] Started monitoring for keyboard hotplug events")
            return True
        return False

    def stop(self):
        """Stop monitoring."""
        was_running = self.is_running
        super().stop()
        if was_running:
            print("[KEYBOARD_MONITOR] Stopped monitoring")
