"""
Device hotplug monitor for hyprwhspr
Uses pyudev to detect when audio devices are plugged/unplugged
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


class DeviceMonitor(SerializedUdevMonitor):
    """Monitor for audio device hotplug events"""

    def __init__(self, on_audio_add=None, on_audio_remove=None):
        self.on_audio_add = on_audio_add
        self.on_audio_remove = on_audio_remove
        super().__init__(
            pyudev,
            subsystem='sound',
            normalize_event=self._normalize_event,
            dispatch_event=self._dispatch_event,
            log_prefix='DEVICE_MONITOR',
        )

        if not PYUDEV_AVAILABLE:
            print("[DEVICE_MONITOR] pyudev not available, hotplug detection disabled")

    @staticmethod
    def _normalize_event(action, device):
        return (action, device) if action in ('add', 'remove') else None

    def _dispatch_event(self, action, device):
        if action == 'add' and self.on_audio_add:
            self.on_audio_add(device)
        elif action == 'remove' and self.on_audio_remove:
            self.on_audio_remove(device)

    def start(self):
        if super().start():
            print("[DEVICE_MONITOR] Started monitoring for audio device hotplug events")
            return True
        return False

    def stop(self):
        """Stop monitoring for device events"""
        was_running = self.is_running
        super().stop()
        if was_running:
            print("[DEVICE_MONITOR] Stopped monitoring")

    @staticmethod
    def get_device_properties(device):
        """Extract useful properties from a udev device"""
        return {
            'vendor_id': device.get('ID_VENDOR_ID'),
            'model_id': device.get('ID_MODEL_ID'),
            'serial': device.get('ID_SERIAL_SHORT'),
            'model_name': device.get('ID_MODEL'),
            'path': device.get('ID_PATH'),
            'device_path': device.device_path,
        }
