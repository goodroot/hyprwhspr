"""
Device hotplug monitor for hyprwhspr
Uses pyudev to detect when audio devices are plugged/unplugged
"""

import threading
import time

try:
    import pyudev
    PYUDEV_AVAILABLE = True
except ImportError:
    PYUDEV_AVAILABLE = False


class DeviceMonitor:
    """Monitor for audio device hotplug events"""

    def __init__(self, on_audio_add=None, on_audio_remove=None):
        self.on_audio_add = on_audio_add
        self.on_audio_remove = on_audio_remove
        self.observer = None
        self.monitor = None
        self.context = None
        self.is_running = False

        if not PYUDEV_AVAILABLE:
            print("[DEVICE_MONITOR] pyudev not available, hotplug detection disabled")

    def start(self):
        """Start monitoring for device events"""
        if not PYUDEV_AVAILABLE:
            return False

        if self.is_running:
            return True

        try:
            self.context = pyudev.Context()
            self.monitor = pyudev.Monitor.from_netlink(self.context)
            self.monitor.filter_by(subsystem='sound')

            def handle_event(action, device):
                try:
                    if action == 'add':
                        if self.on_audio_add:
                            # Run callback in separate thread to avoid blocking observer
                            threading.Thread(
                                target=self.on_audio_add,
                                args=(device,),
                                daemon=True
                            ).start()
                    elif action == 'remove':
                        if self.on_audio_remove:
                            threading.Thread(
                                target=self.on_audio_remove,
                                args=(device,),
                                daemon=True
                            ).start()
                except Exception as e:
                    print(f"[DEVICE_MONITOR] Error handling event: {e}")

            self.observer = pyudev.MonitorObserver(self.monitor, handle_event)
            self.observer.start()
            self.is_running = True
            print("[DEVICE_MONITOR] Started monitoring for audio device hotplug events")
            return True

        except Exception as e:
            print(f"[DEVICE_MONITOR] Failed to start: {e}")
            self.is_running = False
            return False

    def stop(self):
        """Stop monitoring for device events"""
        if not self.is_running:
            return

        if self.observer:
            try:
                self.observer.stop()
                print("[DEVICE_MONITOR] Stopped monitoring")
            except Exception as e:
                print(f"[DEVICE_MONITOR] Error stopping observer: {e}")
            finally:
                self.observer = None
                self.monitor = None
                self.context = None
                self.is_running = False

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
