"""
Suspend/resume monitor for hyprwhspr
Uses D-Bus signals from systemd-logind to detect suspend/resume events
"""

import threading

try:
    import dbus
    from dbus.mainloop.glib import DBusGMainLoop
    from gi.repository import GLib
    DBUS_AVAILABLE = True
except ImportError:
    DBUS_AVAILABLE = False


class SuspendMonitor:
    """Monitor for system suspend/resume events via D-Bus"""

    def __init__(self, on_suspend_callback=None, on_resume_callback=None):
        self.on_suspend = on_suspend_callback
        self.on_resume = on_resume_callback
        self._loop = None
        self._thread = None
        self._running = False

        if not DBUS_AVAILABLE:
            print("[SUSPEND_MONITOR] D-Bus/GLib not available, suspend monitoring disabled")

    def start(self) -> bool:
        """Start monitoring for suspend/resume events"""
        if not DBUS_AVAILABLE:
            return False

        if self._running:
            return True

        try:
            # Set up D-Bus main loop
            DBusGMainLoop(set_as_default=True)

            # Connect to system bus
            bus = dbus.SystemBus()

            # Subscribe to PrepareForSleep signal from systemd-logind
            bus.add_signal_receiver(
                self._handle_sleep_signal,
                signal_name='PrepareForSleep',
                dbus_interface='org.freedesktop.login1.Manager',
                path='/org/freedesktop/login1'
            )

            # Create GLib main loop
            self._loop = GLib.MainLoop()

            # Start main loop in daemon thread
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

            print("[SUSPEND_MONITOR] Started monitoring for suspend/resume events via D-Bus")
            return True

        except Exception as e:
            print(f"[SUSPEND_MONITOR] Failed to start: {e}")
            self._running = False
            return False

    def _run_loop(self):
        """Run GLib main loop (runs in daemon thread)"""
        try:
            self._loop.run()
        except Exception as e:
            print(f"[SUSPEND_MONITOR] Main loop crashed: {e}")
        finally:
            print("[SUSPEND_MONITOR] Main loop exited")

    def _handle_sleep_signal(self, entering_suspend):
        """Handle PrepareForSleep signal from systemd-logind

        Args:
            entering_suspend: True when entering suspend, False when resuming
        """
        try:
            if entering_suspend:
                # System is about to suspend
                # Note: Callback will log details, no need to log here
                if self.on_suspend:
                    # Run callback in separate thread to avoid blocking D-Bus
                    threading.Thread(
                        target=self.on_suspend,
                        daemon=True
                    ).start()
            else:
                # System just resumed
                # Note: Callback will log details, no need to log here
                if self.on_resume:
                    # Run callback in separate thread
                    threading.Thread(
                        target=self.on_resume,
                        daemon=True
                    ).start()
        except Exception as e:
            print(f"[SUSPEND_MONITOR] Error handling sleep signal: {e}")

    def stop(self):
        """Stop monitoring for suspend/resume events"""
        if not self._running:
            return

        print("[SUSPEND_MONITOR] Stopping...")
        self._running = False

        # Quit the GLib main loop
        if self._loop:
            self._loop.quit()

        # Wait for thread to exit
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        print("[SUSPEND_MONITOR] Stopped monitoring")
