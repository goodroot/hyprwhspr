"""
PulseAudio/PipeWire event monitor for hyprwhspr
Uses pulsectl to detect device changes and server restarts
"""

import threading
import time

try:
    import pulsectl
    PULSECTL_AVAILABLE = True
except ImportError:
    PULSECTL_AVAILABLE = False


class PulseAudioMonitor:
    """Monitor for PulseAudio/PipeWire events"""

    def __init__(self, on_default_change_callback=None, on_server_restart_callback=None):
        self.on_default_change = on_default_change_callback
        self.on_server_restart = on_server_restart_callback
        self._pulse = None
        self._monitor_thread = None
        self._running = False
        self._default_source_name = None
        self._pending_server_check = False  # Flag to defer server_info() outside callback

        if not PULSECTL_AVAILABLE:
            print("[PULSE_MONITOR] pulsectl not available, pulse monitoring disabled")

    def start(self) -> bool:
        """Start monitoring pulse events"""
        if not PULSECTL_AVAILABLE:
            return False

        if self._running:
            return True

        try:
            # Create pulse connection
            self._pulse = pulsectl.Pulse('hyprwhspr-monitor')

            # Get initial default source
            try:
                server_info = self._pulse.server_info()
                self._default_source_name = server_info.default_source_name
                print(f"[PULSE_MONITOR] Initial default source: {self._default_source_name}")
            except Exception as e:
                print(f"[PULSE_MONITOR] Could not get initial default source: {e}")

            # Subscribe to pulse events
            # Subscribe to server and source events
            self._pulse.event_mask_set('server', 'source')

            # Set up event listener callback
            self._pulse.event_callback_set(self._event_callback)

            # Start event listener thread
            self._running = True
            self._monitor_thread = threading.Thread(target=self._event_loop, daemon=True)
            self._monitor_thread.start()

            print("[PULSE_MONITOR] Started monitoring for PulseAudio/PipeWire events")
            return True

        except Exception as e:
            print(f"[PULSE_MONITOR] Failed to start: {e}")
            self._running = False
            if self._pulse:
                try:
                    self._pulse.close()
                except Exception:
                    pass
                self._pulse = None
            return False

    def _event_loop(self):
        """Event listener loop (runs in daemon thread)"""
        try:
            while self._running:
                try:
                    # Listen for events (blocking call with timeout)
                    self._pulse.event_listen(timeout=1.0)

                    # Check for pending server info check (deferred from callback to avoid threading violation)
                    if self._pending_server_check:
                        self._pending_server_check = False
                        self._check_default_source_change()
                except pulsectl.PulseDisconnected:
                    # Server disconnected - likely a restart
                    print("[PULSE_MONITOR] PulseAudio server disconnected")
                    if self._running and self.on_server_restart:
                        # Run callback in separate thread
                        threading.Thread(
                            target=self.on_server_restart,
                            daemon=True
                        ).start()

                    # Try to reconnect
                    if self._running:
                        self._reconnect()
                except Exception as e:
                    if self._running:  # Only log if we're supposed to be running
                        print(f"[PULSE_MONITOR] Error in event loop: {e}")
                        time.sleep(1)  # Brief pause before retrying
        except Exception as e:
            print(f"[PULSE_MONITOR] Event loop crashed: {e}")
        finally:
            print("[PULSE_MONITOR] Event loop exited")

    def _reconnect(self):
        """Attempt to reconnect to pulse server"""
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                print(f"[PULSE_MONITOR] Reconnecting... (attempt {attempt + 1}/{max_attempts})")
                if self._pulse:
                    try:
                        self._pulse.close()
                    except Exception:
                        pass

                self._pulse = pulsectl.Pulse('hyprwhspr-monitor')
                self._pulse.event_mask_set('server', 'source')
                self._pulse.event_callback_set(self._event_callback)

                # Get new default source
                try:
                    server_info = self._pulse.server_info()
                    self._default_source_name = server_info.default_source_name
                except Exception:
                    pass

                print("[PULSE_MONITOR] Reconnected successfully")
                return True
            except Exception as e:
                if attempt < max_attempts - 1:
                    time.sleep(2)  # Wait before retrying
                else:
                    print(f"[PULSE_MONITOR] Failed to reconnect after {max_attempts} attempts: {e}")
                    return False

    def _event_callback(self, event):
        """Handle pulse events"""
        try:
            # Check for server events (includes default source changes)
            if event.facility == pulsectl.PulseEventFacilityEnum.server:
                # Defer server_info() check to event loop (can't call blocking operations from callback)
                self._pending_server_check = True

        except Exception as e:
            print(f"[PULSE_MONITOR] Error in event callback: {e}")

    def _check_default_source_change(self):
        """Check if default source changed (called outside event callback to avoid threading violations)"""
        try:
            server_info = self._pulse.server_info()
            new_default_source = server_info.default_source_name

            if new_default_source != self._default_source_name:
                old_default = self._default_source_name
                self._default_source_name = new_default_source
                print(f"[PULSE_MONITOR] Default source changed: {old_default} → {new_default_source}")

                if self.on_default_change:
                    # Run callback in separate thread
                    threading.Thread(
                        target=self.on_default_change,
                        args=(new_default_source,),
                        daemon=True
                    ).start()
        except Exception as e:
            print(f"[PULSE_MONITOR] Error checking default source: {e}")

    def stop(self):
        """Stop monitoring for pulse events"""
        if not self._running:
            return

        print("[PULSE_MONITOR] Stopping...")
        self._running = False

        # Wait for event loop thread to exit
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)

        # Close pulse connection
        if self._pulse:
            try:
                self._pulse.close()
            except Exception as e:
                print(f"[PULSE_MONITOR] Error closing pulse connection: {e}")
            finally:
                self._pulse = None

        print("[PULSE_MONITOR] Stopped monitoring")

    def get_default_source_name(self) -> str:
        """Get current default source name"""
        if not self._pulse:
            return None

        try:
            server_info = self._pulse.server_info()
            return server_info.default_source_name
        except Exception:
            return None
