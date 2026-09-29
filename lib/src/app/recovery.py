"""Device, PulseAudio and suspend monitoring, and audio recovery."""

import os
import sys
import threading
import time

from device_monitor import DeviceMonitor, PYUDEV_AVAILABLE
from paths import MIC_ZERO_VOLUME_FILE, RECOVERY_REQUESTED_FILE, RECOVERY_RESULT_FILE
from service_log import log


class RecoveryMixin:
    """Device, PulseAudio and suspend monitoring, and audio recovery."""

    def _setup_device_monitor(self):
        """Initialize device hotplug monitoring for automatic microphone recovery"""
        if PYUDEV_AVAILABLE:
            self.device_monitor = DeviceMonitor(
                on_audio_add=self._on_audio_device_added,
                on_audio_remove=self._on_audio_device_removed
            )
            if self.device_monitor.start():
                log("[INIT] Device hotplug monitoring enabled")
            else:
                log("[WARN] Failed to start device hotplug monitoring")
                self.device_monitor = None
        else:
            self.device_monitor = None
            log("[WARN] pyudev not available - audio hotplug detection disabled")

    def _setup_pulse_monitor(self):
        """Initialize PulseAudio/PipeWire event monitoring"""
        try:
            from src.pulse_monitor import PulseAudioMonitor
            self.pulse_monitor = PulseAudioMonitor(
                on_default_change_callback=self._on_pulse_default_changed,
                on_server_restart_callback=self._on_pulse_server_restarted
            )
            if self.pulse_monitor.start():
                # Monitored bindings stay fresh via events; record start can skip its pactl poll
                self.audio_capture.set_default_monitor_check(self.pulse_monitor.is_healthy)
                log("[INIT] PulseAudio/PipeWire monitoring enabled")
            else:
                log("[WARN] Failed to start PulseAudio monitoring")
                self.pulse_monitor = None
        except ImportError:
            self.pulse_monitor = None
            log("[WARN] pulsectl not available - pulse monitoring disabled")
        except Exception as e:
            self.pulse_monitor = None
            log(f"[WARN] Failed to setup pulse monitor: {e}")

    def _setup_suspend_monitor(self):
        """Initialize suspend/resume monitoring via D-Bus"""
        try:
            from src.suspend_monitor import SuspendMonitor
            self.suspend_monitor = SuspendMonitor(
                on_suspend_callback=self._on_system_suspend,
                on_resume_callback=self._on_system_resume
            )
            if self.suspend_monitor.start():
                log("[INIT] Suspend/resume monitoring enabled (D-Bus)")
            else:
                log("[WARN] Failed to start suspend monitoring")
                self.suspend_monitor = None
        except ImportError:
            self.suspend_monitor = None
            log("[WARN] D-Bus/GLib not available - suspend monitoring disabled")
        except Exception as e:
            self.suspend_monitor = None
            log(f"[WARN] Failed to setup suspend monitor: {e}")

    def _on_audio_device_added(self, device):
        """Called when audio device is plugged in"""
        try:
            # Ignore hotplug events during startup grace period
            # This prevents false positives from pyudev detecting existing devices on startup
            current_time = time.monotonic()
            if current_time - self._startup_time < self._startup_grace_period:
                remaining = self._startup_grace_period - (current_time - self._startup_time)
                log(f"[HOTPLUG] Ignoring hotplug event during startup grace period ({remaining:.1f}s remaining)")
                return

            device_model = device.get('ID_MODEL') or 'Unknown'

            # Determine if we should trigger recovery
            should_recover = False
            configured_name = self.config.get_setting('audio_device_name')

            if configured_name:
                # User has configured a specific device - only recover if it matches
                if device_model and configured_name in device_model:
                    should_recover = True
            else:
                # No configured device - recover on ANY audio device addition
                if device_model != 'Unknown':
                    should_recover = True

            if should_recover:
                # Debounce recovery attempts: USB reseat generates multiple events.
                # Also cancel any in-progress background recovery and reset its cooldown
                # while still holding _hotplug_lock — this closes the window where another
                # thread reads a stale _last_recovery_attempt_time between the two steps.
                canceled_background_recovery = False
                with self._hotplug_lock:
                    current_time = time.monotonic()
                    if current_time - self._last_hotplug_add_time < 2.0:
                        return  # Skip duplicate
                    self._last_hotplug_add_time = current_time

                    if self._background_recovery_needed.is_set():
                        self._background_recovery_needed.clear()
                        canceled_background_recovery = True
                        with self.audio_capture.recovery_lock:
                            self.audio_capture._last_recovery_attempt_time = 0.0

                if canceled_background_recovery:
                    time.sleep(0.1)

                log(f"[HOTPLUG] Microphone detected - recovering...")
                time.sleep(0.5)  # Let drivers settle

                # Trigger recovery
                if self.audio_capture.recover_audio_capture('hotplug_detected'):
                    log(f"[HOTPLUG] Recovery successful")
                    self._write_recovery_result(True, 'hotplug')
                    with self._mic_state_lock:
                        self._mic_disconnected = False
                    self._background_recovery_needed.clear()
                else:
                    log(f"[HOTPLUG] Recovery failed - will retry in background")
                    self._write_recovery_result(False, 'hotplug')
                    # Re-set flag so background recovery can retry
                    self._background_recovery_needed.set()
        except Exception as e:
            log(f"[HOTPLUG] Error: {e}")

    def _on_audio_device_removed(self, device):
        """Called when audio device is unplugged"""
        try:
            device_model = device.get('ID_MODEL') or 'Unknown'

            # Determine if this is a significant removal
            configured_name = self.config.get_setting('audio_device_name')
            is_significant_removal = False

            if configured_name:
                # User has configured a specific device - only mark disconnected if it matches
                if device_model and configured_name in device_model:
                    is_significant_removal = True
            else:
                # No configured device - mark disconnected for any non-Unknown device
                if device_model != 'Unknown':
                    is_significant_removal = True

            if is_significant_removal:
                # Debounce: USB removal generates multiple events
                with self._hotplug_lock:
                    current_time = time.monotonic()
                    if current_time - self._last_hotplug_remove_time < 2.0:
                        return  # Skip duplicate
                    self._last_hotplug_remove_time = current_time

                with self._mic_state_lock:
                    self._mic_disconnected = True
                log(f"[HOTPLUG] Microphone disconnected")
                
                # Send notification on disconnect
                self._notify_user("hyprwhspr", "Microphone disconnected", "normal")

            # If currently recording, this will fail gracefully in next audio callback
        except Exception as e:
            log(f"[HOTPLUG] Error: {e}")

    def _on_pulse_default_changed(self, new_default_source):
        """Called when user changes system default microphone via PulseAudio/PipeWire"""
        try:
            log(f"[PULSE] Default source changed to: {new_default_source}")

            self.audio_capture.refresh_default_input("pulse_default_changed")
        except Exception as e:
            log(f"[PULSE] Error handling default source change: {e}")

    def _on_unrecoverable_audio_stream(self):
        """A wedged PortAudio stream survived every reclamation attempt (#209).

        Its native thread burns a core and floods stderr until the process
        dies, so under systemd (Restart=on-failure) exit and let it bring us
        back clean. Outside systemd, log and limp on.
        """
        from instance_detection import is_running_under_systemd
        if not is_running_under_systemd():
            log("[RECOVERY] ERROR: wedged audio stream cannot be reclaimed - "
                  "restart hyprwhspr to stop the CPU/log churn")
            return
        log("[RECOVERY] Wedged audio stream cannot be reclaimed - exiting for systemd restart")
        self._notify_user("hyprwhspr", "Audio system wedged - restarting service", "critical")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    def _on_pulse_server_restarted(self):
        """Called when PulseAudio/PipeWire server restarts"""
        try:
            log("[PULSE] Audio server restarted - recovering audio capture")

            # Give audio server time to fully initialize
            time.sleep(1)

            if self.audio_capture.recover_audio_capture('pulse_server_restart'):
                log("[PULSE] Recovery successful after server restart")
                self._write_recovery_result(True, 'pulse_restart')
            else:
                log("[PULSE] Recovery failed after server restart")
                self._write_recovery_result(False, 'pulse_restart')
        except Exception as e:
            log(f"[PULSE] Error handling server restart: {e}")

    def _handle_recovery(self, action):
        injector = getattr(self, 'text_injector', None)
        if injector is None:
            return False, 'Text delivery is still initializing'
        return injector.recover_last(action)

    def _write_recovery_result(self, success, reason):
        """Write recovery result to file for tray script notification"""
        # Use lock to prevent race conditions when multiple threads write results
        with self._recovery_result_lock:
            try:
                RECOVERY_RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)

                status = "success" if success else "failed"
                timestamp = int(time.time())

                with open(RECOVERY_RESULT_FILE, 'w') as f:
                    f.write(f"{status}:{reason}:{timestamp}")

                log(f"[RECOVERY] Result written: {status} ({reason})")

                # If recovery succeeded, clear any error state signals
                if success:
                    self._clear_error_state_signals()

            except Exception as e:
                log(f"[WARN] Failed to write recovery result: {e}")

    def _clear_error_state_signals(self):
        """Clear error state signal files after successful recovery"""
        try:
            # Clear mic zero volume signal
            if MIC_ZERO_VOLUME_FILE.exists():
                MIC_ZERO_VOLUME_FILE.unlink()
                log("[RECOVERY] Cleared mic_zero_volume error signal")

            # Clear any stale recovery request file
            if RECOVERY_REQUESTED_FILE.exists():
                RECOVERY_REQUESTED_FILE.unlink()

        except Exception as e:
            log(f"[WARN] Failed to clear error signals: {e}")

    def _attempt_recovery_if_needed(self):
        """
        Check for recovery request from tray script and attempt recovery once per error state.

        This is called periodically (e.g., in main loop) to check if recovery is needed.
        Only attempts recovery once per error state to avoid infinite retry loops.
        """
        # Check if recovery file exists
        if not RECOVERY_REQUESTED_FILE.exists():
            # No recovery requested - mic is working, reset flag
            if self.recovery_attempted.is_set():
                self.recovery_attempted.clear()
            return
        
        # Recovery file exists - check if we should attempt recovery
        # Don't trigger recovery if transcription is in progress
        if self.is_processing:
            return  # Skip recovery attempt during transcription
        
        # Don't trigger recovery if actively recording - recovery will interfere with recording
        if self.is_recording:
            return  # Skip recovery attempt during active recording
        
        # Check if recovery was already attempted for this error state
        if self.recovery_attempted.is_set():
            # Already attempted - don't try again
            return
        
        # Check file age - if very old (>60s), assume recovery was attempted and failed
        try:
            file_age = time.time() - RECOVERY_REQUESTED_FILE.stat().st_mtime
            if file_age > 60:
                # File is old - assume recovery was attempted and failed
                # Clear it to allow new error detection
                RECOVERY_REQUESTED_FILE.unlink()
                self.recovery_attempted.clear()
                return
        except Exception:
            pass

        # Clear the file now that we're about to attempt recovery
        try:
            RECOVERY_REQUESTED_FILE.unlink()
        except Exception as e:
            log(f"[RECOVERY] Warning: Could not clear recovery request file: {e}")
        
        # Determine reason for recovery
        was_recording = self.is_recording
        reason = "mic_unavailable" if not was_recording else "mic_no_audio"
        
        log(f"[RECOVERY] Recovery requested by tray script ({reason} detected)")
        
        # Mark that we're attempting recovery for this error state
        self.recovery_attempted.set()
        
        # Attempt recovery (will handle stopping current recording if needed)
        if self.audio_capture.recover_audio_capture(f"tray_script_request_{reason}"):
            log("[RECOVERY] Audio recovery successful - mic should now be available")

            # After successful audio recovery, also reinitialize model if needed
            # This handles suspend/resume cases where CUDA context is invalid
            model_reinit_success = self.whisper_manager.reinitialize_after_resume(only_if_idle=True)
            if not model_reinit_success:
                log("[RECOVERY] Model reinitialization failed after audio recovery")

            # Write recovery result for tray script.
            #
            # Important: even if model reinitialization fails, we still continue and attempt to
            # restore an in-progress recording session (was_recording). Otherwise recovery can
            # permanently drop the user's active recording state.
            if model_reinit_success:
                self._write_recovery_result(True, reason)
            else:
                self._write_recovery_result(False, 'suspend_resume_model')

            # Clear disconnected flag - microphone is back
            with self._mic_state_lock:
                self._mic_disconnected = False

            # Clear background recovery flag only if backend is healthy too.
            if model_reinit_success:
                self._background_recovery_needed.clear()

            # Reset flag since recovery succeeded
            self.recovery_attempted.clear()
            
            # If we were recording, we need to restart recording after recovery
            if was_recording:
                log("[RECOVERY] Restarting recording after successful recovery")
                # Get streaming callback if needed
                streaming_callback = self.whisper_manager.get_realtime_streaming_callback()
                try:
                    if not self.audio_capture.start_recording(streaming_callback=streaming_callback):
                        log("[RECOVERY] Failed to restart recording after recovery - start_recording() returned False")
                        self.is_recording = False
                        self._write_recording_status(False)
                        return
                    self._start_audio_level_monitoring()
                except Exception as e:
                    log(f"[RECOVERY] Failed to restart recording after recovery: {e}")
                    self.is_recording = False
                    self._write_recording_status(False)
        else:
            log("[RECOVERY] Recovery failed - please reseat your USB microphone")

            # Write recovery failure result for tray script
            self._write_recovery_result(False, reason)

            # Keep flag set - recovery was attempted and failed, don't retry

    def _on_system_suspend(self):
        """Called when system is about to suspend (D-Bus PrepareForSleep signal)"""
        try:
            log("[SUSPEND] System entering suspend")

            # Close WebSocket connections preemptively (avoid timeout errors)
            self.whisper_manager.close_realtime_connection("system suspend")
        except Exception as e:
            log(f"[SUSPEND] Error handling suspend: {e}")

    def _on_system_resume(self):
        """Called when system resumes from suspend (D-Bus PrepareForSleep signal)"""
        try:
            log("[SUSPEND] System resumed - recovering audio and backends...")
            time.sleep(2)  # Give audio/GPU drivers time to reinitialize

            # Re-attach the shortcut keyboard if suspend dropped it without a udev
            # 'add' event on resume. A delayed second pass catches a late reappear.
            self._resync_shortcut_keyboards("post_suspend_resume")

            if self.audio_capture.recover_audio_capture('post_suspend_resume'):
                # Reinitialize backend state (model / WebSocket) per backend type
                backend_reinit_success = self.whisper_manager.reinitialize_after_resume()
                if not backend_reinit_success:
                    log("[SUSPEND] Recovery failed - backend reinitialization failed")

                # Write recovery result and clear background recovery flag only after ALL recovery steps complete
                if backend_reinit_success:
                    log("[SUSPEND] Recovery successful - microphone ready")
                    self._write_recovery_result(True, 'suspend_resume')
                    with self._mic_state_lock:
                        self._mic_disconnected = False
                    self._background_recovery_needed.clear()
                else:
                    # Backend reinitialization failed - signal that recovery is still needed
                    if self.whisper_manager.active_backend_is_local():
                        self._write_recovery_result(False, 'suspend_resume_model')
                    else:
                        self._write_recovery_result(False, 'suspend_resume_websocket')
                    self._background_recovery_needed.set()
                    # Start background recovery thread
                    if self._background_recovery_thread is None or not self._background_recovery_thread.is_alive():
                        self._background_recovery_thread = threading.Thread(
                            target=self._background_recovery_retry,
                            daemon=True
                        )
                        self._background_recovery_thread.start()
            else:
                # Immediate recovery failed - start background retry
                log("[SUSPEND] Recovery failed - will retry in background (6 attempts over 30s)")
                self._background_recovery_needed.set()

                # Start background recovery thread
                if self._background_recovery_thread is None or not self._background_recovery_thread.is_alive():
                    self._background_recovery_thread = threading.Thread(
                        target=self._background_recovery_retry,
                        daemon=True
                    )
                    self._background_recovery_thread.start()
        except Exception as e:
            log(f"[SUSPEND] Error handling resume: {e}")

    def _background_recovery_retry(self):
        """
        Background thread that retries recovery after suspend/resume.
        Retries every 2 seconds for up to 12 seconds (6 attempts).
        """
        max_attempts = 6
        retry_interval = 2  # seconds

        for attempt in range(1, max_attempts + 1):
            # Check if we should stop (service shutting down or recovery no longer needed)
            if self._background_recovery_stop.is_set() or not self._background_recovery_needed.is_set():
                return

            # Check if hotplug recovery is running (it takes precedence)
            recovery_in_progress = False
            with self.audio_capture.recovery_lock:
                recovery_in_progress = self.audio_capture.recovery_in_progress

            if recovery_in_progress:
                time.sleep(1.0)  # Sleep outside lock to avoid blocking other threads
                if not self._background_recovery_needed.is_set():
                    return
                continue  # Skip this attempt, try again next iteration

            # Don't attempt recovery if user is actively recording/processing
            if self.is_recording or self.is_processing:
                # User activity proves system health - skip this attempt
                time.sleep(retry_interval)
                continue

            # Attempt recovery
            if self.audio_capture.recover_audio_capture(f'background_retry_{attempt}'):
                # Reinitialize backend state (model / WebSocket) per backend type
                backend_reinit_success = self.whisper_manager.reinitialize_after_resume()

                # Write recovery result only after ALL recovery steps complete
                if backend_reinit_success:
                    log("[RECOVERY] Background recovery successful - microphone ready")
                    self._write_recovery_result(True, 'background_retry')
                    with self._mic_state_lock:
                        self._mic_disconnected = False
                    self._background_recovery_needed.clear()
                    return  # Success, exit
                else:
                    # Backend reinitialization failed - continue retrying
                    # Don't write result yet - will retry or write failure after all attempts
                    pass

            # Recovery failed, wait before next attempt (unless this was the last attempt)
            if attempt < max_attempts:
                # Sleep in small increments to allow early exit if stop is signaled
                for _ in range(retry_interval):
                    if self._background_recovery_stop.is_set() or not self._background_recovery_needed.is_set():
                        return
                    time.sleep(1)

        # All attempts failed - check if system is actually healthy now
        if not self._background_recovery_needed.is_set():
            return

        # Only complain if system is still broken
        log("[RECOVERY] Background recovery exhausted - microphone may need manual reseat")
        self._write_recovery_result(False, 'background_retry_exhausted')
        self._background_recovery_needed.clear()
