"""Notifications, status files, mic OSD and the audio-level feed."""

import shutil
import threading
import time

from backend_utils import normalize_backend
from dependency_plan import missing_imports
from paths import (
    AUDIO_LEVEL_FILE, CONFIG_DIR, LONGFORM_STATE_FILE, MIC_ZERO_VOLUME_FILE, MODEL_UNLOADED_FILE,
    RECORDING_CONTROL_FILE, RECORDING_STATUS_FILE, RECOVERY_REQUESTED_FILE, RECOVERY_RESULT_FILE,
    RUNTIME_DIR, TRANSCRIPT_PREVIEW_FILE,
)
from service_log import log


class FeedbackMixin:
    """Notifications, status files, mic OSD and the audio-level feed."""

    def _notify_user(self, title: str, message: str, urgency: str = "normal"):
        """Send desktop notification if notify-send is available.

        Non-critical notifications auto-dismiss so they don't accumulate in the
        notification center; only genuine (critical) errors persist there until
        the user dismisses them. See desktop_notify for the why.
        """
        try:
            from desktop_notify import notify
            timeout = self.config.get_setting('notification_timeout_ms', 5000)
            notify(title, message, urgency=urgency, timeout_ms=timeout)
        except Exception:
            pass  # Silently fail if notify-send not available

    def _report_missing_dependencies(self):
        """Log and notify about Python modules the configured backend cannot import.

        Returns the missing module names.
        """
        backend = normalize_backend(self.config.get_setting('transcription_backend', 'pywhispercpp'))
        provider = self.config.get_setting('websocket_provider', None) if backend == 'realtime-ws' else None
        missing = missing_imports(backend, provider)
        if missing:
            names = ', '.join(missing)
            log(f"[ERROR] Missing Python modules: {names}")
            log("[ERROR] Dictation may fail until they are installed - run: hyprwhspr setup (Reinstall backend: yes)")
            self._notify_user(
                "hyprwhspr", f"Missing Python modules: {names}\n"
                "Run: hyprwhspr setup and reinstall the backend",
                urgency="critical")
        return missing

    def _mic_failure_message(self, fallback: str) -> str:
        """Pick user advice for a failed recording start based on what actually failed.

        Distinguishes "input device missing/initializing" (stream never opened)
        from "device wedged" (stream opened but delivers no callbacks) using the
        open outcome recorded by the capture thread; fallback covers the latter.
        """
        with self._mic_state_lock:
            if self._mic_disconnected:
                return "Microphone disconnected - please replug USB microphone"
        get_selection_error = getattr(self.audio_capture, 'get_input_selection_error', None)
        selection_error = get_selection_error() if callable(get_selection_error) else None
        if selection_error:
            return selection_error
        if not getattr(self.audio_capture, 'stream_opened', True):
            open_error = getattr(self.audio_capture, 'stream_open_error', None)
            if open_error:
                log(f"[ERROR] Stream open failed: {open_error}")
            return "Microphone unavailable - input device missing or still initializing - check the connection and try again"
        return fallback

    def _realtime_unavailable_message(self) -> str:
        """Explain why realtime couldn't start, based on what recovery actually hit.

        "try again in a moment" is only true while a handshake is in flight; for a
        failed connect it invites the user to keep retrying against a dead endpoint.
        """
        reason = self.whisper_manager.realtime_connect_failure()
        if reason == 'failed':
            return "Realtime connection failed — check network or provider status"
        if reason == 'cooldown':
            # Nothing retries in the background — the next attempt is the user's.
            return "Realtime connection failed — try again in a few seconds"
        return "Realtime backend not connected yet — try again in a moment."

    def _notify_zero_volume(self, message: str, log_level: str = "WARN"):
        """Log a mic failure, signal waybar, and show a coalesced desktop notification"""
        # Prevent duplicate handling of the same error within 2 seconds (user
        # might hit record twice). Lock protects the read-modify-write.
        with self._error_log_lock:
            current_time = time.monotonic()
            if (message == self._last_mic_error_message
                    and current_time - self._last_mic_error_log_time < 2.0):
                # Already handled this exact error recently, skip duplicate
                return
            self._last_mic_error_log_time = current_time
            self._last_mic_error_message = message

        # Print to logs (primary record)
        log(f"[{log_level}] {message}")

        # Direct desktop notification: environments without the waybar tray
        # (e.g. Niri) otherwise never see mic failures. Reusing replaces_id
        # coalesces repeats into a single replaced banner instead of stacking.
        try:
            from desktop_notify import send_notification_with_id
            nid = send_notification_with_id(
                "hyprwhspr", message, urgency="normal", timeout_ms=6000,
                replaces_id=self._mic_error_nid)
            if nid is not None:
                self._mic_error_nid = nid
        except Exception:
            pass  # Silently fail if no notification daemon

        # Write waybar signal file (atomic, no conflicts)
        # This allows waybar to detect when mic is present but not recording properly
        try:
            # Use atomic write (write to temp file, then rename)
            temp_file = MIC_ZERO_VOLUME_FILE.with_suffix('.tmp')
            temp_file.write_text(str(int(time.time())))
            temp_file.replace(MIC_ZERO_VOLUME_FILE)
        except Exception:
            pass  # Silently fail - waybar signal is optional

    def _clear_zero_volume_signal(self):
        """Clear zero-volume signal file when valid audio is detected"""
        try:
            if MIC_ZERO_VOLUME_FILE.exists():
                MIC_ZERO_VOLUME_FILE.unlink()
        except Exception:
            pass  # Silently fail - waybar signal cleanup is optional

        # Retire the mic-error banner: recording works (or is being retried),
        # so a stale error notification would just be noise
        nid = self._mic_error_nid
        self._mic_error_nid = None
        if nid is not None:
            try:
                from desktop_notify import close_notification
                close_notification(nid)
            except Exception:
                pass

    def _write_recording_status(self, is_recording):
        """Write recording status to file for tray script"""
        try:
            RECORDING_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)

            if is_recording:
                with open(RECORDING_STATUS_FILE, 'w') as f:
                    f.write('true')
            else:
                # Remove the file when not recording to avoid stale state
                if RECORDING_STATUS_FILE.exists():
                    RECORDING_STATUS_FILE.unlink()
        except Exception as e:
            log(f"[WARN] Failed to write recording status: {e}")

    def _migrate_legacy_state_files(self):
        """One-time cleanup of signal files that lived in CONFIG_DIR before they
        moved to RUNTIME_DIR, plus compat symlinks for the three files external
        consumers (Hyprland binds, GNOME extension) may still use at old paths.
        """
        try:
            RUNTIME_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
            RUNTIME_DIR.chmod(0o700)
        except Exception as e:
            log(f"[WARN] Failed to prepare runtime dir {RUNTIME_DIR}: {e}")

        legacy_names = [
            'recording_status', 'recording_control', 'hyprwhspr.sock',
            'audio_level', 'recovery_requested', 'recovery_result',
            '.mic_zero_volume', 'mic_osd.pid', '.suspend_marker',
            'hyprwhspr.lock', 'visualizer_state', 'longform_state',
            'model_unloaded', 'tray_state',
        ]
        for name in legacy_names:
            try:
                (CONFIG_DIR / name).unlink(missing_ok=True)
            except Exception:
                pass
        shutil.rmtree(CONFIG_DIR / '.recovery_notification_lock', ignore_errors=True)

        # Compat symlinks, kept for one release cycle
        compat = {
            'recording_control': RECORDING_CONTROL_FILE,
            'recording_status': RECORDING_STATUS_FILE,
            'audio_level': AUDIO_LEVEL_FILE,
        }
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log(f"[WARN] Failed to create config dir: {e}")
        for name, target in compat.items():
            try:
                (CONFIG_DIR / name).symlink_to(target)
            except Exception as e:
                log(f"[WARN] Failed to create legacy compat symlink {name}: {e}")

    def _reset_stale_state(self):
        """Clear runtime state files that may be stale from a previous session.

        If the service was killed (SIGKILL, crash, reboot), state files like
        recording_status can be left with stale values. This causes problems
        for external consumers (e.g. 'record toggle' reads recording_status
        to decide whether to send 'start' or 'stop', and a stale 'true' means
        toggle always sends 'stop' — so recording never starts).
        """
        stale_files = [
            RECORDING_STATUS_FILE,
            AUDIO_LEVEL_FILE,
            MIC_ZERO_VOLUME_FILE,
            RECOVERY_REQUESTED_FILE,
            RECOVERY_RESULT_FILE,
            MODEL_UNLOADED_FILE,
            TRANSCRIPT_PREVIEW_FILE,
        ]
        for f in stale_files:
            try:
                if f.exists():
                    f.unlink()
            except Exception:
                pass

        # Reset long-form state to IDLE (rather than deleting, since other
        # components may expect the file to exist with a valid state)
        try:
            if LONGFORM_STATE_FILE.exists():
                content = LONGFORM_STATE_FILE.read_text().strip()
                if content != 'IDLE':
                    LONGFORM_STATE_FILE.write_text('IDLE')
        except Exception:
            pass

    def _show_mic_osd(self):
        """Show mic-osd visualization overlay"""
        # Cancel any pending delayed-hide from a previous recording's _show_result_and_hide
        # so we don't hide the visualizer for this new recording
        with self._cancel_pending_hide_lock:
            self._cancel_pending_hide = True
        if self._mic_osd_runner and self._mic_osd_runner.is_available():
            self._mic_osd_runner.clear_preview_text()
            self._mic_osd_runner.set_state('recording')
            self._mic_osd_runner.show()

    def _hide_mic_osd(self):
        """Hide mic-osd visualization overlay"""
        runner = getattr(self, '_mic_osd_runner', None)
        if runner:
            try:
                runner.hide()
                runner.clear_state()
                runner.clear_preview_text()
            except Exception:
                pass

    def _set_mic_osd_preview_text(self, text: str):
        """Update live transcript preview text in the mic OSD."""
        runner = getattr(self, '_mic_osd_runner', None)
        if runner:
            try:
                runner.set_preview_text(text)
            except Exception:
                pass

    def _clear_mic_osd_preview_text(self):
        runner = getattr(self, '_mic_osd_runner', None)
        if runner:
            try:
                runner.clear_preview_text()
            except Exception:
                pass

    def _set_visualizer_state(self, state: str):
        """Set the visualizer state (recording, paused, processing, error, success)"""
        runner = getattr(self, '_mic_osd_runner', None)
        if runner:
            try:
                runner.set_state(state)
            except Exception:
                pass

    def _show_result_and_hide(self, success: bool):
        """Show success/error state then hide the OSD after a delay."""
        state = 'success' if success else 'error'
        self._set_visualizer_state(state)

        # Clear cancel so this scheduled hide is allowed to run (avoids inheriting
        # cancel from an earlier _show_mic_osd that already completed)
        with self._cancel_pending_hide_lock:
            self._cancel_pending_hide = False

        # Schedule hiding after 1.25 seconds (matches animation fade duration)
        def delayed_hide():
            time.sleep(1.25)
            with self._cancel_pending_hide_lock:
                should_hide = not self._cancel_pending_hide
            if not should_hide:
                return  # New recording started; don't hide
            self._hide_mic_osd()

        hide_thread = threading.Thread(target=delayed_hide, daemon=True)
        hide_thread.start()

    def _start_audio_level_monitoring(self):
        """Start monitoring and writing audio levels to file"""
        # Stop any lingering thread from a previous recording before starting a new one
        self._stop_audio_level_monitoring()

        self._audio_level_stop.clear()

        def monitor_audio_level():
            AUDIO_LEVEL_FILE.parent.mkdir(parents=True, exist_ok=True)

            # Muted mic detection: 5e-7 threshold catches true digital silence but not quiet rooms
            zero_samples = 0
            zero_threshold = 5e-7
            samples_to_cancel = 10  # 1 second at 100ms intervals
            grace_samples = 5  # Skip first 0.5s to let stream stabilize (avoids false mute on rapid toggle)
            total_samples = 0

            try:
                while self.is_recording and not self._audio_level_stop.is_set():
                    try:
                        # Get scaled level for visualization (0.0-1.0)
                        level = self.audio_capture.get_audio_level()
                        with open(AUDIO_LEVEL_FILE, 'w') as f:
                            f.write(f'{level:.3f}')

                        total_samples += 1

                        # Mute detection (only if enabled, after grace period)
                        if self.config.get_setting('mute_detection', True) and total_samples > grace_samples:
                            # get_audio_level() scales by 10x, so we need raw value for accurate detection
                            raw_level = self.audio_capture.current_level
                            if raw_level < zero_threshold:
                                zero_samples += 1
                                if zero_samples >= samples_to_cancel:
                                    self._cancel_recording_muted()
                                    return
                            else:
                                zero_samples = 0
                    except Exception as e:
                        # Rate-limit to avoid log spam on repeated failure
                        import time as _time
                        now = _time.monotonic()
                        if not hasattr(self, '_last_level_error_log') or now - self._last_level_error_log > 10.0:
                            log(f"[WARN] Audio level monitoring error: {e}")
                            self._last_level_error_log = now
                    # Sleep in small increments so the stop event wakes us quickly
                    self._audio_level_stop.wait(0.1)
            finally:
                # Clean up file when not recording (always runs, even on early return)
                try:
                    if AUDIO_LEVEL_FILE.exists():
                        AUDIO_LEVEL_FILE.unlink()
                except Exception:
                    pass

        self.audio_level_thread = threading.Thread(target=monitor_audio_level, daemon=True)
        self.audio_level_thread.start()

    def _stop_audio_level_monitoring(self):
        """Stop audio level monitoring and wait for thread to exit"""
        self._audio_level_stop.set()
        if self.audio_level_thread and self.audio_level_thread.is_alive():
            if threading.current_thread() is not self.audio_level_thread:
                # External caller: join and clear the reference only after the
                # thread (and its finally block) has actually finished.
                self.audio_level_thread.join(timeout=0.3)
                self.audio_level_thread = None
            # else: self-join — leave the reference intact.  The thread exits
            # immediately after returning here; the next _start call from the
            # main thread will find is_alive()==False (or join if still winding
            # down) and clear the reference before starting a new thread.
            # Nulling here would lose the reference and allow a new thread to
            # race against this thread's finally block on AUDIO_LEVEL_FILE.
        else:
            self.audio_level_thread = None
