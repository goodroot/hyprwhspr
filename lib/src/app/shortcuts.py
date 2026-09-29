"""Global shortcuts: setup, press/release policy and push-to-talk latching."""

import threading
import time

from global_shortcuts import GlobalShortcuts
from service_log import log


class ShortcutsMixin:
    """Global shortcuts: setup, press/release policy and push-to-talk latching."""

    def _setup_global_shortcuts(self):
        """Initialize global keyboard shortcuts"""
        # Check if using Hyprland compositor bindings instead
        use_hypr_bindings = self.config.get_setting("use_hypr_bindings", False)
        if use_hypr_bindings:
            log("[INFO] Using Hyprland compositor bindings (evdev shortcuts disabled)")
            log("[INFO] Configure bindings in ~/.config/hypr/hyprland.conf")
            log("[INFO] Use ~/.config/hyprwhspr/recording_control file API for control")
            self.global_shortcuts = None
            return

        try:
            shortcut_key = self.config.get_setting("primary_shortcut", "Super+Alt+D")
            recording_mode = self.config.get_setting("recording_mode", "toggle")
            grab_keys = self.config.get_setting("grab_keys", False)
            selected_device_path = self.config.get_setting("selected_device_path", None)
            selected_device_name = self.config.get_setting("selected_device_name", None)
            keyboard_device_names = self.config.get_setting("keyboard_device_names", None)
            keyboard_hotplug = self.config.get_setting("keyboard_hotplug", True)

            # Register callbacks based on recording mode
            # Validate recording_mode and only register release callback for modes that need it
            if recording_mode in ('toggle', 'continuous'):
                # Toggle/continuous mode: only register press callback
                self.global_shortcuts = GlobalShortcuts(
                    shortcut_key,
                    self._on_shortcut_triggered,
                    None,  # No release callback for toggle modes
                    device_path=selected_device_path,
                    device_name=selected_device_name,
                    grab_keys=grab_keys,
                    keyboard_device_names=keyboard_device_names,
                    keyboard_hotplug=keyboard_hotplug,
                )
            elif recording_mode in ('push_to_talk', 'auto'):
                # Push-to-talk and auto modes: register both press and release callbacks
                self.global_shortcuts = GlobalShortcuts(
                    shortcut_key,
                    self._on_shortcut_triggered,
                    self._on_shortcut_released,
                    device_path=selected_device_path,
                    device_name=selected_device_name,
                    grab_keys=grab_keys,
                    keyboard_device_names=keyboard_device_names,
                    keyboard_hotplug=keyboard_hotplug,
                )
            elif recording_mode == 'long_form':
                # Long-form mode: primary key toggles recording/paused, no release callback
                self.global_shortcuts = GlobalShortcuts(
                    shortcut_key,
                    self._longform.primary_shortcut,
                    None,  # No release callback for long_form mode
                    device_path=selected_device_path,
                    device_name=selected_device_name,
                    grab_keys=grab_keys,
                    keyboard_device_names=keyboard_device_names,
                    keyboard_hotplug=keyboard_hotplug,
                )
                self._longform.ensure_initialized()
            else:
                # Invalid mode: default to toggle behavior (no release callback)
                log(f"[WARNING] Invalid recording_mode '{recording_mode}', defaulting to 'toggle'")
                self.global_shortcuts = GlobalShortcuts(
                    shortcut_key,
                    self._on_shortcut_triggered,
                    None,  # No release callback for invalid modes (treated as toggle)
                    device_path=selected_device_path,
                    device_name=selected_device_name,
                    grab_keys=grab_keys,
                    keyboard_device_names=keyboard_device_names,
                    keyboard_hotplug=keyboard_hotplug,
                )
        except Exception as e:
            log(f"[ERROR] Failed to initialize global shortcuts: {e}")
            self.global_shortcuts = None

        # Set up secondary shortcut if configured
        try:
            secondary_shortcut_key = self.config.get_setting("secondary_shortcut", None)
            if secondary_shortcut_key:
                secondary_language = self.config.get_setting("secondary_language", None)
                if secondary_language:
                    # Register callbacks based on recording mode (same as primary)
                    if recording_mode in ('toggle', 'continuous'):
                        self.secondary_shortcuts = GlobalShortcuts(
                            secondary_shortcut_key,
                            self._on_secondary_shortcut_triggered,
                            None,  # No release callback for toggle modes
                            device_path=selected_device_path,
                            device_name=selected_device_name,
                            grab_keys=grab_keys,
                            keyboard_device_names=keyboard_device_names,
                            keyboard_hotplug=keyboard_hotplug,
                        )
                    elif recording_mode in ('push_to_talk', 'auto'):
                        self.secondary_shortcuts = GlobalShortcuts(
                            secondary_shortcut_key,
                            self._on_secondary_shortcut_triggered,
                            self._on_secondary_shortcut_released,
                            device_path=selected_device_path,
                            device_name=selected_device_name,
                            grab_keys=grab_keys,
                            keyboard_device_names=keyboard_device_names,
                            keyboard_hotplug=keyboard_hotplug,
                        )
                    else:
                        # Invalid mode: default to toggle behavior
                        self.secondary_shortcuts = GlobalShortcuts(
                            secondary_shortcut_key,
                            self._on_secondary_shortcut_triggered,
                            None,
                            device_path=selected_device_path,
                            device_name=selected_device_name,
                            grab_keys=grab_keys,
                            keyboard_device_names=keyboard_device_names,
                            keyboard_hotplug=keyboard_hotplug,
                        )
                    
                    # Start the secondary shortcuts
                    if self.secondary_shortcuts.start():
                        log(f"[INFO] Secondary shortcut registered: {secondary_shortcut_key} (language: {secondary_language})")
                    else:
                        log(f"[WARNING] Failed to start secondary shortcut: {secondary_shortcut_key}")
                        self.secondary_shortcuts = None
                else:
                    log("[WARNING] secondary_shortcut configured but secondary_language is not set. Secondary shortcut disabled.")
        except Exception as e:
            log(f"[ERROR] Failed to initialize secondary shortcuts: {e}")
            self.secondary_shortcuts = None

        # Set up cancel shortcut if configured
        try:
            cancel_shortcut_key = self.config.get_setting("cancel_shortcut", None)
            if cancel_shortcut_key:
                self._cancel_shortcuts = GlobalShortcuts(
                    cancel_shortcut_key,
                    self._on_cancel_shortcut_triggered,
                    None,  # No release callback
                    device_path=selected_device_path,
                    device_name=selected_device_name,
                    grab_keys=grab_keys,
                    keyboard_device_names=keyboard_device_names,
                    keyboard_hotplug=keyboard_hotplug,
                )
                if self._cancel_shortcuts.start():
                    log(f"[INFO] Cancel shortcut registered: {cancel_shortcut_key}")
                else:
                    log(f"[WARNING] Failed to start cancel shortcut: {cancel_shortcut_key}")
                    self._cancel_shortcuts = None
        except Exception as e:
            log(f"[ERROR] Failed to initialize cancel shortcut: {e}")
            self._cancel_shortcuts = None

        # Set up submit shortcut for long-form mode
        if recording_mode == 'long_form':
            try:
                submit_shortcut_key = self.config.get_setting("long_form_submit_shortcut", None)
                if submit_shortcut_key:
                    self._longform_submit_shortcuts = GlobalShortcuts(
                        submit_shortcut_key,
                        self._longform.submit_shortcut,
                        None,  # No release callback
                        device_path=selected_device_path,
                        device_name=selected_device_name,
                        grab_keys=grab_keys,
                        keyboard_device_names=keyboard_device_names,
                        keyboard_hotplug=keyboard_hotplug,
                    )
                    if self._longform_submit_shortcuts.start():
                        log(f"[INFO] Long-form submit shortcut registered: {submit_shortcut_key}")
                    else:
                        log(f"[WARNING] Failed to start long-form submit shortcut: {submit_shortcut_key}")
                        self._longform_submit_shortcuts = None
                else:
                    log("[WARNING] long_form mode enabled but long_form_submit_shortcut not set")
            except Exception as e:
                log(f"[ERROR] Failed to initialize long-form submit shortcut: {e}")
                self._longform_submit_shortcuts = None

    def _on_shortcut_triggered(self):
        """Handle global shortcut trigger (key press)"""
        self._handle_shortcut_triggered()

    def _handle_shortcut_triggered(self, language_override=None):
        """Shared logic for handling shortcut trigger with optional language override"""
        recording_mode = self.config.get_setting("recording_mode", "toggle")

        if recording_mode in ('toggle', 'continuous'):
            # Toggle/continuous mode: start/stop recording
            if self.is_recording:
                if recording_mode == 'continuous':
                    self._continuous_stop_and_wait()
                self._stop_recording()
            else:
                self._start_recording(language_override=language_override)
                if recording_mode == 'continuous':
                    self._continuous_start_silence_monitor()
                elif recording_mode == 'toggle':
                    self._autostop_start_silence_monitor()
        elif recording_mode == 'push_to_talk':
            # A latched session ends on the next press; an unlocked live recording
            # (e.g. started from the tray) just restarts the hold clock.
            if not self.is_recording:
                self._start_recording(language_override=language_override)
            elif self._ptt_locked:
                log("[CONTROL] Locked push-to-talk session ended by key press")
                self._stop_recording()
            else:
                self._ptt_mark_press()
        elif recording_mode == 'auto':
            # Auto mode (hybrid tap/hold): record timestamp and start if not recording
            # Synchronize access to state variables to prevent race conditions
            # Don't call _start_recording() inside the lock to avoid blocking release callback
            # Initialize state variables if they're None (e.g., if mode was changed from non-auto)
            with self._auto_mode_lock:
                # Ensure variables are initialized (handles mode change from non-auto to auto)
                if self._shortcut_press_time is None:
                    self._shortcut_press_time = 0.0
                    self._recording_started_this_press = False
                    self._tap_threshold = 0.4

                self._shortcut_press_time = time.time()
                if not self.is_recording:
                    self._recording_started_this_press = True
                    should_start = True
                else:
                    # Already recording - will be stopped on release if this is a tap
                    self._recording_started_this_press = False
                    should_start = False

            # Call _start_recording() outside the lock to avoid blocking release callback
            # NOTE: auto-stop-on-silence is armed on RELEASE (tap-confirm path), not here, so a
            # >=400ms hold stays pure push-to-talk and is never cut off mid-hold.
            if should_start:
                self._start_recording(language_override=language_override)
        else:
            # Invalid mode, default to toggle behavior
            if self.is_recording:
                self._stop_recording()
            else:
                self._start_recording(language_override=language_override)

    def _on_shortcut_released(self):
        """Handle global shortcut release (key release)
        
        Only called for 'push_to_talk' and 'auto' modes (not 'toggle')
        """
        recording_mode = self.config.get_setting("recording_mode", "toggle")
        
        if recording_mode == 'push_to_talk':
            if self.is_recording and not self._ptt_release_latches():
                self._stop_recording()
        elif recording_mode == 'auto':
            # Auto mode (hybrid tap/hold): determine behavior based on hold duration
            if not self.is_recording:
                return
            
            # Synchronize access to state variables to prevent race conditions
            # Calculate hold_duration inside the lock to ensure consistent timing
            with self._auto_mode_lock:
                press_time = self._shortcut_press_time
                started_this_press = self._recording_started_this_press
                release_time = time.time()  # Capture release time while holding lock
                
                # Validate press_time is not None (handles mode change from non-auto to auto)
                if press_time is None:
                    # State not initialized - treat as hold (stop recording)
                    self._stop_recording()
                    return
                
                hold_duration = release_time - press_time
                tap_threshold = self._tap_threshold if self._tap_threshold is not None else 0.4

            if hold_duration >= tap_threshold:
                # Hold (>= 400ms): always stop recording (push-to-talk behavior)
                self._stop_recording()
            else:
                # Tap (< 400ms): only stop if we didn't start recording on this press (toggle off)
                if not started_this_press:
                    self._stop_recording()
                else:
                    # Tap started the session and we're keeping it: arm auto-stop-on-silence now
                    # (deferred from press so a >=400ms hold stays pure push-to-talk).
                    self._autostop_start_silence_monitor()

    def _on_secondary_shortcut_triggered(self):
        """Handle secondary shortcut trigger (key press) with language override"""
        secondary_language = self.config.get_setting("secondary_language", None)
        self._handle_shortcut_triggered(language_override=secondary_language)

    # Secondary release is identical to primary release - reuse the same handler
    _on_secondary_shortcut_released = _on_shortcut_released

    def _ptt_reset(self):
        """Drop push-to-talk hold state. Caller holds _recording_lock."""
        self._ptt_press_time = None
        self._ptt_locked = False

    def _ptt_mark_press(self):
        """Measure the hold from this press, for a recording already running."""
        with self._recording_lock:
            self._ptt_press_time = time.monotonic()

    def _ptt_release_latches(self):
        """True when this release latches the session, or it is already latched.

        Consumes the pending press. A latched session ends only on the next
        press, never a release: evdev runs press and release callbacks on
        separate threads, so a quick tap's release can arrive first.
        """
        lock_seconds = self._get_float_setting('push_to_talk_lock_seconds', 0.0)
        if lock_seconds <= 0:
            return False
        now = time.monotonic()
        with self._recording_lock:
            if self._ptt_locked:
                return True
            press_time = self._ptt_press_time
            if press_time is None:
                return False
            self._ptt_press_time = None
            held = now - press_time
            if held < lock_seconds:
                return False
            self._ptt_locked = True
        log(f"[CONTROL] Push-to-talk held {held:.1f}s (>= {lock_seconds:.1f}s) - recording locked")
        self._notify_user("hyprwhspr", "Push-to-talk locked - press again to stop the recording", urgency="low")
        return True

    def _on_cancel_shortcut_triggered(self):
        """Handle cancel shortcut trigger - discard recording without transcribing"""
        recording_mode = self.config.get_setting("recording_mode", "toggle")
        if recording_mode == "long_form":
            self._longform.cancel_shortcut()
        else:
            if recording_mode == "continuous":
                self._continuous_cancelled = True
                self._continuous_stop_silence_monitor()
            self._cancel_recording()

    def _resync_shortcut_keyboards(self, reason: str):
        """Re-attach dropped keyboards across all shortcut handlers after resume.

        Each handler (primary, secondary, cancel, long-form submit) runs its own
        device list and independently loses the keyboard on suspend, so all of
        them need resyncing. The device node may reappear a second or two after
        resume, so a single immediate pass can miss it; a delayed pass follows.
        Best-effort; never raises into the caller.
        """
        def _resync_all(suffix: str):
            handlers = (
                self.global_shortcuts,
                self.secondary_shortcuts,
                self._cancel_shortcuts,
                self._longform_submit_shortcuts,
            )
            for handler in handlers:
                if handler is None:
                    continue
                try:
                    handler.resync_devices(f"{reason}{suffix}")
                except Exception as e:
                    log(f"[SUSPEND] Keyboard resync failed: {e}")

        _resync_all("")

        def _delayed():
            time.sleep(2.0)
            _resync_all("_delayed")
        threading.Thread(target=_delayed, daemon=True).start()
