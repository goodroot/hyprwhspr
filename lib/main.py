#!/usr/bin/env python3
"""
hyprwhspr - stt
"""

import sys
import time
import math
import threading
import os
import socket
import fcntl
import atexit
import subprocess
import select
import shutil
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None  # Will be checked when needed

# Whisper hallucination markers for silence/noise segments
_HALLUCINATION_MARKERS = {
    'blank audio', 'blank', 'silence', 'no speech',
    'you', 'thank you', 'thanks for watching', 'thank you for watching',
    'video playback', 'music', 'music playing', 'keyboard clicking',
}


def _looks_like_wlroots_session() -> bool:
    desktop = ':'.join([
        os.environ.get('XDG_CURRENT_DESKTOP', ''),
        os.environ.get('XDG_SESSION_DESKTOP', ''),
        os.environ.get('DESKTOP_SESSION', ''),
    ]).lower()
    tokens = set(filter(None, desktop.replace('-', ':').replace('_', ':').split(':')))
    return bool(
        tokens & {'hyprland', 'sway', 'river', 'wayfire', 'labwc'}
        or os.environ.get('HYPRLAND_INSTANCE_SIGNATURE')
        or os.environ.get('SWAYSOCK')
    )

# Ensure unbuffered output for journald logging
if sys.stdout.isatty():
    # Interactive terminal - keep buffering
    pass
else:
    # Non-interactive (systemd/journald) - unbuffer
    # Note: reconfigure() was added in Python 3.7, and may not exist on all stdout/stderr objects
    # We use try/except to handle cases where it's not available
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass  # Fall back to PYTHONUNBUFFERED environment variable
    
    try:
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass  # Fall back to PYTHONUNBUFFERED environment variable

# Add the lib directory to the Python path (for mic_osd imports)
lib_path = Path(__file__).parent
sys.path.insert(0, str(lib_path))
# Add the src directory to the Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

# Lock file for preventing multiple instances
_lock_file = None
_lock_file_path = None

from config_manager import ConfigManager
from audio_capture import AudioCapture
from whisper_manager import WhisperManager
from session_environment import ensure_wayland_display
from text_injector import TextInjector
from global_shortcuts import GlobalShortcuts
from audio_manager import AudioManager
from audio_ducker import AudioDucker
from device_monitor import DeviceMonitor, PYUDEV_AVAILABLE
from paths import (
    RECORDING_STATUS_FILE, RECORDING_CONTROL_FILE, AUDIO_LEVEL_FILE, RECOVERY_REQUESTED_FILE,
    RECOVERY_RESULT_FILE, MIC_ZERO_VOLUME_FILE, LOCK_FILE, LONGFORM_STATE_FILE, LONGFORM_SEGMENTS_DIR,
    MODEL_UNLOADED_FILE, SOCKET_FILE, TRANSCRIPT_PREVIEW_FILE, CONFIG_DIR, RUNTIME_DIR
)
from backend_utils import normalize_backend
from longform_controller import LongFormController

class hyprwhsprApp:
    """Main application class for hyprwhspr voice dictation (Headless Mode)"""

    def __init__(self):
        ensure_wayland_display()

        # Initialize core components
        self.config = ConfigManager()

        # Initialize audio capture with configured device
        audio_device_id = self.config.get_setting('audio_device_id', None)
        self.audio_capture = AudioCapture(device_id=audio_device_id, config_manager=self.config)
        self.audio_capture.on_unrecoverable_stream = self._on_unrecoverable_audio_stream

        # Initialize audio feedback manager
        self.audio_manager = AudioManager(self.config)

        # Initialize audio ducker for reducing system volume during recording
        ducking_percent = self.config.get_setting('audio_ducking_percent', 50)
        self.audio_ducker = AudioDucker(reduction_percent=ducking_percent)

        # Initialize whisper manager with shared config
        self.whisper_manager = WhisperManager(config_manager=self.config)
        self.text_injector = TextInjector(self.config)
        self.global_shortcuts = None
        self.secondary_shortcuts = None
        self._cancel_shortcuts = None

        # Application state
        self.is_recording = False
        self._current_language_override = None  # Language override for current recording session
        self.is_processing = False
        self.current_transcription = ""
        self.audio_level_thread = None
        self._audio_level_stop = threading.Event()  # Signals audio level thread to exit immediately
        self.recovery_attempted = threading.Event()  # Thread-safe flag: track if recovery was attempted for current error state
        self.last_recovery_time = 0.0  # Track when recovery last completed (for cooldown)
        self._last_mic_error_log_time = 0.0  # Track when we last logged mic error (prevent duplicates)
        self._last_mic_error_message = None  # Last mic error message (dedupe identical repeats only)
        self._mic_error_nid = None  # Desktop notification id for coalescing mic-error banners
        self._mic_disconnected = False  # Track if microphone was disconnected via hotplug event
        self._last_hotplug_add_time = float('-inf')  # Track last USB add event (for debouncing multiple events)
        
        # Lock to prevent concurrent recording starts (race condition protection)
        self._recording_lock = threading.Lock()

        # Lock for auto mode state variables (protects against race conditions between trigger/release callbacks)
        self._auto_mode_lock = threading.Lock()
        
        # Lock for error logging deduplication (protects read-modify-write on _last_mic_error_log_time)
        self._error_log_lock = threading.Lock()
        
        # Lock for hotplug event debouncing (protects read-modify-write on _last_hotplug_add_time and _last_hotplug_remove_time)
        self._hotplug_lock = threading.Lock()
        self._last_hotplug_remove_time = float('-inf')  # Last time we processed a device removal

        # Lock for microphone disconnect state (protects _mic_disconnected flag)
        self._mic_state_lock = threading.Lock()

        # Lock for recovery result writes (prevents race conditions when multiple threads write results)
        self._recovery_result_lock = threading.Lock()

        # Cancel pending delayed-hide from _show_result_and_hide when a new recording starts
        self._cancel_pending_hide = False
        self._cancel_pending_hide_lock = threading.Lock()

        # Background recovery retry state (for suspend/resume)
        self._background_recovery_needed = threading.Event()  # Signal that recovery should be retried
        self._background_recovery_thread = None  # Background thread handle
        self._background_recovery_stop = threading.Event()  # Signal to stop background recovery

        # Set when model is loading in background (e.g. slow backends like cohere-transcribe)
        # Recording is blocked while True; cleared once initialize() succeeds or fails.
        self._model_initializing = False
        # Set when background initialize() failed; record start retries init instead
        # of recording audio that can never be transcribed
        self._backend_init_failed = False
        self._backend_init_lock = threading.Lock()

        # Recording control FIFO (for immediate push-to-talk response)
        self._recording_control_thread = None  # Background thread handle for FIFO listener
        self._recording_control_stop = threading.Event()  # Signal to stop FIFO listener

        # Capture socket state (for capture on the recording - streams transcription back to client)
        self._capture_socket_thread = None  # Background thread handle for accepting socket connections
        self._capture_socket_stop = threading.Event()  # Signal to stop for capture socket thread
        self._capture_subscriber = None  # Active client connection, or None
        self._capture_subscriber_lock = threading.Lock()  # Guards claim and release of subscriber connection
        self._capture_subscriber_done = threading.Event()  # Set by notifier so that handler closes cleanly

        # Hybrid tap/hold mode state tracking (auto mode)
        recording_mode = self.config.get_setting('recording_mode', 'toggle')
        if recording_mode == 'auto':
            self._shortcut_press_time = 0.0
            self._recording_started_this_press = False
            self._tap_threshold = 0.4  # 400ms - shorter than this is a "tap", longer is a "hold"
        else:
            # Initialize to None to avoid AttributeError if accidentally accessed
            self._shortcut_press_time = None
            self._recording_started_this_press = None
            self._tap_threshold = None

        # Continuous mode state (auto-paste on speech pause)
        self._continuous_silence_thread = None
        self._continuous_silence_stop = threading.Event()
        self._continuous_flush_lock = threading.Lock()
        self._continuous_transcription_done = threading.Event()
        self._continuous_transcription_done.set()  # no transcription in flight
        self._continuous_cancelled = False  # set on cancel to suppress in-flight injection

        # Auto-stop-on-silence state (toggle/auto modes). The stop Event is created fresh
        # per session (not reused) so a stale monitor generation can never signal a newer one.
        self._autostop_silence_thread = None
        self._autostop_silence_stop = None
        self._autostop_lock = threading.Lock()  # guards thread/event bookkeeping across threads

        # Long-form state and transitions live in an isolated controller.
        self._longform = LongFormController(
            config=self.config,
            audio_capture=self.audio_capture,
            audio_manager=self.audio_manager,
            whisper_manager=self.whisper_manager,
            inject_text=self._inject_text,
            notify_capture=self._notify_capture_subscriber,
            set_visualizer_state=self._set_visualizer_state,
            show_mic_osd=self._show_mic_osd,
            hide_mic_osd=self._hide_mic_osd,
            show_result_and_hide=self._show_result_and_hide,
            write_recording_status=self._write_recording_status,
            set_processing=self._set_longform_processing,
            hallucination_markers=_HALLUCINATION_MARKERS,
        )
        self._longform_submit_shortcuts = None  # Submit shortcut handler

        # Track startup time BEFORE any monitors are initialized
        # This prevents race condition where hotplug events arrive before _startup_time is set
        self._startup_time = time.monotonic()
        self._startup_grace_period = 5.0  # Ignore hotplug events for 5 seconds after startup

        # Clear stale runtime state from any previous session (crash, SIGKILL, reboot)
        self._migrate_legacy_state_files()
        self._reset_stale_state()

        # Set up device hotplug monitoring (for automatic mic recovery)
        self._setup_device_monitor()

        # Set up PulseAudio/PipeWire event monitoring
        self._setup_pulse_monitor()

        # Set up suspend/resume monitoring
        self._setup_suspend_monitor()

        # Set up recording control FIFO (for immediate push-to-talk response)
        self._setup_recording_control_fifo()

        # Pre-initialize mic-osd daemon (eliminates latency on recording)
        self._mic_osd_runner = None
        if self.config.get_setting('mic_osd_enabled', True):
            try:
                from mic_osd import MicOSDRunner, NotificationPresenter
                def _use_notification_status(reason: str):
                    if _looks_like_wlroots_session():
                        print(f"[INIT] {reason}, falling back to notifications", flush=True)
                    presenter = NotificationPresenter(
                        active_timeout_ms=self.config.get_setting('notification_timeout_ms', 5000))
                    if presenter.is_available():
                        self._mic_osd_runner = presenter
                        print("[INIT] Recording status via notifications", flush=True)
                    else:
                        print("[WARN] No layer-shell overlay and no desktop notifications; recording has no status indicator", flush=True)

                if MicOSDRunner.is_available() and MicOSDRunner.layer_shell_active():
                    # Feed the OSD meter from the capture stream (issue #205).
                    # get_viz_frame's default num_buckets must match the
                    # waveform visualization's num_bars (32).
                    runner = MicOSDRunner(
                        level_source=self.audio_capture.get_viz_frame,
                        style=self.config.get_setting('mic_osd_style', 'waveform'),
                    )
                    if runner._ensure_daemon():  # Start daemon now
                        self._mic_osd_runner = runner
                        print("[INIT] Mic-OSD daemon started", flush=True)
                    else:
                        print("[WARN] Failed to start mic-osd daemon", flush=True)
                        _use_notification_status("mic-osd daemon failed to start")
                else:
                    # No layer-shell (e.g. GNOME/Mutter): the overlay would steal
                    # keyboard focus and swallow the paste keystroke. Show recording
                    # status via desktop notifications instead.
                    _use_notification_status("layer-shell not supported")
            except Exception as e:
                print(f"[WARN] Failed to initialize recording status indicator: {e}", flush=True)
                import traceback
                traceback.print_exc()

        if hasattr(self.whisper_manager, 'set_realtime_partial_callback'):
            self.whisper_manager.set_realtime_partial_callback(self._set_mic_osd_preview_text)

        # Set up global shortcuts (needed for headless operation)
        self._setup_global_shortcuts()

    def _setup_global_shortcuts(self):
        """Initialize global keyboard shortcuts"""
        # Check if using Hyprland compositor bindings instead
        use_hypr_bindings = self.config.get_setting("use_hypr_bindings", False)
        if use_hypr_bindings:
            print("[INFO] Using Hyprland compositor bindings (evdev shortcuts disabled)", flush=True)
            print("[INFO] Configure bindings in ~/.config/hypr/hyprland.conf", flush=True)
            print("[INFO] Use ~/.config/hyprwhspr/recording_control file API for control", flush=True)
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
                print(f"[WARNING] Invalid recording_mode '{recording_mode}', defaulting to 'toggle'")
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
            print(f"[ERROR] Failed to initialize global shortcuts: {e}", flush=True)
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
                        print(f"[INFO] Secondary shortcut registered: {secondary_shortcut_key} (language: {secondary_language})", flush=True)
                    else:
                        print(f"[WARNING] Failed to start secondary shortcut: {secondary_shortcut_key}", flush=True)
                        self.secondary_shortcuts = None
                else:
                    print("[WARNING] secondary_shortcut configured but secondary_language is not set. Secondary shortcut disabled.", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed to initialize secondary shortcuts: {e}", flush=True)
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
                    print(f"[INFO] Cancel shortcut registered: {cancel_shortcut_key}", flush=True)
                else:
                    print(f"[WARNING] Failed to start cancel shortcut: {cancel_shortcut_key}", flush=True)
                    self._cancel_shortcuts = None
        except Exception as e:
            print(f"[ERROR] Failed to initialize cancel shortcut: {e}", flush=True)
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
                        print(f"[INFO] Long-form submit shortcut registered: {submit_shortcut_key}", flush=True)
                    else:
                        print(f"[WARNING] Failed to start long-form submit shortcut: {submit_shortcut_key}", flush=True)
                        self._longform_submit_shortcuts = None
                else:
                    print("[WARNING] long_form mode enabled but long_form_submit_shortcut not set", flush=True)
            except Exception as e:
                print(f"[ERROR] Failed to initialize long-form submit shortcut: {e}", flush=True)
                self._longform_submit_shortcuts = None

    def _setup_device_monitor(self):
        """Initialize device hotplug monitoring for automatic microphone recovery"""
        if PYUDEV_AVAILABLE:
            self.device_monitor = DeviceMonitor(
                on_audio_add=self._on_audio_device_added,
                on_audio_remove=self._on_audio_device_removed
            )
            if self.device_monitor.start():
                print("[INIT] Device hotplug monitoring enabled")
            else:
                print("[WARN] Failed to start device hotplug monitoring")
                self.device_monitor = None
        else:
            self.device_monitor = None
            print("[WARN] pyudev not available - audio hotplug detection disabled")

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
                print("[INIT] PulseAudio/PipeWire monitoring enabled")
            else:
                print("[WARN] Failed to start PulseAudio monitoring")
                self.pulse_monitor = None
        except ImportError:
            self.pulse_monitor = None
            print("[WARN] pulsectl not available - pulse monitoring disabled")
        except Exception as e:
            self.pulse_monitor = None
            print(f"[WARN] Failed to setup pulse monitor: {e}")

    def _setup_suspend_monitor(self):
        """Initialize suspend/resume monitoring via D-Bus"""
        try:
            from src.suspend_monitor import SuspendMonitor
            self.suspend_monitor = SuspendMonitor(
                on_suspend_callback=self._on_system_suspend,
                on_resume_callback=self._on_system_resume
            )
            if self.suspend_monitor.start():
                print("[INIT] Suspend/resume monitoring enabled (D-Bus)")
            else:
                print("[WARN] Failed to start suspend monitoring")
                self.suspend_monitor = None
        except ImportError:
            self.suspend_monitor = None
            print("[WARN] D-Bus/GLib not available - suspend monitoring disabled")
        except Exception as e:
            self.suspend_monitor = None
            print(f"[WARN] Failed to setup suspend monitor: {e}")

    def _on_audio_device_added(self, device):
        """Called when audio device is plugged in"""
        try:
            # Ignore hotplug events during startup grace period
            # This prevents false positives from pyudev detecting existing devices on startup
            current_time = time.monotonic()
            if current_time - self._startup_time < self._startup_grace_period:
                remaining = self._startup_grace_period - (current_time - self._startup_time)
                print(f"[HOTPLUG] Ignoring hotplug event during startup grace period ({remaining:.1f}s remaining)", flush=True)
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

                print(f"[HOTPLUG] Microphone detected - recovering...", flush=True)
                time.sleep(0.5)  # Let drivers settle

                # Trigger recovery
                if self.audio_capture.recover_audio_capture('hotplug_detected'):
                    print(f"[HOTPLUG] Recovery successful", flush=True)
                    self._write_recovery_result(True, 'hotplug')
                    with self._mic_state_lock:
                        self._mic_disconnected = False
                    self._background_recovery_needed.clear()
                else:
                    print(f"[HOTPLUG] Recovery failed - will retry in background", flush=True)
                    self._write_recovery_result(False, 'hotplug')
                    # Re-set flag so background recovery can retry
                    self._background_recovery_needed.set()
        except Exception as e:
            print(f"[HOTPLUG] Error: {e}", flush=True)

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
                print(f"[HOTPLUG] Microphone disconnected", flush=True)
                
                # Send notification on disconnect
                self._notify_user("hyprwhspr", "Microphone disconnected", "normal")

            # If currently recording, this will fail gracefully in next audio callback
        except Exception as e:
            print(f"[HOTPLUG] Error: {e}", flush=True)

    def _on_pulse_default_changed(self, new_default_source):
        """Called when user changes system default microphone via PulseAudio/PipeWire"""
        try:
            print(f"[PULSE] Default source changed to: {new_default_source}", flush=True)

            self.audio_capture.refresh_default_input("pulse_default_changed")
        except Exception as e:
            print(f"[PULSE] Error handling default source change: {e}", flush=True)

    def _on_unrecoverable_audio_stream(self):
        """A wedged PortAudio stream survived every reclamation attempt (#209).

        Its native thread burns a core and floods stderr until the process
        dies, so under systemd (Restart=on-failure) exit and let it bring us
        back clean. Outside systemd, log and limp on.
        """
        from instance_detection import is_running_under_systemd
        if not is_running_under_systemd():
            print("[RECOVERY] ERROR: wedged audio stream cannot be reclaimed - "
                  "restart hyprwhspr to stop the CPU/log churn", flush=True)
            return
        print("[RECOVERY] Wedged audio stream cannot be reclaimed - exiting for systemd restart", flush=True)
        self._notify_user("hyprwhspr", "Audio system wedged - restarting service", "critical")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    def _on_pulse_server_restarted(self):
        """Called when PulseAudio/PipeWire server restarts"""
        try:
            print("[PULSE] Audio server restarted - recovering audio capture", flush=True)

            # Give audio server time to fully initialize
            time.sleep(1)

            if self.audio_capture.recover_audio_capture('pulse_server_restart'):
                print("[PULSE] Recovery successful after server restart", flush=True)
                self._write_recovery_result(True, 'pulse_restart')
            else:
                print("[PULSE] Recovery failed after server restart", flush=True)
                self._write_recovery_result(False, 'pulse_restart')
        except Exception as e:
            print(f"[PULSE] Error handling server restart: {e}", flush=True)

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
            # Push-to-talk mode: only start recording on key press
            if not self.is_recording:
                self._start_recording(language_override=language_override)
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
            # Push-to-talk mode: stop recording on key release
            if self.is_recording:
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

    # Continuous mode: auto-paste on speech pause
    _POLL_INTERVAL = 0.1  # seconds between silence checks

    def _get_float_setting(self, key, default=0.0):
        """Coerce a config value to float; falls back to `default` if missing, invalid, or non-finite (NaN/Infinity)."""
        value = self.config.get_setting(key, default)
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return value if math.isfinite(value) else default

    def _calibrate_noise_floor(self, stop_event):
        """Sample the mic's noise floor over ~0.6s. Returns None if stopped/recording ended early."""
        samples = []
        for _ in range(6):
            stop_event.wait(0.1)
            if stop_event.is_set() or not self.is_recording:
                return None
            samples.append(self.audio_capture.rolling_avg_level)
        noise_floor = min(samples)
        return max(noise_floor * 2, 2e-4)

    def _continuous_start_silence_monitor(self):
        """Start monitoring for silence to trigger auto-paste in continuous mode"""
        self._continuous_cancelled = False
        self._continuous_stop_silence_monitor()
        self._continuous_silence_stop.clear()

        silence_seconds = self._get_float_setting('continuous_silence_seconds', 2.0)
        configured_threshold = self._get_float_setting('continuous_silence_threshold', 0)
        samples_needed = max(1, int(silence_seconds / self._POLL_INTERVAL))

        def monitor():
            silent_count = 0
            try:
                # Auto-calibrate threshold from noise floor if not manually configured
                threshold = configured_threshold
                if threshold <= 0:
                    threshold = self._calibrate_noise_floor(self._continuous_silence_stop)
                    if threshold is None:
                        return
                    print(f"[CONTINUOUS] Auto-calibrated threshold={threshold:.5f}", flush=True)

                while self.is_recording and not self._continuous_silence_stop.is_set():
                    raw_level = self.audio_capture.rolling_avg_level
                    if raw_level < threshold:
                        silent_count += 1
                        if silent_count >= samples_needed:
                            self._continuous_flush_audio()
                            silent_count = 0
                    else:
                        silent_count = 0
                    self._continuous_silence_stop.wait(self._POLL_INTERVAL)
            except Exception as e:
                print(f"[CONTINUOUS] Silence monitor error: {e}", flush=True)

        self._continuous_silence_thread = threading.Thread(target=monitor, daemon=True)
        self._continuous_silence_thread.start()

    def _continuous_stop_silence_monitor(self):
        """Stop the continuous silence monitor"""
        self._continuous_silence_stop.set()
        if self._continuous_silence_thread and self._continuous_silence_thread.is_alive():
            self._continuous_silence_thread.join(timeout=0.5)
        self._continuous_silence_thread = None

    def _autostop_start_silence_monitor(self):
        """Auto-stop recording after `silence_timeout` seconds of silence (toggle/auto modes).

        Arms only after speech has been detected, so it can't fire while the user is
        still composing their first sentence. The silence threshold auto-calibrates from
        the noise floor (same approach as continuous mode). No-op when silence_timeout <= 0.
        """
        silence_timeout = self._get_float_setting('silence_timeout', 0)
        configured_threshold = self._get_float_setting('continuous_silence_threshold', 0)

        with self._autostop_lock:
            self._autostop_teardown_locked()
            if silence_timeout <= 0:
                return  # feature disabled (default)

            stop_event = threading.Event()
            samples_needed = max(1, int(silence_timeout / self._POLL_INTERVAL))

            def monitor():
                thread = threading.current_thread()
                try:
                    threshold = configured_threshold
                    if threshold <= 0:
                        threshold = self._calibrate_noise_floor(stop_event)
                        if threshold is None:
                            return

                    armed = False          # don't count silence until speech has been heard
                    silent_count = 0
                    while self.is_recording and not stop_event.is_set():
                        level = self.audio_capture.rolling_avg_level
                        if level >= threshold:
                            armed = True
                            silent_count = 0
                        elif armed:
                            silent_count += 1
                            if silent_count >= samples_needed:
                                stop_event.set()
                                # Deregister ourselves before handing off to _stop_recording().
                                # If a newer generation already replaced us here, our session
                                # already ended some other way - don't stop whatever is
                                # recording now, it isn't ours.
                                with self._autostop_lock:
                                    still_current = self._autostop_silence_thread is thread
                                    if still_current:
                                        self._autostop_silence_thread = None
                                        self._autostop_silence_stop = None
                                if still_current:
                                    print(f"[AUTOSTOP] {silence_timeout:.1f}s of silence - stopping recording", flush=True)
                                    self._stop_recording()   # plays the stop beep; transcribes + pastes
                                return
                        stop_event.wait(self._POLL_INTERVAL)
                except Exception as e:
                    print(f"[AUTOSTOP] Silence monitor error: {e}", flush=True)

            self._autostop_silence_thread = threading.Thread(target=monitor, daemon=True)
            self._autostop_silence_stop = stop_event
            self._autostop_silence_thread.start()

    def _autostop_stop_silence_monitor(self):
        """Stop the auto-stop silence monitor (safe to call from anywhere, incl. the monitor)."""
        with self._autostop_lock:
            self._autostop_teardown_locked()

    def _autostop_teardown_locked(self):
        """Tear down whatever autostop monitor is currently registered. Caller must hold `_autostop_lock`.

        No generation check here (unlike the monitor's self-stop path) - accepted edge case.
        """
        stop_event = self._autostop_silence_stop
        thread = self._autostop_silence_thread
        if stop_event is not None:
            stop_event.set()
        # Avoid self-join deadlock when the monitor thread is the one triggering the stop
        if thread and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=0.5)
        self._autostop_silence_thread = None
        self._autostop_silence_stop = None

    def _continuous_stop_and_wait(self):
        """Stop the silence monitor and wait for any in-progress transcription"""
        self._continuous_stop_silence_monitor()
        self._continuous_transcription_done.wait(timeout=30)

    def _continuous_flush_audio(self):
        """Flush accumulated audio: transcribe and paste without stopping recording"""
        if not self._continuous_flush_lock.acquire(blocking=False):
            return  # another flush/transcription in progress

        # Lock is now held — all paths must go through the finally that releases it.
        self._continuous_transcription_done.clear()
        should_transcribe = False
        audio_data = None
        try:
            audio_data = self.audio_capture.flush_buffer()
            if audio_data is None or len(audio_data) == 0:
                return

            duration = len(audio_data) / self.audio_capture.sample_rate
            if duration < 0.5 or self._is_zero_volume(audio_data):
                return

            print(f"[CONTINUOUS] Flushing {duration:.1f}s of audio for transcription", flush=True)
            should_transcribe = True
        except Exception as e:
            print(f"[CONTINUOUS] Flush error: {e}", flush=True)
        finally:
            if not should_transcribe:
                self._continuous_flush_lock.release()
                self._continuous_transcription_done.set()

        if not should_transcribe:
            return

        # Transcribe in background thread; lock is held until transcription
        # completes so the next flush is blocked until this one finishes.
        def process():
            try:
                transcription = self.whisper_manager.transcribe_audio(
                    audio_data,
                    sample_rate=self.audio_capture.sample_rate,
                    language_override=self._current_language_override,
                )
                if transcription and transcription.strip():
                    text = transcription.strip()
                    lower = text.lower().strip('.!? ')
                    if lower in _HALLUCINATION_MARKERS or text.startswith('♪'):
                        print(f"[CONTINUOUS] Hallucination ignored: {text!r}", flush=True)
                        return
                    if self._continuous_cancelled:
                        print("[CONTINUOUS] Cancelled — discarding transcription", flush=True)
                        return
                    self._inject_text(text)
                    print(f"[CONTINUOUS] Pasted: {text[:80]}{'...' if len(text) > 80 else ''}", flush=True)
                else:
                    print("[CONTINUOUS] No transcription from flushed audio", flush=True)
            except Exception as e:
                print(f"[CONTINUOUS] Transcription error: {e}", flush=True)
            finally:
                self._notify_capture_subscriber("", final=True)
                self._continuous_flush_lock.release()
                self._continuous_transcription_done.set()

        threading.Thread(target=process, daemon=True).start()

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

    def _set_longform_processing(self, processing):
        self.is_processing = processing

    def _start_recording(self, language_override=None):
        """Start voice recording
        
        Args:
            language_override: Optional language code to use for this recording session
                              (overrides the default language from config)
        """
        # Gate checks happen under the same lock that sets the flag, so a
        # concurrent toggle never sees is_recording=True for a start that
        # gets blocked (and stops a recording that never began)
        with self._recording_lock:
            if self.is_recording:
                return
            if self._model_initializing:
                blocked = 'initializing'
            elif self._backend_init_failed:
                blocked = 'init-failed'
            elif getattr(self.whisper_manager, '_model_manually_unloaded', False):
                blocked = 'unloaded'
            elif self.whisper_manager.realtime_client_missing():
                blocked = 'realtime-reconnect'
            else:
                blocked = None
                # Set flag immediately to prevent duplicate starts
                self.is_recording = True
                # Store language override for this recording session
                self._current_language_override = language_override

        # Model is still loading in background
        if blocked == 'initializing':
            self._notify_user("hyprwhspr", "Model still loading, please wait…", urgency="normal")
            print("[CONTROL] Recording blocked: model is still initializing", flush=True)
            return

        # Backend init failed (e.g. deps missing at boot): retry it instead
        # of recording audio that can never be transcribed
        if blocked == 'init-failed':
            self._notify_user("hyprwhspr", "Backend failed to load — retrying, try again shortly", urgency="normal")
            print("[CONTROL] Recording blocked: backend init failed - retrying", flush=True)
            self._start_backend_init_background()
            return

        # Realtime client was torn down; rebuild in background instead of blocking
        if blocked == 'realtime-reconnect':
            self._notify_user("hyprwhspr", "Reconnecting — try again in a moment", urgency="normal")
            print("[CONTROL] Recording blocked: realtime client not connected - reconnecting", flush=True)
            self._start_backend_init_background()
            return

        # Model was deliberately unloaded to free GPU resources
        if blocked == 'unloaded':
            self._notify_user(
                "hyprwhspr",
                "Model unloaded — run: hyprwhspr model reload",
                urgency="normal",
            )
            print("[CONTROL] Recording blocked: model is unloaded. Run: hyprwhspr model reload", flush=True)
            return

        try:
            self._clear_mic_osd_preview_text()

            # Clear zero-volume signal file when starting a new recording
            # This allows waybar to recover immediately on successful start
            self._clear_zero_volume_signal()
            
            # Write recording status to file for tray script
            self._write_recording_status(True)

            # Update language in realtime client if override is provided
            if language_override is not None:
                self.whisper_manager.update_realtime_language(language_override)
            
            # Check if using realtime-ws backend and get streaming callback
            streaming_callback = self.whisper_manager.get_realtime_streaming_callback()
            backend = normalize_backend(self.config.get_setting('transcription_backend', 'pywhispercpp'))
            if backend == 'realtime-ws' and streaming_callback is None:
                # Fail fast: realtime-ws requires an active streaming callback (and a connected client)
                with self._recording_lock:
                    self.is_recording = False
                self._write_recording_status(False)
                self._hide_mic_osd()
                self._stop_audio_level_monitoring()
                self._notify_zero_volume(
                    "Realtime backend not connected yet — try again in a moment.",
                    log_level="ERROR",
                )
                # Restore audio if it was ducked
                if self.audio_ducker.is_ducked:
                    self.audio_ducker.restore()
                return
            
            # Helper function to verify stream is working and play sound
            def verify_and_play_sound():
                """Wait for callbacks and play sound if stream works"""
                import time
                start_time = time.monotonic()
                while time.monotonic() - start_time < 1.5:  # Wait up to 1.5s
                    # Read frames_since_start with lock held to avoid data race
                    with self.audio_capture.lock:
                        frames_count = self.audio_capture.frames_since_start
                    if frames_count > 0:
                        # At least one callback received - stream is working
                        self.audio_manager.play_start_sound()
                        return True
                    time.sleep(0.05)
                # No callbacks received - stream likely broken (will be handled by caller)
                return False
            
            # Helper function to verify stream continues working after initial check
            def verify_stream_stable():
                """Verify stream continues receiving callbacks after initial verification"""
                import time
                initial_frames = 0
                with self.audio_capture.lock:
                    initial_frames = self.audio_capture.frames_since_start
                
                # Wait a bit more to ensure stream is stable
                time.sleep(0.2)
                
                with self.audio_capture.lock:
                    current_frames = self.audio_capture.frames_since_start
                    # Stream should have received more callbacks if it's stable
                    return current_frames > initial_frames
            
            # Start audio capture (with streaming callback for realtime-ws)
            try:
                if not self.audio_capture.start_recording(streaming_callback=streaming_callback):
                    raise RuntimeError("start_recording() returned False")
                
                # Verify stream is working before playing sound
                if not verify_and_play_sound():
                    # Stream broken - stop recording (thread will clean up stream)
                    self.audio_capture.stop_recording()

                    # Reset state
                    with self._recording_lock:
                        self.is_recording = False
                    self._write_recording_status(False)
                    
                    # Hide mic-osd visualization
                    self._hide_mic_osd()

                    message = self._mic_failure_message(
                        "Microphone not responding - please unplug and replug USB microphone, then try recording again")
                    self._notify_zero_volume(message, log_level="ERROR")

                    # Restore audio if it was ducked
                    if self.audio_ducker.is_ducked:
                        self.audio_ducker.restore()
                    return  # Don't attempt recovery during user-initiated recording

                # Stream is verified working - show mic-osd visualization
                print("Recording started", flush=True)
                self._show_mic_osd()
                
                # Additional stability check - verify stream continues working
                if not verify_stream_stable():
                    # Stream stopped working shortly after starting
                    self.audio_capture.stop_recording()
                    with self._recording_lock:
                        self.is_recording = False
                    self._write_recording_status(False)
                    
                    # Hide mic-osd visualization
                    self._hide_mic_osd()

                    fallback = "Microphone stream unstable - please wait a moment and try recording again"
                    message = self._mic_failure_message(fallback)
                    self._notify_zero_volume(message, log_level="WARN" if message == fallback else "ERROR")

                    # Restore audio if it was ducked
                    if self.audio_ducker.is_ducked:
                        self.audio_ducker.restore()
                    return
                
                # Recording is confirmed working - abort any in-progress recovery and clear background retries
                try:
                    self.audio_capture.abort_recovery()
                except Exception:
                    pass
                if self._background_recovery_needed.is_set():
                    print("[HEALTH] Recording succeeded - canceling background recovery", flush=True)
                    self._background_recovery_needed.clear()
                
                # Duck system audio now that stream is confirmed working
                if self.config.get_setting('audio_ducking', False):
                    self.audio_ducker.duck()

                # Stream is working and stable - start monitoring
                self._start_audio_level_monitoring()
                    
            except (RuntimeError, Exception) as e:
                print(f"[ERROR] Failed to start recording: {e}", flush=True)

                # Clean up resources
                self._hide_mic_osd()
                self._stop_audio_level_monitoring()

                self.whisper_manager.close_realtime_connection("recording start failure")

                # Stop recording (will clean up if thread started)
                try:
                    self.audio_capture.stop_recording()
                except Exception:
                    pass  # Ignore if already stopped

                # Reset state - fail fast, don't attempt recovery
                with self._recording_lock:
                    self.is_recording = False
                self._write_recording_status(False)
                self._notify_zero_volume(
                    self._mic_failure_message(
                        "Microphone disconnected or not responding - please unplug and replug USB microphone, then try recording again"),
                    log_level="ERROR")

                # Restore audio if it was ducked
                if self.audio_ducker.is_ducked:
                    self.audio_ducker.restore()
                return

        except Exception as e:
            print(f"[ERROR] Failed to start recording: {e}", flush=True)

            # Clean up resources
            self._hide_mic_osd()
            self._stop_audio_level_monitoring()

            self.whisper_manager.close_realtime_connection("recording start failure")

            with self._recording_lock:
                self.is_recording = False
            self._write_recording_status(False)

            # Restore audio if it was ducked
            if self.audio_ducker.is_ducked:
                self.audio_ducker.restore()

    def _cleanup_recording_state(self):
        """Best-effort cleanup after any recording ends. Safe to call multiple times."""
        self._notify_capture_subscriber("", final=True)
        self._autostop_stop_silence_monitor()

        try:
            self._clear_mic_osd_preview_text()
        except Exception:
            pass
        try:
            self._hide_mic_osd()
        except Exception:
            pass
        try:
            self._stop_audio_level_monitoring()
        except Exception:
            pass
        try:
            self._write_recording_status(False)
        except Exception:
            pass
        try:
            if self.audio_ducker.is_ducked:
                self.audio_ducker.restore()
        except Exception:
            pass

    def _cancel_recording_muted(self):
        """Cancel recording early due to muted microphone"""
        with self._recording_lock:
            if not self.is_recording:
                return
            self.is_recording = False
            self._current_language_override = None  # Clear language override on error

        print("[MUTE] Recording cancelled - microphone returned silence for 1 second", flush=True)

        self._cleanup_recording_state()
        try:
            self.audio_capture.stop_recording()
            self.audio_manager.play_error_sound()
            # Note: No desktop notification - tray will detect muted state via audio level monitoring
        except Exception as e:
            print(f"[ERROR] Error canceling recording: {e}", flush=True)

    def _cancel_recording(self):
        """Cancel recording and discard audio without transcribing or injecting text"""
        with self._recording_lock:
            if not self.is_recording:
                return
            self.is_recording = False
            self._current_language_override = None

        print("Recording cancelled (discarded)", flush=True)

        self._cleanup_recording_state()
        try:
            # Stop capture and discard the audio data
            self.audio_capture.stop_recording()

            # Discard buffered realtime audio but keep the WebSocket alive so the
            # next recording starts without paying a fresh handshake
            self.whisper_manager.discard_realtime_audio()

            self.audio_manager.play_error_sound()
        except Exception as e:
            print(f"[ERROR] Error cancelling recording: {e}", flush=True)

    def _stop_recording(self):
        """Stop voice recording and process audio"""
        with self._recording_lock:
            if not self.is_recording:
                return
            self.is_recording = False

        print("Recording stopped", flush=True)

        # Tear down the auto-stop silence monitor if it was running (toggle/auto modes)
        self._autostop_stop_silence_monitor()

        try:
            self._clear_mic_osd_preview_text()

            # Set visualizer to processing state (keep it visible during transcription)
            self._set_visualizer_state('processing')
            
            # Stop audio level monitoring
            self._stop_audio_level_monitoring()
            
            # Write recording status to file for tray script
            self._write_recording_status(False)

            # Restore system audio if it was ducked
            if self.audio_ducker.is_ducked:
                self.audio_ducker.restore()

            # Check backend type
            backend = self.config.get_setting('transcription_backend', 'pywhispercpp')
            backend = normalize_backend(backend)
            
            # Stop audio capture
            audio_data = self.audio_capture.stop_recording()

            # Check for zero-volume or broken stream
            if audio_data is None:
                # Stream was broken - check if we got any callbacks
                self.audio_manager.play_error_sound()
                with self.audio_capture.lock:
                    frames_count = self.audio_capture.frames_since_start
                if frames_count == 0:
                    # No callbacks received - mic disconnected during recording
                    self._notify_zero_volume("Microphone disconnected during recording - no audio captured. Try recording again after reseating.")
                else:
                    # Had callbacks but no data - stream broke mid-recording
                    self._notify_zero_volume("Audio stream broke during recording - no audio data captured. Try recording again after reseating.")
                # Show error state and hide OSD
                self._show_result_and_hide(False)
                self._notify_capture_subscriber("", final=True)
            elif self._is_zero_volume(audio_data):
                # Audio data exists but is all zeros - mic not producing sound
                # Play error sound and notify user (may be intentional muting, but still inform)
                self.audio_manager.play_error_sound()
                self._notify_zero_volume("Microphone not producing audio (zero volume detected). This may be intentional muting, or the microphone may need to be reseated.")
                # Show error state and hide OSD
                self._show_result_and_hide(False)
                self._notify_capture_subscriber("", final=True)
            else:
                # Valid audio data - process it
                self.audio_manager.play_stop_sound()
                self._process_audio(audio_data)
                
            # Clear language override after transcription completes
            self._current_language_override = None
                
        except Exception as e:
            print(f"[ERROR] Error stopping recording: {e}", flush=True)
            self._notify_capture_subscriber("", final=True)
            # Ensure cleanup even if error occurs
            try:
                self.is_recording = False
                self._current_language_override = None  # Clear language override on cancel
                self._show_result_and_hide(False)
                self._stop_audio_level_monitoring()
                self._write_recording_status(False)
                self._continuous_stop_silence_monitor()

                self.whisper_manager.close_realtime_connection("recording stop error")
            except Exception:
                pass  # Best effort cleanup

    def _process_audio(self, audio_data):
        """Process captured audio through Whisper"""
        if self.is_processing:
            return

        success = False
        try:
            self.is_processing = True

            # Transcribe audio with language override if set
            transcription = self.whisper_manager.transcribe_audio(
                audio_data,
                sample_rate=self.audio_capture.sample_rate,
                language_override=self._current_language_override,
            )

            if transcription and transcription.strip():
                text = transcription.strip()

                # Filter out Whisper hallucination markers - don't touch clipboard
                normalized = text.lower().replace('_', ' ').strip('[]().!?, ')
                if normalized in _HALLUCINATION_MARKERS or text.startswith('♪'):
                    print(f"[INFO] Whisper hallucination detected: {text!r} - ignoring")
                    self.audio_manager.play_error_sound()
                    success = False
                    # Explicitly handle cleanup before returning to ensure visualizer state is updated
                    self.is_processing = False
                    self._show_result_and_hide(False)
                    return

                self.current_transcription = text

                # Inject text
                self._inject_text(self.current_transcription)
                success = True
            else:
                print("[WARN] No transcription generated")
                self.audio_manager.play_error_sound()

        except Exception as e:
            print(f"[ERROR] Error processing audio: {e}", flush=True)
        finally:
            self._notify_capture_subscriber("", final=True)
            self._clear_mic_osd_preview_text()
            self.is_processing = False
            # Show success/error state and hide OSD after delay
            self._show_result_and_hide(success)

    def _inject_text(self, text):
        """Inject transcribed text into active application"""

        # Capture mode: route text to client instead of injecting into active app
        if self._capture_subscriber is not None:
            self._notify_capture_subscriber(text, final=True)
            return True

        try:
            if not self.text_injector.inject_text(text):
                print(f"[ERROR] Text injection failed ({len(text)} chars)", flush=True)
                return False

            print(f"[INJECT] Text injected ({len(text)} chars)", flush=True)

            # Text injection succeeded - system is fully healthy
            # Cancel any pending background recovery
            if self._background_recovery_needed.is_set():
                print("[HEALTH] Successful recording detected - canceling background recovery", flush=True)
                self._background_recovery_needed.clear()
                # Write recovery success result (system self-healed via user activity)
                self._write_recovery_result(True, 'user_activity_validated')
                with self._mic_state_lock:
                    self._mic_disconnected = False
                self._clear_error_state_signals()
            try:
                # Ensure any active recovery is aborted once user activity proves health
                self.audio_capture.abort_recovery()
            except Exception:
                pass
            return True
        except Exception as e:
            print(f"[ERROR] Text injection failed: {e}", flush=True)
            return False

    def _is_zero_volume(self, audio_data) -> bool:
        """Check if audio data has zero or near-zero volume"""
        if np is None:
            # numpy not available, can't check - assume not zero
            return False
        
        if audio_data is None or len(audio_data) == 0:
            return True
        
        try:
            # Check if all samples are zero
            if np.all(audio_data == 0.0):
                return True
            
            # Check RMS level (very quiet = likely broken)
            rms = np.sqrt(np.mean(audio_data**2))
            if rms < 1e-6:  # Extremely quiet threshold
                return True
        except Exception:
            # If check fails, assume not zero (safer)
            return False
        
        return False

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

    def _mic_failure_message(self, fallback: str) -> str:
        """Pick user advice for a failed recording start based on what actually failed.

        Distinguishes "input device missing/initializing" (stream never opened)
        from "device wedged" (stream opened but delivers no callbacks) using the
        open outcome recorded by the capture thread; fallback covers the latter.
        """
        with self._mic_state_lock:
            if self._mic_disconnected:
                return "Microphone disconnected - please replug USB microphone"
        if not getattr(self.audio_capture, 'stream_opened', True):
            open_error = getattr(self.audio_capture, 'stream_open_error', None)
            if open_error:
                print(f"[ERROR] Stream open failed: {open_error}", flush=True)
            return "Microphone unavailable - input device missing or still initializing - check the connection and try again"
        return fallback

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
        print(f"[{log_level}] {message}", flush=True)

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
            print(f"[WARN] Failed to write recording status: {e}")

    def _migrate_legacy_state_files(self):
        """One-time cleanup of signal files that lived in CONFIG_DIR before they
        moved to RUNTIME_DIR, plus compat symlinks for the three files external
        consumers (Hyprland binds, GNOME extension) may still use at old paths.
        """
        try:
            RUNTIME_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
            RUNTIME_DIR.chmod(0o700)
        except Exception as e:
            print(f"[WARN] Failed to prepare runtime dir {RUNTIME_DIR}: {e}")

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
            print(f"[WARN] Failed to create config dir: {e}")
        for name, target in compat.items():
            try:
                (CONFIG_DIR / name).symlink_to(target)
            except Exception as e:
                print(f"[WARN] Failed to create legacy compat symlink {name}: {e}")

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

                print(f"[RECOVERY] Result written: {status} ({reason})", flush=True)

                # If recovery succeeded, clear any error state signals
                if success:
                    self._clear_error_state_signals()

            except Exception as e:
                print(f"[WARN] Failed to write recovery result: {e}")

    def _clear_error_state_signals(self):
        """Clear error state signal files after successful recovery"""
        try:
            # Clear mic zero volume signal
            if MIC_ZERO_VOLUME_FILE.exists():
                MIC_ZERO_VOLUME_FILE.unlink()
                print("[RECOVERY] Cleared mic_zero_volume error signal", flush=True)

            # Clear any stale recovery request file
            if RECOVERY_REQUESTED_FILE.exists():
                RECOVERY_REQUESTED_FILE.unlink()

        except Exception as e:
            print(f"[WARN] Failed to clear error signals: {e}", flush=True)

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
                            print(f"[WARN] Audio level monitoring error: {e}", flush=True)
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

    def _setup_recording_control_fifo(self):
        """Create named pipe (FIFO) for immediate recording control"""
        try:
            # Ensure config directory exists
            RECORDING_CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)

            # Retry once: a concurrent writer can create a dirent between exists() and mkfifo().
            for _ in range(2):
                # Check if existing file is a FIFO (if regular file, remove it)
                if RECORDING_CONTROL_FILE.exists():
                    if RECORDING_CONTROL_FILE.is_fifo():
                        # Already a FIFO, we're good
                        print("[INIT] Recording control FIFO already exists", flush=True)
                        return
                    # Old regular file - remove it
                    try:
                        RECORDING_CONTROL_FILE.unlink()
                        print("[INIT] Removed old recording_control file (replacing with FIFO)", flush=True)
                    except Exception as e:
                        print(f"[WARN] Failed to remove old recording_control file: {e}", flush=True)
                        return

                try:
                    os.mkfifo(str(RECORDING_CONTROL_FILE))
                    print(f"[INIT] Created recording control FIFO: {RECORDING_CONTROL_FILE}", flush=True)
                    return
                except FileExistsError:
                    continue

            print("[WARN] Failed to create recording control FIFO: file kept reappearing", flush=True)
            print("[WARN] Recording control will fall back to file polling (1 second delay)", flush=True)

        except OSError as e:
            # Handle permission errors, read-only filesystem, etc.
            print(f"[WARN] Failed to create recording control FIFO: {e}", flush=True)
            print("[WARN] Recording control will fall back to file polling (1 second delay)", flush=True)
        except Exception as e:
            print(f"[WARN] Unexpected error creating recording control FIFO: {e}", flush=True)

    def _recording_control_listener(self):
        """Listen on FIFO for recording control commands (blocking, immediate)"""
        while not self._recording_control_stop.is_set():
            try:
                # Check if FIFO exists, recreate if needed
                if not RECORDING_CONTROL_FILE.exists() or not RECORDING_CONTROL_FILE.is_fifo():
                    if self._recording_control_stop.is_set():
                        break
                    # Recreate FIFO
                    try:
                        if RECORDING_CONTROL_FILE.exists():
                            RECORDING_CONTROL_FILE.unlink()
                        os.mkfifo(str(RECORDING_CONTROL_FILE))
                        print("[CONTROL] Recreated recording control FIFO", flush=True)
                    except Exception as e:
                        print(f"[CONTROL] Failed to recreate FIFO: {e}", flush=True)
                        # Wait a bit before retrying
                        time.sleep(1)
                        continue
                
                # Open FIFO for reading (blocks until writer appears)
                with open(RECORDING_CONTROL_FILE, 'r') as f:
                    raw_data = f.read()

                # Handle multiple commands written to FIFO before read
                # (e.g., user clicks rapidly during timeout - "start\nstart")
                # Take only the last valid command (most recent intent)
                # Commands can be: 'start', 'start:lang', 'stop', 'cancel', 'submit',
                #                  'model_unload', 'model_reload'
                valid_base_commands = {'start', 'stop', 'cancel', 'submit', 'model_unload', 'model_reload'}
                lines = [line.strip() for line in raw_data.splitlines() if line.strip()]

                # Parse commands - extract base command and optional language
                parsed_commands = []
                for line in lines:
                    line_lower = line.lower()
                    if ':' in line_lower and line_lower.startswith('start:'):
                        # start:lang format - preserve language case
                        parts = line.split(':', 1)
                        lang = parts[1].strip() if len(parts) > 1 else None
                        parsed_commands.append(('start', lang))
                    elif line_lower in valid_base_commands:
                        parsed_commands.append((line_lower, None))

                if not parsed_commands:
                    if lines:
                        print(f"[CONTROL] No valid commands in: {lines}", flush=True)
                    continue

                action, language_param = parsed_commands[-1]  # Take the last valid command
                
                # Check recording mode to route to appropriate handler
                recording_mode = self.config.get_setting("recording_mode", "toggle")
                
                # Process action immediately
                if action == "start":
                    lang_info = f" (language: {language_param})" if language_param else ""
                    if recording_mode == "long_form":
                        self._longform.request_start(language_override=language_param)
                    elif not self.is_recording:
                        print(f"[CONTROL] Recording start requested (immediate){lang_info}", flush=True)
                        self._start_recording(language_override=language_param)
                        if recording_mode == "continuous":
                            self._continuous_start_silence_monitor()
                        elif recording_mode in ("toggle", "auto"):
                            self._autostop_start_silence_monitor()
                    else:
                        print("[CONTROL] Recording already in progress, ignoring start request", flush=True)
                elif action == "stop":
                    if recording_mode == "long_form":
                        self._longform.request_pause()
                    elif self.is_recording:
                        print("[CONTROL] Recording stop requested (immediate)", flush=True)
                        if recording_mode == "continuous":
                            self._continuous_stop_and_wait()
                        self._stop_recording()
                    else:
                        print("[CONTROL] Not currently recording, ignoring stop request", flush=True)
                elif action == "cancel":
                    if recording_mode == "long_form":
                        self._longform.request_cancel()
                    elif self.is_recording:
                        print("[CONTROL] Recording cancel requested (immediate)", flush=True)
                        if recording_mode == "continuous":
                            self._continuous_cancelled = True
                            self._continuous_stop_silence_monitor()
                        self._cancel_recording()
                    else:
                        print("[CONTROL] Not currently recording, ignoring cancel request", flush=True)
                elif action == "submit":
                    # Submit command for long-form mode submit shortcut
                    if recording_mode == "long_form":
                        print("[CONTROL] Long-form submit requested (immediate)", flush=True)
                        self._longform.submit_shortcut()
                    else:
                        print("[CONTROL] Submit command only valid in long_form mode", flush=True)
                elif action == "model_unload":
                    if self.is_recording:
                        print("[CONTROL] Cannot unload model while recording", flush=True)
                        self._notify_user("hyprwhspr", "Stop recording before unloading model", urgency="normal")
                    else:
                        print("[CONTROL] Model unload requested", flush=True)
                        if self.whisper_manager.unload_model():
                            try:
                                MODEL_UNLOADED_FILE.touch()
                            except Exception:
                                pass
                            self._notify_user("hyprwhspr", "Model unloaded — GPU resources freed", urgency="low")
                        else:
                            self._notify_user("hyprwhspr", "Unload not applicable for this backend", urgency="normal")
                elif action == "model_reload":
                    print("[CONTROL] Model reload requested", flush=True)
                    if self.whisper_manager.reload_model():
                        try:
                            MODEL_UNLOADED_FILE.unlink(missing_ok=True)
                        except Exception:
                            pass
                        self._notify_user("hyprwhspr", "Model reloaded — ready to record", urgency="low")
                    else:
                        self._notify_user("hyprwhspr", "Model reload failed — check logs", urgency="critical")
                else:
                    print(f"[CONTROL] Unknown recording control action: {action}", flush=True)
                    
            except FileNotFoundError:
                # FIFO was deleted - will be recreated on next iteration
                if not self._recording_control_stop.is_set():
                    print("[CONTROL] FIFO deleted, will recreate on next iteration", flush=True)
                    time.sleep(0.1)  # Brief pause before retrying
            except OSError as e:
                # Permission errors, broken pipe, etc.
                if not self._recording_control_stop.is_set():
                    print(f"[CONTROL] FIFO error: {e}, retrying...", flush=True)
                    time.sleep(0.1)  # Brief pause before retrying
            except Exception as e:
                if not self._recording_control_stop.is_set():
                    print(f"[CONTROL] Error in FIFO listener: {e}", flush=True)
                    time.sleep(0.1)  # Brief pause before retrying

    def _notify_capture_subscriber(self, text, final):
        """
        Forward a transcription chunk to the active capture subscriber if any.

        Handler thread owns the connection fd and is the only closer.
        This method only writes; on final=True it signals the handler to close.
        """
        with self._capture_subscriber_lock:
            subscriber = self._capture_subscriber
            if subscriber is None:
                if final:
                    self._capture_subscriber_done.set()
                return

            if text:
                try:
                    subscriber.sendall(text.encode('utf-8'))
                except (BrokenPipeError, ConnectionError, OSError) as e:
                    # handler select loop will detect disconnect and clean up.
                    print(f"[CAPTURE] Subscriber write failed: {e}", flush=True)

            if final:
                self._capture_subscriber_done.set()

    def _setup_capture_socket(self):
        """
        Create unix-domain socket for `record capture` client requests.

        Returns (sock: socket.socket or None)
        """
        try:
            SOCKET_FILE.parent.mkdir(parents=True, exist_ok=True)

            # we do a cleanup at the startup to ensure no stale socket remains
            if SOCKET_FILE.exists():
                try:
                    SOCKET_FILE.unlink()
                    print("[INIT] Removed stale capture socket", flush=True)
                except OSError as e:
                    print(f"[WARN] Failed to remove stale capture socket: {e}", flush=True)
                    return None

            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(str(SOCKET_FILE))
            os.chmod(str(SOCKET_FILE), 0o600)
            sock.listen(4)
            sock.settimeout(1.0)
            print(f"[INIT] Created capture socket: {SOCKET_FILE}", flush=True)

            return sock
        except OSError as e:
            print(f"[WARN] Failed to create capture socket: {e}", flush=True)
            print("[WARN] `record capture` will be unavailable this session", flush=True)

            return None

    def _capture_socket_listener(self):
        """
        Connection handler for the capture socket.
        """
        sock = self._setup_capture_socket()
        if sock is None:
            return

        try:
            while not self._capture_socket_stop.is_set():
                try:
                    conn, _ = sock.accept()
                except socket.timeout:
                    # proceed to check stop flag
                    continue
                except OSError as e:
                    if self._capture_socket_stop.is_set():
                        break
                    print(f"[CAPTURE] Accept error: {e}", flush=True)
                    time.sleep(0.1)
                    continue

                threading.Thread(
                    target=self._handle_capture_connection,
                    args=(conn,),
                    daemon=True,
                    name="CaptureConnectionHandler",
                ).start()
        finally:
            try:
                sock.close()
            except OSError:
                pass
            try:
                SOCKET_FILE.unlink(missing_ok=True)
            except OSError:
                pass

    def _handle_capture_connection(self, conn):
        """
        Handle one capture client that is claiming the subscriber slot to receive transcription text.
        """

        try:
            # fetch the request line with a timeout
            conn.settimeout(1.0)
            line = conn.recv(256).split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()

            verb, _, language = line.partition(":")
            language = language.strip() or None

            if verb != "capture":
                print(f"[CAPTURE] Unknown request: {line!r}", flush=True)
                return

            conn.settimeout(None)

            # claim the subscriber slot if available
            with self._capture_subscriber_lock:
                if self._capture_subscriber is not None:
                    print("[CAPTURE] Rejecting — slot occupied", flush=True)
                    try:
                        conn.sendall(b"ERROR:slot_occupied\n")
                    except OSError:
                        pass
                    return
                self._capture_subscriber = conn
                self._capture_subscriber_done.clear()

            # trigger recording via fifo
            if not self.is_recording:
                cmd = f"start:{language}\n" if language else "start\n"
                try:
                    fd = os.open(str(RECORDING_CONTROL_FILE), os.O_WRONLY | os.O_NONBLOCK)
                    try:
                        os.write(fd, cmd.encode())
                    finally:
                        os.close(fd)
                except OSError as e:
                    print(f"[CAPTURE] Failed to self-trigger via FIFO: {e}", flush=True)

            # either wait for recording to finish or for client disconnect
            client_disconnected = False
            while True:
                if self._capture_subscriber_done.is_set():
                    break
                try:
                    readable, _, _ = select.select([conn], [], [], 0.5)
                except (OSError, ValueError):
                    client_disconnected = True
                    break
                if readable:
                    try:
                        peek = conn.recv(1, socket.MSG_PEEK)
                    except OSError:
                        peek = b""
                    if not peek:
                        client_disconnected = True
                        break

            # we should cleanup if there was a premature disconnect
            if client_disconnected and self.is_recording:
                print("[CAPTURE] Client disconnected, cancelling recording", flush=True)
                try:
                    fd = os.open(str(RECORDING_CONTROL_FILE), os.O_WRONLY | os.O_NONBLOCK)
                    try:
                        os.write(fd, b"cancel\n")
                    finally:
                        os.close(fd)
                except OSError:
                    pass

        except Exception as e:
            print(f"[CAPTURE] Handler error: {e}", flush=True)
        finally:
            # clear lock since we are the owner
            with self._capture_subscriber_lock:
                if self._capture_subscriber is conn:
                    self._capture_subscriber = None
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass

    def _start_backend_init_background(self):
        """Initialize the transcription backend in a background thread.

        Used for slow backends at startup and to retry after a failed init.
        Guarded so concurrent callers can't spawn duplicate init threads.
        """
        with self._backend_init_lock:
            if self._model_initializing:
                return
            self._model_initializing = True
            self._backend_init_failed = False

        def _bg_init():
            ok = self.whisper_manager.initialize()
            self._backend_init_failed = not ok
            self._model_initializing = False
            if ok:
                print("[READY] Model ready — recording now available", flush=True)
                self._notify_user("hyprwhspr", "Ready", urgency="low")
            else:
                print("[ERROR] Failed to initialize backend in background", flush=True)

        threading.Thread(target=_bg_init, daemon=True, name="BackendInit").start()

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
            print(f"[RECOVERY] Warning: Could not clear recovery request file: {e}", flush=True)
        
        # Determine reason for recovery
        was_recording = self.is_recording
        reason = "mic_unavailable" if not was_recording else "mic_no_audio"
        
        print(f"[RECOVERY] Recovery requested by tray script ({reason} detected)", flush=True)
        
        # Mark that we're attempting recovery for this error state
        self.recovery_attempted.set()
        
        # Attempt recovery (will handle stopping current recording if needed)
        if self.audio_capture.recover_audio_capture(f"tray_script_request_{reason}"):
            print("[RECOVERY] Audio recovery successful - mic should now be available", flush=True)

            # After successful audio recovery, also reinitialize model if needed
            # This handles suspend/resume cases where CUDA context is invalid
            model_reinit_success = self.whisper_manager.reinitialize_after_resume(only_if_idle=True)
            if not model_reinit_success:
                print("[RECOVERY] Model reinitialization failed after audio recovery", flush=True)

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
                print("[RECOVERY] Restarting recording after successful recovery", flush=True)
                # Get streaming callback if needed
                streaming_callback = self.whisper_manager.get_realtime_streaming_callback()
                try:
                    if not self.audio_capture.start_recording(streaming_callback=streaming_callback):
                        print("[RECOVERY] Failed to restart recording after recovery - start_recording() returned False", flush=True)
                        self.is_recording = False
                        self._write_recording_status(False)
                        return
                    self._start_audio_level_monitoring()
                except Exception as e:
                    print(f"[RECOVERY] Failed to restart recording after recovery: {e}", flush=True)
                    self.is_recording = False
                    self._write_recording_status(False)
        else:
            print("[RECOVERY] Recovery failed - please reseat your USB microphone", flush=True)

            # Write recovery failure result for tray script
            self._write_recovery_result(False, reason)

            # Keep flag set - recovery was attempted and failed, don't retry

    def _on_system_suspend(self):
        """Called when system is about to suspend (D-Bus PrepareForSleep signal)"""
        try:
            print("[SUSPEND] System entering suspend", flush=True)

            # Close WebSocket connections preemptively (avoid timeout errors)
            self.whisper_manager.close_realtime_connection("system suspend")
        except Exception as e:
            print(f"[SUSPEND] Error handling suspend: {e}", flush=True)

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
                    print(f"[SUSPEND] Keyboard resync failed: {e}", flush=True)

        _resync_all("")

        def _delayed():
            time.sleep(2.0)
            _resync_all("_delayed")
        threading.Thread(target=_delayed, daemon=True).start()

    def _on_system_resume(self):
        """Called when system resumes from suspend (D-Bus PrepareForSleep signal)"""
        try:
            print("[SUSPEND] System resumed - recovering audio and backends...", flush=True)
            time.sleep(2)  # Give audio/GPU drivers time to reinitialize

            # Re-attach the shortcut keyboard if suspend dropped it without a udev
            # 'add' event on resume. A delayed second pass catches a late reappear.
            self._resync_shortcut_keyboards("post_suspend_resume")

            if self.audio_capture.recover_audio_capture('post_suspend_resume'):
                # Reinitialize backend state (model / WebSocket) per backend type
                backend_reinit_success = self.whisper_manager.reinitialize_after_resume()
                if not backend_reinit_success:
                    print("[SUSPEND] Recovery failed - backend reinitialization failed", flush=True)

                # Write recovery result and clear background recovery flag only after ALL recovery steps complete
                if backend_reinit_success:
                    print("[SUSPEND] Recovery successful - microphone ready", flush=True)
                    self._write_recovery_result(True, 'suspend_resume')
                    with self._mic_state_lock:
                        self._mic_disconnected = False
                    self._background_recovery_needed.clear()
                else:
                    # Backend reinitialization failed - signal that recovery is still needed
                    if backend in pywhispercpp_variants or backend == 'faster-whisper':
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
                print("[SUSPEND] Recovery failed - will retry in background (6 attempts over 30s)", flush=True)
                self._background_recovery_needed.set()

                # Start background recovery thread
                if self._background_recovery_thread is None or not self._background_recovery_thread.is_alive():
                    self._background_recovery_thread = threading.Thread(
                        target=self._background_recovery_retry,
                        daemon=True
                    )
                    self._background_recovery_thread.start()
        except Exception as e:
            print(f"[SUSPEND] Error handling resume: {e}", flush=True)

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
                    print("[RECOVERY] Background recovery successful - microphone ready", flush=True)
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
        print("[RECOVERY] Background recovery exhausted - microphone may need manual reseat", flush=True)
        self._write_recovery_result(False, 'background_retry_exhausted')
        self._background_recovery_needed.clear()

    def run(self):
        """Start the application"""
        # Restore user's preferred default source (persisted by mic-select picker)
        saved_source_file = Path.home() / '.config' / 'hyprwhspr' / '.default_source'
        if saved_source_file.exists():
            try:
                source_name = saved_source_file.read_text().strip()
                if source_name:
                    result = subprocess.run(
                        ['pactl', 'set-default-source', source_name],
                        timeout=5, check=False,
                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                    )
                    if result.returncode == 0:
                        print(f"[INIT] Restored default source: {source_name}", flush=True)
                    else:
                        err = result.stderr.decode(errors='replace').strip()
                        print(f"[WARN] Could not restore default source '{source_name}': {err}", flush=True)
            except Exception as e:
                print(f"[WARN] Could not restore default source: {e}", flush=True)

        # Check audio capture availability
        if not self.audio_capture.is_available():
            print("[ERROR] Audio capture not available!")
            return False

        # Start global shortcuts (unless using Hyprland compositor bindings)
        use_hypr_bindings = self.config.get_setting("use_hypr_bindings", False)
        if self.global_shortcuts:
            if not self.global_shortcuts.start():
                print("[ERROR] Failed to start global shortcuts!")
                print("[ERROR] Check permissions: you may need to be in 'input' group")
                return False
        elif not use_hypr_bindings:
            print("[ERROR] Global shortcuts not initialized!")
            return False

        # Start FIFO listener thread for immediate recording control
        if RECORDING_CONTROL_FILE.exists() and RECORDING_CONTROL_FILE.is_fifo():
            self._recording_control_thread = threading.Thread(
                target=self._recording_control_listener,
                daemon=True,
                name="RecordingControlListener"
            )
            self._recording_control_thread.start()
            print("[INIT] Started recording control FIFO listener", flush=True)
        else:
            print("[WARN] Recording control FIFO not available, using fallback polling", flush=True)

        # start capture socket listener thread
        self._capture_socket_thread = threading.Thread(
            target=self._capture_socket_listener,
            daemon=True,
            name="CaptureSocketListener",
        )
        self._capture_socket_thread.start()
        print("[INIT] Started capture socket listener", flush=True)

        # Initialize whisper backend. Slow backends (e.g. cohere-transcribe loading a
        # 4 GB model onto the GPU) run in a background thread so shortcuts and the FIFO
        # listener are active immediately. Recording is blocked until ready.
        backend = self.config.get_setting('transcription_backend', 'pywhispercpp')
        slow_backends = {'cohere-transcribe'}
        if backend in slow_backends:
            print(f"\n[INIT] Loading model in background (shortcuts active, recording will unblock when ready)...", flush=True)
            self._start_backend_init_background()
        else:
            if not self.whisper_manager.initialize():
                # Stay alive with recording blocked; the record gate retries init
                print("[ERROR] Failed to initialize backend - will retry on next record attempt", flush=True)
                self._backend_init_failed = True

        if use_hypr_bindings:
            print("\n[READY] hyprwhspr ready - using Hyprland compositor bindings", flush=True)
        else:
            print("\n[READY] hyprwhspr ready - press shortcut to start dictation", flush=True)

        # Give microphone 1 second to fully initialize before checking for recovery
        # This prevents spurious errors on startup if device is still settling
        time.sleep(1)

        try:
            # Keep the application running
            while True:
                # Recording control now handled by FIFO listener thread (immediate)
                # Check for recovery requests from tray script (non-blocking)
                self._attempt_recovery_if_needed()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Shutting down hyprwhspr...")
            self._cleanup()
        except Exception as e:
            print(f"[ERROR] Error in main loop: {e}", flush=True)
            self._cleanup()
            return False
        
        return True

    def _cleanup(self):
        """Clean up resources when shutting down"""
        def cleanup_step(name, action):
            try:
                action()
            except Exception as exc:
                print(f"[WARN] Cleanup step {name!r} failed: {exc}", flush=True)

        def stop_recording_listener():
            if not hasattr(self, '_recording_control_stop'):
                return
            self._recording_control_stop.set()
            thread = getattr(self, '_recording_control_thread', None)
            if thread and thread.is_alive():
                print("[SHUTDOWN] Stopping recording control FIFO listener...", flush=True)
                try:
                    fd = os.open(str(RECORDING_CONTROL_FILE), os.O_WRONLY | os.O_NONBLOCK)
                    os.close(fd)
                except (OSError, FileNotFoundError):
                    pass
                thread.join(timeout=1.0)
                if thread.is_alive():
                    print("[WARN] Recording control thread did not stop cleanly", flush=True)

        def stop_thread(stop_name, thread_name, label, timeout):
            stop = getattr(self, stop_name, None)
            if stop is None:
                return
            stop.set()
            thread = getattr(self, thread_name, None)
            if thread and thread.is_alive():
                print(f"[SHUTDOWN] Stopping {label}...", flush=True)
                thread.join(timeout=timeout)
                if thread.is_alive():
                    print(f"[WARN] {label} did not stop cleanly", flush=True)

        def call_optional(attribute, method):
            target = getattr(self, attribute, None)
            if target is not None:
                getattr(target, method)()

        try:
            # Stop recording control FIFO listener thread
            cleanup_step("stop recording-control listener", stop_recording_listener)

            # Stop capture socket listener thread
            cleanup_step("stop capture-socket listener", lambda: stop_thread(
                '_capture_socket_stop', '_capture_socket_thread',
                'capture socket listener', 2.0))

            # Stop background recovery thread
            cleanup_step("stop background recovery", lambda: stop_thread(
                '_background_recovery_stop', '_background_recovery_thread',
                'background recovery thread', 2.0))

            # Hide mic-osd overlay if visible
            cleanup_step("hide mic-osd", self._hide_mic_osd)
            
            # Stop mic-osd daemon
            cleanup_step("stop mic-osd", lambda: call_optional('_mic_osd_runner', 'stop'))
            
            # Stop device monitor
            cleanup_step("stop device monitor", lambda: call_optional('device_monitor', 'stop'))

            # Stop pulse monitor
            cleanup_step("stop pulse monitor", lambda: call_optional('pulse_monitor', 'stop'))

            # Stop suspend monitor
            cleanup_step("stop suspend monitor", lambda: call_optional('suspend_monitor', 'stop'))

            # Stop global shortcuts
            cleanup_step("stop global shortcuts", lambda: call_optional('global_shortcuts', 'stop'))
            
            # Stop secondary shortcuts
            cleanup_step("stop secondary shortcuts", lambda: call_optional('secondary_shortcuts', 'stop'))

            # Stop cancel shortcut
            cleanup_step("stop cancel shortcut", lambda: call_optional('_cancel_shortcuts', 'stop'))

            # Prevent a long-form autosave callback from racing shutdown.
            cleanup_step("stop long-form timer", self._longform.stop_auto_save_timer)

            # Stop audio capture
            if getattr(self, 'is_recording', False):
                cleanup_step("stop audio capture", lambda: call_optional('audio_capture', 'stop_recording'))

            # Cleanup whisper manager (closes WebSocket connections, etc.)
            cleanup_step("close transcription backend", lambda: call_optional('whisper_manager', 'cleanup'))

            # Tear down our private ydotoold daemon (only started if the uinput
            # paste fallback was used; no-op otherwise).
            cleanup_step("close text injector", lambda: call_optional('text_injector', 'close'))

            # Save configuration
            cleanup_step("save configuration", lambda: call_optional('config', 'save_config'))

            # Clear runtime state files so external consumers (tray, CLI)
            # don't see stale values after shutdown
            cleanup_step("clear runtime state", self._reset_stale_state)

            print("[CLEANUP] Cleanup completed", flush=True)

        except Exception as e:
            print(f"[WARN] Error during cleanup: {e}", flush=True)
        finally:
            # Release lock file
            _release_lock_file()

            # Clean up mic-osd PID file (safety cleanup in case runner.stop() wasn't called)
            from src.paths import MIC_OSD_PID_FILE
            if MIC_OSD_PID_FILE.exists():
                try:
                    MIC_OSD_PID_FILE.unlink()
                except Exception:
                    pass


def _acquire_lock_file():
    """
    Acquire a lock file to prevent multiple instances from running.
    Returns (success: bool, message: str or None)
    """
    global _lock_file, _lock_file_path
    
    # Check if we're running under systemd
    # If we are, systemd already manages single instances - skip the lock file
    running_under_systemd = False
    try:
        ppid = os.getppid()
        try:
            with open(f'/proc/{ppid}/comm', 'r', encoding='utf-8') as f:
                parent_comm = f.read().strip()
                if 'systemd' in parent_comm:
                    running_under_systemd = True
        except (FileNotFoundError, IOError):
            pass
        
        if os.environ.get('INVOCATION_ID') or os.environ.get('JOURNAL_STREAM'):
            running_under_systemd = True
    except Exception:
        pass
    
    if running_under_systemd:
        # Trust systemd to manage single instances
        return True, None
    
    # Set up lock file path
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    _lock_file_path = LOCK_FILE
    
    try:
        # Try to open/create the lock file
        _lock_file = open(_lock_file_path, 'w')
        
        # Try to acquire an exclusive non-blocking lock
        try:
            fcntl.flock(_lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            # Lock acquired successfully - write our PID
            _lock_file.write(str(os.getpid()))
            _lock_file.flush()
            
            # Register cleanup handler
            atexit.register(_release_lock_file)
            
            return True, None
            
        except (IOError, OSError):
            # Lock is held by another process
            _lock_file.close()
            _lock_file = None
            
            # Check if the PID in the lock file is still valid
            try:
                with open(_lock_file_path, 'r') as f:
                    lock_pid_str = f.read().strip()
                    if lock_pid_str:
                        try:
                            lock_pid = int(lock_pid_str)
                            # Check if process is still running
                            os.kill(lock_pid, 0)
                            # Process exists - another instance is running
                            return False, f"lock file (PID: {lock_pid})"
                        except (ValueError, ProcessLookupError, PermissionError):
                            # Stale lock file - remove it and try again
                            try:
                                _lock_file_path.unlink()
                                # Retry acquiring lock
                                _lock_file = open(_lock_file_path, 'w')
                                fcntl.flock(_lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                                _lock_file.write(str(os.getpid()))
                                _lock_file.flush()
                                atexit.register(_release_lock_file)
                                return True, None
                            except (IOError, OSError):
                                # Still can't acquire - another process got it
                                if _lock_file:
                                    _lock_file.close()
                                    _lock_file = None
                                return False, "lock file (another instance starting)"
            except (FileNotFoundError, IOError):
                # Can't read lock file - assume another instance is running
                return False, "lock file"
                
    except (IOError, OSError, PermissionError) as e:
        # Can't create or access lock file
        if _lock_file:
            _lock_file.close()
            _lock_file = None
        return False, f"lock file (error: {e})"


def _release_lock_file():
    """Release the lock file"""
    global _lock_file, _lock_file_path
    
    if _lock_file:
        try:
            fcntl.flock(_lock_file.fileno(), fcntl.LOCK_UN)
            _lock_file.close()
        except Exception:
            pass
        _lock_file = None
    
    if _lock_file_path and _lock_file_path.exists():
        try:
            _lock_file_path.unlink()
        except Exception:
            pass


def _is_hyprwhspr_running():
    """Check if hyprwhspr is already running"""
    try:
        from instance_detection import is_hyprwhspr_running
        return is_hyprwhspr_running()
    except ImportError:
        # Fallback if import fails (shouldn't happen in normal operation)
        return False, None


def main():
    """Main entry point"""
    # First, try to acquire lock file (primary detection method)
    lock_acquired, lock_message = _acquire_lock_file()
    if not lock_acquired:
        print("[ERROR] hyprwhspr is already running!")
        if lock_message:
            print(f"[ERROR] Detected via: {lock_message}")
        print("\n[INFO] To check the status of the running instance:")
        print("  • Run: hyprwhspr status")
        print("\n[INFO] To stop the running instance:")
        print("  • If running via systemd: systemctl --user stop hyprwhspr")
        print("  • If running manually: kill the process or press Ctrl+C in its terminal")
        print("\n[INFO] For more information, run: hyprwhspr --help")
        sys.exit(1)
    
    # Fallback: also check via process detection
    is_running, how = _is_hyprwhspr_running()
    if is_running:
        # Release lock since we detected another instance
        _release_lock_file()
        print("[ERROR] hyprwhspr is already running!")
        print(f"[ERROR] Detected via: {how}")
        print("\n[INFO] To check the status of the running instance:")
        print("  • Run: hyprwhspr status")
        print("\n[INFO] To stop the running instance:")
        print("  • If running via systemd: systemctl --user stop hyprwhspr")
        print("  • If running manually: kill the process or press Ctrl+C in its terminal")
        print("\n[INFO] For more information, run: hyprwhspr --help")
        sys.exit(1)
    
    try:
        app = hyprwhsprApp()
        app.run()
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Stopping hyprwhspr...")
        if 'app' in locals():
            app._cleanup()
        _release_lock_file()
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        _release_lock_file()
        sys.exit(1)


if __name__ == "__main__":
    # Safety check: if a CLI subcommand was passed, redirect to CLI instead of starting the service
    # This handles cases where an old bin/hyprwhspr wrapper doesn't recognize newer CLI subcommands
    # Keep in sync with the subcommand route in bin/hyprwhspr
    CLI_SUBCOMMANDS = ['setup', 'install', 'config', 'waybar', 'noctalia', 'systemd', 'status',
                       'model', 'validate', 'uninstall', 'backend', 'state', 'mic-osd',
                       'keyboard', 'record', 'test']
    if len(sys.argv) > 1 and sys.argv[1] in CLI_SUBCOMMANDS:
        print(f"[REDIRECT] Detected CLI subcommand '{sys.argv[1]}', redirecting to CLI...")
        # Execute CLI with same arguments
        cli_path = Path(__file__).parent / 'cli.py'
        os.execv(sys.executable, [sys.executable, str(cli_path)] + sys.argv[1:])

    main()
    
