#!/usr/bin/env python3
"""
hyprwhspr - stt
"""

import sys
import time
import threading
import os
import fcntl
import atexit
import subprocess
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None  # Will be checked when needed

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

# Add the src directory to the Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

# Lock file for preventing multiple instances
_lock_file = None
_lock_file_path = None

from config_manager import ConfigManager
from audio_capture import AudioCapture
from whisper_manager import WhisperManager
from text_injector import TextInjector
from global_shortcuts import GlobalShortcuts
from audio_manager import AudioManager
from device_monitor import DeviceMonitor, PYUDEV_AVAILABLE
from paths import (
    RECORDING_STATUS_FILE, RECORDING_CONTROL_FILE, AUDIO_LEVEL_FILE, RECOVERY_REQUESTED_FILE,
    RECOVERY_RESULT_FILE, MIC_ZERO_VOLUME_FILE, LOCK_FILE
)
from backend_utils import normalize_backend

class hyprwhsprApp:
    """Main application class for hyprwhspr voice dictation (Headless Mode)"""

    def __init__(self):
        # Initialize core components
        self.config = ConfigManager()

        # Initialize audio capture with configured device
        audio_device_id = self.config.get_setting('audio_device_id', None)
        self.audio_capture = AudioCapture(device_id=audio_device_id, config_manager=self.config)

        # Initialize audio feedback manager
        self.audio_manager = AudioManager(self.config)

        # Initialize whisper manager with shared config
        self.whisper_manager = WhisperManager(config_manager=self.config)
        self.text_injector = TextInjector(self.config)
        self.global_shortcuts = None

        # Application state
        self.is_recording = False
        self.is_processing = False
        self.current_transcription = ""
        self.audio_level_thread = None
        self.recovery_attempted = threading.Event()  # Thread-safe flag: track if recovery was attempted for current error state
        self.last_recovery_time = 0.0  # Track when recovery last completed (for cooldown)
        self._last_mic_error_log_time = 0.0  # Track when we last logged mic error (prevent duplicates)
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

        # Background recovery retry state (for suspend/resume)
        self._background_recovery_needed = threading.Event()  # Signal that recovery should be retried
        self._background_recovery_thread = None  # Background thread handle
        self._background_recovery_stop = threading.Event()  # Signal to stop background recovery

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

        # Track startup time BEFORE any monitors are initialized
        # This prevents race condition where hotplug events arrive before _startup_time is set
        self._startup_time = time.monotonic()
        self._startup_grace_period = 5.0  # Ignore hotplug events for 5 seconds after startup

        # Set up device hotplug monitoring (for automatic mic recovery)
        self._setup_device_monitor()

        # Set up PulseAudio/PipeWire event monitoring
        self._setup_pulse_monitor()

        # Set up suspend/resume monitoring
        self._setup_suspend_monitor()

        # Pre-initialize mic-osd daemon (eliminates latency on recording)
        self._mic_osd_runner = None
        if self.config.get_setting('mic_osd_enabled', True):
            try:
                from mic_osd import MicOSDRunner
                runner = MicOSDRunner()
                if runner.is_available():
                    if runner._ensure_daemon():  # Start daemon now
                        self._mic_osd_runner = runner
            except Exception:
                pass

        # Set up global shortcuts (needed for headless operation)
        self._setup_global_shortcuts()

    def _setup_global_shortcuts(self):
        """Initialize global keyboard shortcuts"""
        try:
            shortcut_key = self.config.get_setting("primary_shortcut", "Super+Alt+D")
            recording_mode = self.config.get_setting("recording_mode", "toggle")
            grab_keys = self.config.get_setting("grab_keys", True)
            selected_device_path = self.config.get_setting("selected_device_path", None)

            # Register callbacks based on recording mode
            # Validate recording_mode and only register release callback for modes that need it
            if recording_mode == 'toggle':
                # Toggle mode: only register press callback
                self.global_shortcuts = GlobalShortcuts(
                    shortcut_key,
                    self._on_shortcut_triggered,
                    None,  # No release callback for toggle mode
                    device_path=selected_device_path,
                    grab_keys=grab_keys,
                )
            elif recording_mode in ('push_to_talk', 'auto'):
                # Push-to-talk and auto modes: register both press and release callbacks
                self.global_shortcuts = GlobalShortcuts(
                    shortcut_key,
                    self._on_shortcut_triggered,
                    self._on_shortcut_released,
                    device_path=selected_device_path,
                    grab_keys=grab_keys,
                )
            else:
                # Invalid mode: default to toggle behavior (no release callback)
                print(f"[WARNING] Invalid recording_mode '{recording_mode}', defaulting to 'toggle'")
                self.global_shortcuts = GlobalShortcuts(
                    shortcut_key,
                    self._on_shortcut_triggered,
                    None,  # No release callback for invalid modes (treated as toggle)
                    device_path=selected_device_path,
                    grab_keys=grab_keys,
                )
        except Exception as e:
            print(f"[ERROR] Failed to initialize global shortcuts: {e}")
            self.global_shortcuts = None

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
                # Debounce recovery attempts: USB reseat generates multiple events
                with self._hotplug_lock:
                    current_time = time.monotonic()
                    if current_time - self._last_hotplug_add_time < 2.0:
                        return  # Skip duplicate
                    self._last_hotplug_add_time = current_time

                # Check if background recovery is already running - cancel it
                if self._background_recovery_needed.is_set():
                    self._background_recovery_needed.clear()
                    # Reset cooldown timer to allow hotplug recovery to proceed immediately
                    # Without this, if background recovery just attempted, hotplug would hit cooldown
                    # Must hold recovery_lock when writing _last_recovery_attempt_time (same lock used in recover_audio_capture)
                    with self.audio_capture.recovery_lock:
                        self.audio_capture._last_recovery_attempt_time = 0.0
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

            # Check if we're using system default (no specific device configured)
            if self.config.get_setting('audio_device_id', None) is None:
                print("[PULSE] Re-enumerating devices (no specific device configured, using system default)")
                # Re-initialize audio capture to pick up new default
                # Hold recovery_lock to prevent concurrent modifications with recovery
                # Check again right before calling to avoid TOCTOU race condition
                with self.audio_capture.recovery_lock:
                    if self.audio_capture.recovery_in_progress:
                        print(f"[PULSE] Default source changed during recovery - skipping (recovery will handle)", flush=True)
                        return
                    # Call _initialize_sounddevice() while holding the lock to prevent
                    # concurrent modifications to sd.default.* global state
                    self.audio_capture._initialize_sounddevice()
            else:
                print("[PULSE] Specific device configured, ignoring system default change")
        except Exception as e:
            print(f"[PULSE] Error handling default source change: {e}", flush=True)

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
        recording_mode = self.config.get_setting("recording_mode", "toggle")
        
        if recording_mode == 'toggle':
            # Toggle mode: start/stop recording
            if self.is_recording:
                self._stop_recording()
            else:
                self._start_recording()
        elif recording_mode == 'push_to_talk':
            # Push-to-talk mode: only start recording on key press
            if not self.is_recording:
                self._start_recording()
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
            if should_start:
                self._start_recording()
        else:
            # Invalid mode, default to toggle behavior
            if self.is_recording:
                self._stop_recording()
            else:
                self._start_recording()

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
                # Otherwise, keep recording (tap started it, let it continue)

    def _start_recording(self):
        """Start voice recording"""
        # Use a lock to prevent concurrent starts (race condition protection)
        with self._recording_lock:
            if self.is_recording:
                return
            
            # Set flag immediately to prevent duplicate starts
            self.is_recording = True
        
        print("Recording started", flush=True)
        
        try:
            # Clear zero-volume signal file when starting a new recording
            # This allows waybar to recover immediately on successful start
            self._clear_zero_volume_signal()
            
            # Write recording status to file for tray script
            self._write_recording_status(True)
            
            # Check if using realtime-ws backend and get streaming callback
            streaming_callback = self.whisper_manager.get_realtime_streaming_callback()
            
            # Helper function to verify stream is working and play sound
            def verify_and_play_sound():
                """Wait for callbacks and play sound if stream works"""
                import time
                start_time = time.monotonic()
                while time.monotonic() - start_time < 0.5:  # Wait up to 500ms
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
                    self.is_recording = False
                    self._write_recording_status(False)
                    
                    # Hide mic-osd visualization
                    self._hide_mic_osd()

                    # Check if we know the microphone was disconnected
                    with self._mic_state_lock:
                        mic_was_disconnected = self._mic_disconnected

                    if mic_was_disconnected:
                        self._notify_zero_volume("Microphone disconnected - please replug USB microphone", log_level="ERROR")
                    else:
                        self._notify_zero_volume("Microphone not responding - please unplug and replug USB microphone, then try recording again", log_level="ERROR")
                    return  # Don't attempt recovery during user-initiated recording
                
                # Stream is verified working - show mic-osd visualization
                self._show_mic_osd()
                
                # Additional stability check - verify stream continues working
                if not verify_stream_stable():
                    # Stream stopped working shortly after starting
                    self.audio_capture.stop_recording()
                    self.is_recording = False
                    self._write_recording_status(False)
                    
                    # Hide mic-osd visualization
                    self._hide_mic_osd()

                    # Check if we know the microphone was disconnected
                    with self._mic_state_lock:
                        mic_was_disconnected = self._mic_disconnected

                    if mic_was_disconnected:
                        self._notify_zero_volume("Microphone disconnected - please replug USB microphone", log_level="ERROR")
                    else:
                        self._notify_zero_volume("Microphone stream unstable - please wait a moment and try recording again", log_level="WARN")
                    return
                
                # Recording is confirmed working - abort any in-progress recovery and clear background retries
                try:
                    self.audio_capture.abort_recovery()
                except Exception:
                    pass
                if self._background_recovery_needed.is_set():
                    print("[HEALTH] Recording succeeded - canceling background recovery", flush=True)
                    self._background_recovery_needed.clear()
                
                # Stream is working and stable - start monitoring
                self._start_audio_level_monitoring()
                    
            except (RuntimeError, Exception) as e:
                print(f"[ERROR] Failed to start recording: {e}", flush=True)

                # Clean up resources
                self._hide_mic_osd()
                self._stop_audio_level_monitoring()

                # Close WebSocket if using realtime-ws backend
                backend = normalize_backend(self.config.get_setting('transcription_backend', 'pywhispercpp'))
                if backend == 'realtime-ws' and self.whisper_manager._realtime_client:
                    print("[CLEANUP] Closing WebSocket after recording start failure", flush=True)
                    self.whisper_manager._cleanup_realtime_client()

                # Stop recording (will clean up if thread started)
                try:
                    self.audio_capture.stop_recording()
                except Exception:
                    pass  # Ignore if already stopped

                # Reset state - fail fast, don't attempt recovery
                self.is_recording = False
                self._write_recording_status(False)
                self._notify_zero_volume("Microphone disconnected or not responding - please unplug and replug USB microphone, then try recording again", log_level="ERROR")
                return
            
        except Exception as e:
            print(f"[ERROR] Failed to start recording: {e}", flush=True)

            # Clean up resources
            self._hide_mic_osd()
            self._stop_audio_level_monitoring()

            # Close WebSocket if using realtime-ws backend
            backend = normalize_backend(self.config.get_setting('transcription_backend', 'pywhispercpp'))
            if backend == 'realtime-ws' and self.whisper_manager._realtime_client:
                print("[CLEANUP] Closing WebSocket after recording start failure", flush=True)
                self.whisper_manager._cleanup_realtime_client()

            self.is_recording = False
            self._write_recording_status(False)

    def _cancel_recording_muted(self):
        """Cancel recording early due to muted microphone"""
        if not self.is_recording:
            return

        try:
            self.is_recording = False
            self._hide_mic_osd()
            self._stop_audio_level_monitoring()
            self._write_recording_status(False)
            self.audio_capture.stop_recording()
            self.audio_manager.play_error_sound()
            # Note: No desktop notification - tray will detect muted state via audio level monitoring
        except Exception as e:
            print(f"[ERROR] Error canceling recording: {e}")
            # Ensure cleanup even if error occurs
            try:
                self._hide_mic_osd()
                self._stop_audio_level_monitoring()
                self._write_recording_status(False)
            except Exception:
                pass  # Best effort cleanup

    def _stop_recording(self):
        """Stop voice recording and process audio"""
        if not self.is_recording:
            return

        print("Recording stopped", flush=True)
        
        try:
            self.is_recording = False
            
            # Hide mic-osd visualization
            self._hide_mic_osd()
            
            # Stop audio level monitoring
            self._stop_audio_level_monitoring()
            
            # Write recording status to file for tray script
            self._write_recording_status(False)
            
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
            elif self._is_zero_volume(audio_data):
                # Audio data exists but is all zeros - mic not producing sound
                # Play error sound and notify user (may be intentional muting, but still inform)
                self.audio_manager.play_error_sound()
                self._notify_zero_volume("Microphone not producing audio (zero volume detected). This may be intentional muting, or the microphone may need to be reseated.")
            else:
                # Valid audio data - process it
                self.audio_manager.play_stop_sound()
                self._process_audio(audio_data)
                
        except Exception as e:
            print(f"[ERROR] Error stopping recording: {e}", flush=True)
            # Ensure cleanup even if error occurs
            try:
                self.is_recording = False
                self._hide_mic_osd()
                self._stop_audio_level_monitoring()
                self._write_recording_status(False)

                # Close WebSocket if using realtime-ws backend
                backend = normalize_backend(self.config.get_setting('transcription_backend', 'pywhispercpp'))
                if backend == 'realtime-ws' and self.whisper_manager._realtime_client:
                    print("[CLEANUP] Closing WebSocket after recording stop error", flush=True)
                    self.whisper_manager._cleanup_realtime_client()
            except Exception:
                pass  # Best effort cleanup

    def _process_audio(self, audio_data):
        """Process captured audio through Whisper"""
        if self.is_processing:
            return

        try:
            self.is_processing = True

            # Transcribe audio
            transcription = self.whisper_manager.transcribe_audio(audio_data)

            if transcription and transcription.strip():
                text = transcription.strip()

                # Filter out Whisper hallucination markers - don't touch clipboard
                normalized = text.lower().replace('_', ' ').strip('[]() ')
                hallucination_markers = ('blank audio', 'blank', 'video playback', 'music', 'music playing', 'keyboard clicking')
                if normalized in hallucination_markers:
                    print(f"[INFO] Whisper hallucination detected: {text!r} - ignoring")
                    self.audio_manager.play_error_sound()
                    return

                self.current_transcription = text

                # Inject text
                self._inject_text(self.current_transcription)
            else:
                print("[WARN] No transcription generated")
                self.audio_manager.play_error_sound()
                
        except Exception as e:
            print(f"[ERROR] Error processing audio: {e}")
        finally:
            self.is_processing = False

    def _inject_text(self, text):
        """Inject transcribed text into active application"""
        try:
            self.text_injector.inject_text(text)

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
        except Exception as e:
            print(f"[ERROR] Text injection failed: {e}")

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
        """Send desktop notification if notify-send is available"""
        try:
            subprocess.run(
                ["notify-send", "-u", urgency, title, message],
                timeout=2,
                check=False,
                capture_output=True
            )
        except Exception:
            pass  # Silently fail if notify-send not available

    def _notify_zero_volume(self, message: str, log_level: str = "WARN"):
        """Log zero-volume recording and signal waybar (no desktop notifications)"""
        # Prevent duplicate error logs within 2 seconds (user might hit record twice)
        # Use lock to ensure thread-safe read-modify-write on _last_mic_error_log_time
        if "Microphone disconnected or not responding" in message:
            with self._error_log_lock:
                current_time = time.monotonic()
                if current_time - self._last_mic_error_log_time < 2.0:
                    # Already logged this error recently, skip duplicate log
                    return
                self._last_mic_error_log_time = current_time

        # Print to logs (primary notification)
        print(f"[{log_level}] {message}", flush=True)

        # Note: No desktop notification - tray monitors state files and handles all user notifications

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

    def _show_mic_osd(self):
        """Show mic-osd visualization overlay"""
        if self._mic_osd_runner and self._mic_osd_runner.is_available():
            self._mic_osd_runner.show()

    def _hide_mic_osd(self):
        """Hide mic-osd visualization overlay"""
        runner = getattr(self, '_mic_osd_runner', None)
        if runner:
            try:
                runner.hide()
            except Exception:
                pass

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
        if self.audio_level_thread and self.audio_level_thread.is_alive():
            return

        def monitor_audio_level():
            AUDIO_LEVEL_FILE.parent.mkdir(parents=True, exist_ok=True)

            # Muted mic detection: 5e-7 threshold catches true digital silence but not quiet rooms
            zero_samples = 0
            zero_threshold = 5e-7
            samples_to_cancel = 10  # 1 second at 100ms intervals

            try:
                while self.is_recording:
                    try:
                        # Get scaled level for visualization (0.0-1.0)
                        level = self.audio_capture.get_audio_level()
                        with open(AUDIO_LEVEL_FILE, 'w') as f:
                            f.write(f'{level:.3f}')

                        # Mute detection (only if enabled)
                        if self.config.get_setting('mute_detection', True):
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
                        # Silently fail - don't spam errors
                        pass
                    time.sleep(0.1)  # Update 10 times per second
            finally:
                # Clean up file when not recording (always runs, even on early return)
                try:
                    if AUDIO_LEVEL_FILE.exists():
                        AUDIO_LEVEL_FILE.unlink()
                except:
                    pass

        self.audio_level_thread = threading.Thread(target=monitor_audio_level, daemon=True)
        self.audio_level_thread.start()

    def _stop_audio_level_monitoring(self):
        """Stop audio level monitoring"""
        if self.audio_level_thread and self.audio_level_thread.is_alive():
            # Thread will exit when is_recording becomes False
            pass

    def _check_recording_control(self):
        """Check for recording control requests from tray script and start/stop recording"""
        # Check if recording control file exists
        if not RECORDING_CONTROL_FILE.exists():
            return
        
        try:
            # Read the control file to determine action
            with open(RECORDING_CONTROL_FILE, 'r') as f:
                action = f.read().strip().lower()
            
            # Clear the file immediately to prevent duplicate processing
            RECORDING_CONTROL_FILE.unlink()
            
            if action == "start":
                # Start recording if not already recording
                if not self.is_recording:
                    print("[CONTROL] Recording start requested by tray script", flush=True)
                    self._start_recording()
                else:
                    print("[CONTROL] Recording already in progress, ignoring start request", flush=True)
            elif action == "stop":
                # Stop recording if currently recording
                if self.is_recording:
                    print("[CONTROL] Recording stop requested by tray script", flush=True)
                    self._stop_recording()
                else:
                    print("[CONTROL] Not currently recording, ignoring stop request", flush=True)
            else:
                print(f"[CONTROL] Unknown recording control action: {action}", flush=True)
                
        except Exception as e:
            print(f"[CONTROL] Error processing recording control: {e}", flush=True)
            # Try to clear the file even if there was an error
            try:
                if RECORDING_CONTROL_FILE.exists():
                    RECORDING_CONTROL_FILE.unlink()
            except Exception:
                pass

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
            print("[RECOVERY] Recovery successful - mic should now be available", flush=True)

            # Write recovery success result for tray script
            self._write_recovery_result(True, reason)

            # Clear disconnected flag - microphone is back
            with self._mic_state_lock:
                self._mic_disconnected = False

            # Clear background recovery flag (manual recovery succeeded, no need to retry)
            self._background_recovery_needed.clear()

            # After successful audio recovery, also reinitialize model if needed
            # This handles suspend/resume cases where CUDA context is invalid
            backend = self.config.get_setting('transcription_backend', 'pywhispercpp')
            backend = normalize_backend(backend)
            
            if backend == 'pywhispercpp' and hasattr(self.whisper_manager, '_pywhisper_model') and self.whisper_manager._pywhisper_model:
                # Check if model needs reinitialization (long idle = suspend/resume)
                current_time = time.monotonic()
                if hasattr(self.whisper_manager, '_last_use_time'):
                    time_since_last = current_time - self.whisper_manager._last_use_time
                    if time_since_last > 300 and self.whisper_manager._last_use_time > 0:
                        print("[RECOVERY] Reinitializing model after audio recovery (suspend/resume detected)", flush=True)
                        self.whisper_manager._reinitialize_model()
            
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
            backend = normalize_backend(self.config.get_setting('transcription_backend', 'pywhispercpp'))
            if backend == 'realtime-ws' and self.whisper_manager._realtime_client:
                print("[SUSPEND] Closing WebSocket before suspend", flush=True)
                self.whisper_manager._cleanup_realtime_client()
        except Exception as e:
            print(f"[SUSPEND] Error handling suspend: {e}", flush=True)

    def _on_system_resume(self):
        """Called when system resumes from suspend (D-Bus PrepareForSleep signal)"""
        try:
            print("[SUSPEND] System resumed - recovering audio and backends...", flush=True)
            time.sleep(2)  # Give audio/GPU drivers time to reinitialize

            if self.audio_capture.recover_audio_capture('post_suspend_resume'):
                # Reinitialize backend based on type
                backend = normalize_backend(self.config.get_setting('transcription_backend', 'pywhispercpp'))
                backend_reinit_success = True

                # Backends that use pywhispercpp (model in memory, CUDA context)
                pywhispercpp_variants = ['pywhispercpp', 'cpu', 'nvidia', 'amd']
                if backend in pywhispercpp_variants:
                    if not self.whisper_manager._reinitialize_model():
                        print("[SUSPEND] Recovery failed - model reinitialization failed", flush=True)
                        backend_reinit_success = False
                # WebSocket backend (persistent connection)
                elif backend == 'realtime-ws':
                    if not self.whisper_manager.initialize():
                        print("[SUSPEND] Recovery failed - WebSocket reinitialization failed", flush=True)
                        backend_reinit_success = False
                # Stateless backends (rest-api, parakeet) - no reinitialization needed
                # elif backend in ['rest-api', 'parakeet']:
                #     pass  # No persistent state to reinitialize

                # Write recovery result and clear background recovery flag only after ALL recovery steps complete
                if backend_reinit_success:
                    print("[SUSPEND] Recovery successful - microphone ready", flush=True)
                    self._write_recovery_result(True, 'suspend_resume')
                    with self._mic_state_lock:
                        self._mic_disconnected = False
                    self._background_recovery_needed.clear()
                else:
                    # Backend reinitialization failed - signal that recovery is still needed
                    if backend in pywhispercpp_variants:
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
                # Reinitialize backend based on type
                backend = normalize_backend(self.config.get_setting('transcription_backend', 'pywhispercpp'))
                backend_reinit_success = True

                # Backends that use pywhispercpp (model in memory, CUDA context)
                pywhispercpp_variants = ['pywhispercpp', 'cpu', 'nvidia', 'amd']
                if backend in pywhispercpp_variants:
                    if not self.whisper_manager._reinitialize_model():
                        backend_reinit_success = False
                # WebSocket backend (persistent connection)
                elif backend == 'realtime-ws':
                    if not self.whisper_manager.initialize():
                        backend_reinit_success = False
                # Stateless backends (rest-api, parakeet) - no reinitialization needed

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
        # Check audio capture availability
        if not self.audio_capture.is_available():
            print("[ERROR] Audio capture not available!")
            return False

        # Initialize whisper manager
        if not self.whisper_manager.initialize():
            print("[ERROR] Failed to initialize Whisper.")
            return False
        
        # Start global shortcuts
        if self.global_shortcuts:
            if not self.global_shortcuts.start():
                print("[ERROR] Failed to start global shortcuts!")
                print("[ERROR] Check permissions: you may need to be in 'input' group")
                return False
        else:
            print("[ERROR] Global shortcuts not initialized!")
            return False
        
        print("\n[READY] hyprwhspr ready - press shortcut to start dictation", flush=True)

        # Clean up any stale recovery file (tray script no longer creates these)
        if RECOVERY_REQUESTED_FILE.exists():
            try:
                RECOVERY_REQUESTED_FILE.unlink()
                print("[STARTUP] Removed stale recovery file", flush=True)
            except Exception:
                pass

        # Give microphone 1 second to fully initialize before checking for recovery
        # This prevents spurious errors on startup if device is still settling
        time.sleep(1)

        try:
            # Keep the application running
            while True:
                # Check for recording control requests from tray script (non-blocking)
                self._check_recording_control()
                # Check for recovery requests from tray script (non-blocking)
                self._attempt_recovery_if_needed()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Shutting down hyprwhspr...")
            self._cleanup()
        except Exception as e:
            print(f"[ERROR] Error in main loop: {e}")
            self._cleanup()
            return False
        
        return True

    def _cleanup(self):
        """Clean up resources when shutting down"""
        try:
            # Stop background recovery thread
            if hasattr(self, '_background_recovery_stop'):
                self._background_recovery_stop.set()
                if hasattr(self, '_background_recovery_thread') and self._background_recovery_thread and self._background_recovery_thread.is_alive():
                    print("[SHUTDOWN] Stopping background recovery thread...", flush=True)
                    self._background_recovery_thread.join(timeout=2.0)

            # Hide mic-osd overlay if visible
            self._hide_mic_osd()
            
            # Stop mic-osd daemon
            if hasattr(self, '_mic_osd_runner') and self._mic_osd_runner:
                try:
                    self._mic_osd_runner.stop()
                except Exception:
                    pass  # Silently fail - daemon cleanup is best-effort
            
            # Stop device monitor
            if hasattr(self, 'device_monitor') and self.device_monitor:
                self.device_monitor.stop()

            # Stop pulse monitor
            if hasattr(self, 'pulse_monitor') and self.pulse_monitor:
                try:
                    self.pulse_monitor.stop()
                except Exception:
                    pass  # Silently fail - pulse monitor cleanup is best-effort

            # Stop suspend monitor
            if hasattr(self, 'suspend_monitor') and self.suspend_monitor:
                try:
                    self.suspend_monitor.stop()
                except Exception:
                    pass  # Silently fail - suspend monitor cleanup is best-effort

            # Stop global shortcuts
            if self.global_shortcuts:
                self.global_shortcuts.stop()

            # Stop audio capture
            if self.is_recording:
                self.audio_capture.stop_recording()

            # Cleanup whisper manager (closes WebSocket connections, etc.)
            if self.whisper_manager:
                self.whisper_manager.cleanup()

            # Save configuration
            self.config.save_config()

            print("[CLEANUP] Cleanup completed")
            
        except Exception as e:
            print(f"[WARN] Error during cleanup: {e}")
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
    CLI_SUBCOMMANDS = ['setup', 'omarchy', 'config', 'waybar', 'systemd', 'status',
                       'model', 'validate', 'uninstall', 'backend', 'state', 'mic-osd']
    if len(sys.argv) > 1 and sys.argv[1] in CLI_SUBCOMMANDS:
        print(f"[REDIRECT] Detected CLI subcommand '{sys.argv[1]}', redirecting to CLI...")
        # Execute CLI with same arguments
        cli_path = Path(__file__).parent / 'cli.py'
        os.execv(sys.executable, [sys.executable, str(cli_path)] + sys.argv[1:])

    main()
    