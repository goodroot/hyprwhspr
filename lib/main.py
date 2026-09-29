#!/usr/bin/env python3
"""
hyprwhspr - stt
"""

import sys
sys.dont_write_bytecode = True
import time
import math
import threading
import os
import fcntl
import atexit
import subprocess
import json
from pathlib import Path

# Recovery dispatch must precede optional runtime imports.
if __name__ == '__main__' and (sys.argv[1:2] == ['update'] or sys.argv[1:3] in (['install', 'repair'], ['install', 'status'])):
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    from managed_install import main as lifecycle_main
    raise SystemExit(lifecycle_main(sys.argv[1:]))

try:
    import numpy as np
except ImportError:
    np = None  # Will be checked when needed


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
from hallucination import is_hallucination
from audio_capture import AudioCapture
from whisper_manager import WhisperManager
from service_log import log
from session_environment import ensure_wayland_display
from text_injector import TextInjector, InjectionOutcome
from processing_trace import build_processing_trace
from audio_manager import AudioManager
from playback_suppressor import PlaybackSuppressor
from paths import (
    RECORDING_CONTROL_FILE, LOCK_FILE, LONGFORM_SEGMENTS_DIR, SOCKET_FILE, DEBUG_RECORDINGS_DIR
)
from backend_utils import normalize_backend
from longform_controller import LongFormController
from recording_control_server import RecordingControlServer
from app.shortcuts import ShortcutsMixin
from app.silence import SilenceMixin
from app.feedback import FeedbackMixin
from app.control import ControlMixin
from app.recovery import RecoveryMixin

class hyprwhsprApp(ShortcutsMixin, SilenceMixin, FeedbackMixin, ControlMixin, RecoveryMixin):
    """Main application class for hyprwhspr voice dictation (Headless Mode)"""

    def _diagnostics_snapshot(self):
        from diagnostics import daemon_snapshot
        return daemon_snapshot(self)

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

        # Initialize playback suppression (duck volume or pause players while recording)
        ducking_percent = self.config.get_setting('audio_ducking_percent', 50)
        self.playback_suppressor = PlaybackSuppressor(reduction_percent=ducking_percent)

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
        self._file_transcription_active = False
        # Long-form keeps its own state machine, so it claims this flag under
        # _recording_lock to stay mutually exclusive with file transcription.
        self._longform_active = False
        # Held across a model load/unload, which runs outside _recording_lock.
        self._model_operation_active = False
        # Covers the short stop-recording window before _process_audio claims
        # is_processing, so file requests cannot steal the backend in between.
        self._recording_finalizing = threading.Event()
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
        self._playback_lock = threading.Lock()
        self._playback_session = None
        self._recording_session = None
        self._recording_starting = False
        self._start_settled = threading.Event()
        self._start_settled.set()
        self._start_owner = None
        self._playback_shutdown = False

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
        # Set when a start was refused while loading; the next successful init
        # announces Ready even if it was quick
        self._notify_when_ready = False

        self._recording_control_server = RecordingControlServer(
            fifo_path=RECORDING_CONTROL_FILE,
            socket_path=SOCKET_FILE,
            on_command=self._handle_control_command,
            is_recording=lambda: self.is_recording,
            on_file_transcribe=self._handle_file_transcribe,
            on_recover=self._handle_recovery,
            on_diagnostics=self._diagnostics_snapshot,
        )

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

        # Push-to-talk hold-to-lock state (push_to_talk mode)
        self._ptt_press_time = None        # time.monotonic() of the press the current hold is measured from
        self._ptt_locked = False           # guarded by _recording_lock, like the press time

        # Continuous mode state (auto-paste on speech pause)
        self._continuous_silence_thread = None
        self._continuous_silence_stop = threading.Event()
        self._continuous_flush_lock = threading.Lock()
        self._continuous_transcription_done = threading.Event()
        self._continuous_transcription_done.set()  # no transcription in flight
        self._continuous_cancelled = False  # set on cancel to suppress in-flight injection
        self._continuous_delivery_failure_notified = False

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
            notify_capture=self._notify_capture,
            set_visualizer_state=self._set_visualizer_state,
            show_mic_osd=self._show_mic_osd,
            hide_mic_osd=self._hide_mic_osd,
            show_result_and_hide=self._show_result_and_hide,
            write_recording_status=self._write_recording_status,
            set_processing=self._set_longform_processing,
            claim_recording=self._claim_longform_recording,
            release_recording=self._release_longform_recording,
            hallucination_markers=self.config.get_hallucination_markers(),
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
        self._recording_control_server.prepare_fifo()

        # Pre-initialize mic-osd daemon (eliminates latency on recording)
        self._mic_osd_runner = None
        if self.config.get_setting('mic_osd_enabled', True):
            try:
                from mic_osd import MicOSDRunner, NotificationPresenter
                def _use_notification_status(reason: str):
                    if _looks_like_wlroots_session():
                        log(f"[INIT] {reason}, falling back to notifications")
                    presenter = NotificationPresenter(
                        active_timeout_ms=self.config.get_setting('notification_timeout_ms', 5000))
                    if presenter.is_available():
                        self._mic_osd_runner = presenter
                        log("[INIT] Recording status via notifications")
                    else:
                        log("[WARN] No layer-shell overlay and no desktop notifications; recording has no status indicator")

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
                        log("[INIT] Mic-OSD daemon started")
                    else:
                        log("[WARN] Failed to start mic-osd daemon")
                        _use_notification_status("mic-osd daemon failed to start")
                else:
                    # No layer-shell (e.g. GNOME/Mutter): the overlay would steal
                    # keyboard focus and swallow the paste keystroke. Show recording
                    # status via desktop notifications instead.
                    _use_notification_status("layer-shell not supported")
            except Exception as e:
                log(f"[WARN] Failed to initialize recording status indicator: {e}")
                import traceback
                traceback.print_exc()

        if hasattr(self.whisper_manager, 'set_realtime_partial_callback'):
            self.whisper_manager.set_realtime_partial_callback(self._set_mic_osd_preview_text)

        # Set up global shortcuts (needed for headless operation)
        self._setup_global_shortcuts()

    def _get_float_setting(self, key, default=0.0):
        """Coerce a config value to float; falls back to `default` if missing, invalid, or non-finite (NaN/Infinity)."""
        value = self.config.get_setting(key, default)
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return value if math.isfinite(value) else default

    def _set_longform_processing(self, processing):
        with self._recording_lock:
            self.is_processing = processing

    def _claim_longform_recording(self):
        """Reserve the backend for a long-form session; False if busy or not loaded."""
        with self._recording_lock:
            busy = self._file_transcription_active or self._model_operation_active
            loading = self._model_initializing or self._backend_init_failed
            if not (busy or loading):
                self._longform_active = True
                return True
        if loading:
            # As for a refused normal start: announce Ready later, retry a failed init
            self._notify_when_ready = True
            if self._backend_init_failed:
                self._start_backend_init_background()
        return False

    def _release_longform_recording(self):
        with self._recording_lock:
            self._longform_active = False

    def _suppress_recording_playback(self, session):
        """Serialize external audio operations without holding recording state."""
        with self._playback_lock:
            with self._recording_lock:
                active = (self.is_recording and self._recording_session is session
                          and not self._playback_shutdown)
            if not active:
                return False
            if self.config.get_setting('audio_ducking', False):
                self._playback_session = session
                try:
                    self.playback_suppressor.suppress(
                        mode=self.config.get_setting('audio_ducking_mode', 'duck'),
                        reduction_percent=self.config.get_setting('audio_ducking_percent', 50))
                finally:
                    with self._recording_lock:
                        active = (self.is_recording and self._recording_session is session
                                  and not self._playback_shutdown)
                    if not active:
                        self.playback_suppressor.restore()
                        self._playback_session = None
            return active

    def _restore_recording_playback(self, session=None, shutdown=False):
        with self._playback_lock:
            if shutdown:
                self._playback_shutdown = True
            owned = session is not None and self._playback_session is session
            if shutdown or owned:
                self.playback_suppressor.restore()
                self._playback_session = None

    def _wait_for_start_settled(self):
        """Let an in-flight start finish before a stop touches capture.

        Otherwise the stop can run before capture opens (orphaning it), or race
        the aborted start's teardown and lose the recorded audio.
        """
        if self._start_owner != threading.get_ident():
            if not self._start_settled.wait(timeout=5.0):
                log("[WARN] Recording start still settling; stopping capture anyway")

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
            if self._recording_starting:
                # A stopped start is still releasing capture; refuse until it settles.
                blocked = 'starting'
            elif self._model_initializing:
                blocked = 'initializing'
            elif self._backend_init_failed:
                blocked = 'init-failed'
            elif self._file_transcription_active:
                blocked = 'file-transcription'
            elif self._model_operation_active:
                blocked = 'model-operation'
            elif getattr(self.whisper_manager, '_model_manually_unloaded', False):
                blocked = 'unloaded'
            elif self.whisper_manager.realtime_client_missing():
                blocked = 'realtime-reconnect'
            else:
                blocked = None
                # Set flag immediately to prevent duplicate starts
                self.is_recording = True
                self._recording_starting = True
                self._start_settled.clear()
                self._start_owner = threading.get_ident()
                session = object()
                self._recording_session = session
                # Store language override for this recording session
                self._current_language_override = language_override
                # Push-to-talk hold is measured from this accepted press
                self._ptt_press_time = time.monotonic()
                self._ptt_locked = False

        # A capture client self-triggered this start over the FIFO and is now
        # blocking on completion. Release it here or it waits forever and keeps
        # the capture slot occupied (no recording will ever finish for it).
        if blocked is not None:
            self._release_blocked_capture()

        # Model is still loading in background
        if blocked in ('initializing', 'init-failed', 'realtime-reconnect'):
            self._notify_when_ready = True

        if blocked == 'starting':
            self._notify_user("hyprwhspr", "Still stopping — try again", urgency="normal")
            log("[CONTROL] Recording blocked: previous start still settling")
            return

        if blocked == 'initializing':
            self._notify_user("hyprwhspr", "Model still loading, please wait…", urgency="normal")
            log("[CONTROL] Recording blocked: model is still initializing")
            return

        # Backend init failed (e.g. deps missing at boot): retry it instead
        # of recording audio that can never be transcribed
        if blocked == 'init-failed':
            self._notify_user("hyprwhspr", "Backend failed to load — retrying, try again shortly", urgency="normal")
            log("[CONTROL] Recording blocked: backend init failed - retrying")
            self._start_backend_init_background()
            return

        if blocked == 'file-transcription':
            self._notify_user("hyprwhspr", "File transcription in progress", urgency="normal")
            log("[CONTROL] Recording blocked: file transcription in progress")
            return

        if blocked == 'model-operation':
            self._notify_user("hyprwhspr", "Model load in progress — try again shortly", urgency="normal")
            log("[CONTROL] Recording blocked: model load/unload in progress")
            return

        # Realtime client was torn down; rebuild in background instead of blocking
        if blocked == 'realtime-reconnect':
            self._notify_user("hyprwhspr", "Reconnecting — try again in a moment", urgency="normal")
            log("[CONTROL] Recording blocked: realtime client not connected - reconnecting")
            self._start_backend_init_background()
            return

        # Model was deliberately unloaded to free GPU resources
        if blocked == 'unloaded':
            self._notify_user(
                "hyprwhspr",
                "Model unloaded — run: hyprwhspr model reload",
                urgency="normal",
            )
            log("[CONTROL] Recording blocked: model is unloaded. Run: hyprwhspr model reload")
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
                self._release_blocked_capture()
                self._hide_mic_osd()
                self._stop_audio_level_monitoring()
                self._notify_zero_volume(
                    self._realtime_unavailable_message(),
                    log_level="ERROR",
                )
                # Restore audio if it was ducked or paused
                self._restore_recording_playback(session)
                return
            
            # Helper function to verify stream is working and play sound
            def verify_and_play_sound():
                """Wait for callbacks and play sound if stream works"""
                import time
                start_time = time.monotonic()
                while time.monotonic() - start_time < 1.5:  # Wait up to 1.5s
                    with self._recording_lock:
                        if not self.is_recording or self._playback_shutdown:
                            return None
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
            
            def abandon_start():
                # Stop/cancel wait for this start to settle and then own capture
                # teardown (and the audio). Shutdown may already have stopped
                # capture before it opened here, so release it ourselves.
                if self._playback_shutdown:
                    self.audio_capture.stop_recording()

            # Start audio capture (with streaming callback for realtime-ws)
            try:
                if not self.audio_capture.start_recording(streaming_callback=streaming_callback):
                    raise RuntimeError("start_recording() returned False")
                
                # Verify stream is working before playing sound
                verified = verify_and_play_sound()
                if verified is None:
                    abandon_start()
                    return
                if not verified:
                    # Stream broken - stop recording (thread will clean up stream)
                    self.audio_capture.stop_recording()

                    # Reset state
                    with self._recording_lock:
                        self.is_recording = False
                    self._write_recording_status(False)
                    self._release_blocked_capture()
                    
                    # Hide mic-osd visualization
                    self._hide_mic_osd()

                    message = self._mic_failure_message(
                        "Microphone not responding - please unplug and replug USB microphone, then try recording again")
                    self._notify_zero_volume(message, log_level="ERROR")

                    # Restore audio if it was ducked or paused
                    self._restore_recording_playback(session)
                    return  # Don't attempt recovery during user-initiated recording

                if not self._suppress_recording_playback(session):
                    abandon_start()
                    return

                # Stream is verified working - show mic-osd visualization
                log("Recording started")
                self._show_mic_osd()
                
                # Additional stability check - verify stream continues working
                stable = verify_stream_stable()
                with self._recording_lock:
                    stopped = not self.is_recording or self._playback_shutdown
                if stopped:
                    # Stop freezes the frame count; that is not an unstable stream.
                    abandon_start()
                    return
                if not stable:
                    # Stream stopped working shortly after starting
                    self.audio_capture.stop_recording()
                    with self._recording_lock:
                        self.is_recording = False
                    self._write_recording_status(False)
                    self._release_blocked_capture()
                    
                    # Hide mic-osd visualization
                    self._hide_mic_osd()

                    fallback = "Microphone stream unstable - please wait a moment and try recording again"
                    message = self._mic_failure_message(fallback)
                    self._notify_zero_volume(message, log_level="WARN" if message == fallback else "ERROR")

                    # Restore audio if it was ducked or paused
                    self._restore_recording_playback(session)
                    return
                
                with self._recording_lock:
                    if not self.is_recording or self._playback_shutdown:
                        return

                # Recording is confirmed working - abort any in-progress recovery and clear background retries
                try:
                    self.audio_capture.abort_recovery()
                except Exception:
                    pass
                if self._background_recovery_needed.is_set():
                    log("[HEALTH] Recording succeeded - canceling background recovery")
                    self._background_recovery_needed.clear()
                
                # Stream is working and stable - start monitoring
                self._start_audio_level_monitoring()
                    
            except (RuntimeError, Exception) as e:
                log(f"[ERROR] Failed to start recording: {e}")

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
                self._release_blocked_capture()
                self._notify_zero_volume(
                    self._mic_failure_message(
                        "Microphone disconnected or not responding - please unplug and replug USB microphone, then try recording again"),
                    log_level="ERROR")

                # Restore audio if it was ducked or paused
                self._restore_recording_playback(session)
                return

        except Exception as e:
            log(f"[ERROR] Failed to start recording: {e}")

            # Clean up resources
            self._hide_mic_osd()
            self._stop_audio_level_monitoring()

            self.whisper_manager.close_realtime_connection("recording start failure")

            with self._recording_lock:
                self.is_recording = False
            self._write_recording_status(False)
            # No recording will complete, so any capture client waiting on this
            # start has to be released here too or it holds the slot forever
            self._release_blocked_capture()

            # Restore audio if it was ducked or paused
            self._restore_recording_playback(session)

        finally:
            with self._recording_lock:
                self._recording_starting = False
            self._start_settled.set()

    def _cleanup_recording_state(self, session=None):
        """Best-effort cleanup after any recording ends. Safe to call multiple times."""
        session = session if session is not None else self._recording_session
        # Release recording state and capture clients before teardown can block.
        try:
            self._write_recording_status(False)
        except Exception:
            pass
        self._notify_capture("", final=True)
        self._restore_recording_playback(session)

        # Retire this recording's monitor before hide can block long enough for
        # a later recording to install its own monitor in the shared slot.
        self._autostop_stop_silence_monitor()
        try:
            self._hide_mic_osd()
        except Exception:
            pass

        try:
            self._clear_mic_osd_preview_text()
        except Exception:
            pass
        try:
            self._stop_audio_level_monitoring()
        except Exception:
            pass
        try:
            self._restore_recording_playback(session)
        except Exception:
            pass

    def _cancel_recording_muted(self):
        """Cancel recording early due to muted microphone"""
        with self._recording_lock:
            if not self.is_recording:
                return
            session = self._recording_session
            self.is_recording = False
            self._current_language_override = None  # Clear language override on error
            self._ptt_reset()

        log("[MUTE] Recording cancelled - microphone returned silence for 1 second")

        self._cleanup_recording_state(session)
        try:
            self._wait_for_start_settled()
            self.audio_capture.stop_recording()
            self.audio_manager.play_error_sound()
            # Note: No desktop notification - tray will detect muted state via audio level monitoring
        except Exception as e:
            log(f"[ERROR] Error canceling recording: {e}")

    def _cancel_recording(self):
        """Cancel recording and discard audio without transcribing or injecting text"""
        with self._recording_lock:
            if not self.is_recording:
                return
            session = self._recording_session
            self.is_recording = False
            self._current_language_override = None
            self._ptt_reset()

        log("Recording cancelled (discarded)")

        self._cleanup_recording_state(session)
        try:
            # Stop capture and discard the audio data
            self._wait_for_start_settled()
            self.audio_capture.stop_recording()

            # Discard buffered realtime audio but keep the WebSocket alive so the
            # next recording starts without paying a fresh handshake
            self.whisper_manager.discard_realtime_audio()

            self.audio_manager.play_error_sound()
        except Exception as e:
            log(f"[ERROR] Error cancelling recording: {e}")

    _DEBUG_RECORDINGS_KEEP = 3

    def _save_debug_recording(self, audio_data):
        """Keep the last few raw recordings in DEBUG_RECORDINGS_DIR (debug_recordings).

        Called from both the stop path and continuous-mode flushes, which can
        land in the same second and race on pruning.
        """
        if audio_data is None or not self.config.get_setting('debug_recordings', False):
            return
        try:
            DEBUG_RECORDINGS_DIR.mkdir(mode=0o700, exist_ok=True)
            millis = time.time_ns() // 1_000_000 % 1000
            path = DEBUG_RECORDINGS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{millis:03d}.wav"
            self.audio_capture.save_audio_to_wav(audio_data, str(path))
            for old in sorted(DEBUG_RECORDINGS_DIR.glob('*.wav'))[:-self._DEBUG_RECORDINGS_KEEP]:
                old.unlink(missing_ok=True)
        except Exception as e:
            log(f"[WARN] Failed to save debug recording: {e}")

    def _stop_recording(self):
        """Stop voice recording and process audio"""
        with self._recording_lock:
            if not self.is_recording:
                return
            self.is_recording = False
            self._recording_finalizing.set()
            session = self._recording_session
            self._ptt_reset()

        try:
            log("Recording stopped")

            # Tear down the auto-stop silence monitor if it was running (toggle/auto modes)
            self._autostop_stop_silence_monitor()

            self._clear_mic_osd_preview_text()

            # Set visualizer to processing state (keep it visible during transcription)
            self._set_visualizer_state('processing')
            
            # Stop audio level monitoring
            self._stop_audio_level_monitoring()
            
            # Write recording status to file for tray script
            self._write_recording_status(False)

            # Restore system audio if it was ducked or paused
            self._restore_recording_playback(session)

            # Check backend type
            backend = self.config.get_setting('transcription_backend', 'pywhispercpp')
            backend = normalize_backend(backend)
            
            # Stop audio capture
            self._wait_for_start_settled()
            audio_data = self.audio_capture.stop_recording()
            self._save_debug_recording(audio_data)

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
                self._notify_capture("", final=True)
            elif self._is_zero_volume(audio_data):
                # Audio data exists but is all zeros - mic not producing sound
                # Play error sound and notify user (may be intentional muting, but still inform)
                self.audio_manager.play_error_sound()
                self._notify_zero_volume("Microphone not producing audio (zero volume detected). This may be intentional muting, or the microphone may need to be reseated.")
                # Show error state and hide OSD
                self._show_result_and_hide(False)
                self._notify_capture("", final=True)
            else:
                # Valid audio data - process it
                self.audio_manager.play_stop_sound()
                self._process_audio(audio_data)
                
            # Clear language override after transcription completes
            self._current_language_override = None
                
        except Exception as e:
            log(f"[ERROR] Error stopping recording: {e}")
            self._notify_capture("", final=True)
            # Ensure cleanup even if error occurs
            try:
                self.is_recording = False
                self._current_language_override = None  # Clear language override on cancel
                self._show_result_and_hide(False)
                self._stop_audio_level_monitoring()
                self._write_recording_status(False)
                self._continuous_stop_silence_monitor()
                self._restore_recording_playback(session)

                self.whisper_manager.close_realtime_connection("recording stop error")
            except Exception:
                pass  # Best effort cleanup
        finally:
            self._recording_finalizing.clear()

    def _process_audio(self, audio_data):
        """Process captured audio through Whisper"""
        with self._recording_lock:
            if self.is_processing:
                return
            self.is_processing = True

        success = False
        try:
            # Transcribe audio with language override if set
            transcription = self.whisper_manager.transcribe_audio(
                audio_data,
                sample_rate=self.audio_capture.sample_rate,
                language_override=self._current_language_override,
            )

            if transcription and transcription.strip():
                text = transcription.strip()

                # Filter out Whisper hallucination markers - don't touch clipboard
                if is_hallucination(text, self.config.get_hallucination_markers()):
                    log(f"[INFO] Whisper hallucination detected: {text!r} - ignoring")
                    if self._recording_control_server.is_trace_capture():
                        self._notify_capture(text, final=True)
                    self.audio_manager.play_error_sound()
                    success = False
                    # Explicitly handle cleanup before returning to ensure visualizer state is updated
                    with self._recording_lock:
                        self.is_processing = False
                    self._show_result_and_hide(False)
                    return

                # Inject text
                outcome = self._inject_text(text)
                success = outcome != InjectionOutcome.FAILED
            else:
                log("[WARN] No transcription generated")
                self.audio_manager.play_error_sound()

        except Exception as e:
            log(f"[ERROR] Error processing audio: {e}")
        finally:
            self._notify_capture("", final=True)
            self._clear_mic_osd_preview_text()
            with self._recording_lock:
                self.is_processing = False
            # Show success/error state and hide OSD after delay
            self._show_result_and_hide(success)

    def _notify_capture(self, text="", final=True):
        """Complete a capture request, encoding trace requests as one JSON document."""
        if self._recording_control_server.is_trace_capture():
            text = json.dumps(
                build_processing_trace(text, self.config),
                ensure_ascii=False,
                separators=(',', ':'),
            ) + '\n'
        self._recording_control_server.notify_capture(text, final=final)

    def _release_blocked_capture(self):
        """Finish a capture request whose recording was refused before it began."""
        if self._recording_control_server.has_capture_subscriber():
            self._notify_capture("", final=True)

    def _inject_text(self, text):
        """Inject transcribed text into active application"""

        # Capture mode: route text to client instead of injecting into active app
        if self._recording_control_server.has_capture_subscriber():
            self._notify_capture(text, final=True)
            return InjectionOutcome.INJECTED

        try:
            outcome = self.text_injector.inject_text(text)
            if outcome == InjectionOutcome.FAILED:
                log(f"[ERROR] Text injection failed ({len(text)} chars)")
                notify = True
                if self.config.get_setting('recording_mode', 'toggle') == 'continuous':
                    with self._recording_lock:
                        notify = not self._continuous_delivery_failure_notified
                        self._continuous_delivery_failure_notified = True
                if notify:
                    self._notify_user(
                        "hyprwhspr", "Text delivery failed. Recover with hyprwhspr record copy-last or record paste-last",
                        urgency="normal",
                    )
                return InjectionOutcome.FAILED

            if outcome == InjectionOutcome.CONSUMED:
                log("[INJECT] Post-transcription hook consumed transcription")
            else:
                log(f"[INJECT] Injection dispatched ({len(text)} chars)")

            # Text injection succeeded (or was intentionally consumed) - system is fully healthy
            # Cancel any pending background recovery
            if self._background_recovery_needed.is_set():
                log("[HEALTH] Successful recording detected - canceling background recovery")
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
            return outcome
        except Exception as e:
            log(f"[ERROR] Text injection failed: {e}")
            return InjectionOutcome.FAILED

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
                        log(f"[INIT] Restored default source: {source_name}")
                    else:
                        err = result.stderr.decode(errors='replace').strip()
                        log(f"[WARN] Could not restore default source '{source_name}': {err}")
            except Exception as e:
                log(f"[WARN] Could not restore default source: {e}")

        # Check audio capture availability
        if not self.audio_capture.is_available():
            log("[ERROR] Audio capture not available!")
            return False

        # Start global shortcuts (unless using Hyprland compositor bindings)
        use_hypr_bindings = self.config.get_setting("use_hypr_bindings", False)
        if self.global_shortcuts:
            if not self.global_shortcuts.start():
                log("[ERROR] Failed to start global shortcuts!")
                log("[ERROR] Check permissions: you may need to be in 'input' group")
                return False
        elif not use_hypr_bindings:
            log("[ERROR] Global shortcuts not initialized!")
            return False

        self._recording_control_server.start()

        # Initialize whisper backend. Slow backends (e.g. cohere-transcribe loading a
        # 4 GB model onto the GPU) run in a background thread so shortcuts and the FIFO
        # listener are active immediately. Recording is blocked until ready.
        if self.whisper_manager.configured_backend_loads_in_background():
            log(f"\n[INIT] Loading model in background (shortcuts active, recording will unblock when ready)...")
            self._start_backend_init_background()
        else:
            if not self.whisper_manager.initialize():
                # Stay alive with recording blocked; the record gate retries init
                log("[ERROR] Failed to initialize backend - will retry on next record attempt")
                self._backend_init_failed = True

        if use_hypr_bindings:
            log("\n[READY] hyprwhspr ready - using Hyprland compositor bindings")
        else:
            log("\n[READY] hyprwhspr ready - press shortcut to start dictation")

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
            log("\n[SHUTDOWN] Shutting down hyprwhspr...")
            self._cleanup()
        except Exception as e:
            log(f"[ERROR] Error in main loop: {e}")
            self._cleanup()
            return False
        
        return True

    def _cleanup(self):
        """Clean up resources when shutting down"""
        self._playback_shutdown = True
        def cleanup_step(name, action):
            try:
                action()
            except Exception as exc:
                log(f"[WARN] Cleanup step {name!r} failed: {exc}")

        def stop_thread(stop_name, thread_name, label, timeout):
            stop = getattr(self, stop_name, None)
            if stop is None:
                return
            stop.set()
            thread = getattr(self, thread_name, None)
            if thread and thread.is_alive():
                log(f"[SHUTDOWN] Stopping {label}...")
                thread.join(timeout=timeout)
                if thread.is_alive():
                    log(f"[WARN] {label} did not stop cleanly")

        def call_optional(attribute, method):
            target = getattr(self, attribute, None)
            if target is not None:
                getattr(target, method)()

        try:
            cleanup_step("stop recording control server", self._recording_control_server.stop)

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

            # Shutting down mid-recording must not leave other apps ducked or paused
            cleanup_step("restore playback", lambda: self._restore_recording_playback(shutdown=True))

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

            log("[CLEANUP] Cleanup completed")

        except Exception as e:
            log(f"[WARN] Error during cleanup: {e}")
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
        log("[ERROR] hyprwhspr is already running!")
        if lock_message:
            log(f"[ERROR] Detected via: {lock_message}")
        log("\n[INFO] To check the status of the running instance:")
        log("  • Run: hyprwhspr status")
        log("\n[INFO] To stop the running instance:")
        log("  • If running via systemd: systemctl --user stop hyprwhspr")
        log("  • If running manually: kill the process or press Ctrl+C in its terminal")
        log("\n[INFO] For more information, run: hyprwhspr --help")
        sys.exit(1)
    
    # Fallback: also check via process detection
    is_running, how = _is_hyprwhspr_running()
    if is_running:
        # Release lock since we detected another instance
        _release_lock_file()
        log("[ERROR] hyprwhspr is already running!")
        log(f"[ERROR] Detected via: {how}")
        log("\n[INFO] To check the status of the running instance:")
        log("  • Run: hyprwhspr status")
        log("\n[INFO] To stop the running instance:")
        log("  • If running via systemd: systemctl --user stop hyprwhspr")
        log("  • If running manually: kill the process or press Ctrl+C in its terminal")
        log("\n[INFO] For more information, run: hyprwhspr --help")
        sys.exit(1)
    
    try:
        app = hyprwhsprApp()
        app.run()
    except KeyboardInterrupt:
        log("\n[SHUTDOWN] Stopping hyprwhspr...")
        if 'app' in locals():
            app._cleanup()
        _release_lock_file()
    except Exception as e:
        log(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        _release_lock_file()
        sys.exit(1)


if __name__ == "__main__":
    # Safety check: if a CLI subcommand was passed, redirect to CLI instead of starting the service
    # This handles cases where an old bin/hyprwhspr wrapper doesn't recognize newer CLI subcommands
    # Keep in sync with the subcommand route in bin/hyprwhspr
    CLI_SUBCOMMANDS = ['update', 'setup', 'install', 'config', 'waybar', 'noctalia', 'systemd', 'status',
                       'model', 'validate', 'uninstall', 'backend', 'state', 'mic-osd',
                       'keyboard', 'record', 'test', 'transcribe']
    if len(sys.argv) > 1 and sys.argv[1] in CLI_SUBCOMMANDS:
        log(f"[REDIRECT] Detected CLI subcommand '{sys.argv[1]}', redirecting to CLI...")
        # Execute CLI with same arguments
        cli_path = Path(__file__).parent / 'cli.py'
        os.execv(sys.executable, [sys.executable, str(cli_path)] + sys.argv[1:])

    main()
    
