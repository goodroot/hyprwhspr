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
from pathlib import Path

# Recovery dispatch must precede optional runtime imports.
if __name__ == '__main__' and (sys.argv[1:2] == ['update'] or sys.argv[1:3] in (['install', 'repair'], ['install', 'status'])):
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    from managed_install import main as lifecycle_main
    raise SystemExit(lifecycle_main(sys.argv[1:]))


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
from service_log import log
from session_environment import ensure_wayland_display
from text_injector import TextInjector
from audio_manager import AudioManager
from playback_suppressor import PlaybackSuppressor
from paths import RECORDING_CONTROL_FILE, LOCK_FILE, LONGFORM_SEGMENTS_DIR, SOCKET_FILE
from longform_controller import LongFormController
from recording_control_server import RecordingControlServer
from app.recording import RecordingMixin
from app.shortcuts import ShortcutsMixin
from app.silence import SilenceMixin
from app.feedback import FeedbackMixin
from app.control import ControlMixin
from app.recovery import RecoveryMixin

class hyprwhsprApp(RecordingMixin, ShortcutsMixin, SilenceMixin, FeedbackMixin, ControlMixin, RecoveryMixin):
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
        self._ptt_locked = False           # written under _recording_lock, like the press time

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
    
