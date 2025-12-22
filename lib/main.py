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

class hyprwhsprApp:
    """Main application class for hyprwhspr voice dictation (Headless Mode)"""

    def __init__(self):
        # Initialize core components
        self.config = ConfigManager()

        # Initialize audio capture with configured device
        audio_device_id = self.config.get_setting('audio_device', None)
        self.audio_capture = AudioCapture(device_id=audio_device_id)

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
        self.recovery_attempted_for_current_error = False  # Track if recovery was attempted for current error state
        self.last_recovery_time = 0.0  # Track when recovery last completed (for cooldown)
        
        # Lock to prevent concurrent recording starts (race condition protection)
        self._recording_lock = threading.Lock()

        # Set up global shortcuts (needed for headless operation)
        self._setup_global_shortcuts()

    def _setup_global_shortcuts(self):
        """Initialize global keyboard shortcuts"""
        try:
            shortcut_key = self.config.get_setting("primary_shortcut", "Super+Alt+D")
            push_to_talk = self.config.get_setting("push_to_talk", False)
            grab_keys = self.config.get_setting("grab_keys", True)
            selected_device_path = self.config.get_setting("selected_device_path", None)

            if push_to_talk:
                # Push-to-talk mode: register both press and release callbacks
                self.global_shortcuts = GlobalShortcuts(
                    shortcut_key,
                    self._on_shortcut_triggered,
                    self._on_shortcut_released,
                    device_path=selected_device_path,
                    grab_keys=grab_keys,
                )
            else:
                # Toggle mode: only register press callback
                self.global_shortcuts = GlobalShortcuts(
                    shortcut_key,
                    self._on_shortcut_triggered,
                    device_path=selected_device_path,
                    grab_keys=grab_keys,
                )
        except Exception as e:
            print(f"[ERROR] Failed to initialize global shortcuts: {e}")
            self.global_shortcuts = None

    def _on_shortcut_triggered(self):
        """Handle global shortcut trigger"""
        push_to_talk = self.config.get_setting("push_to_talk", False)

        if push_to_talk:
            # Push-to-talk mode: only start recording on key press
            if not self.is_recording:
                self._start_recording()
        else:
            # Toggle mode: start/stop recording
            if self.is_recording:
                self._stop_recording()
            else:
                self._start_recording()

    def _on_shortcut_released(self):
        """Handle global shortcut release (push-to-talk mode)"""
        push_to_talk = self.config.get_setting("push_to_talk", False)

        if push_to_talk and self.is_recording:
            # Push-to-talk mode: stop recording on key release
            self._stop_recording()

    def _start_recording(self):
        """Start voice recording"""
        # Use a lock to prevent concurrent starts (race condition protection)
        with self._recording_lock:
            if self.is_recording:
                return
            
            # Set flag immediately to prevent duplicate starts
            self.is_recording = True
        
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
                    self._notify_zero_volume("Microphone disconnected or not responding - please unplug and replug USB microphone, then try recording again", log_level="ERROR")
                    return  # Don't attempt recovery during user-initiated recording
                
                # Stream is working - start monitoring
                self._start_audio_level_monitoring()
                    
            except (RuntimeError, Exception) as e:
                print(f"[ERROR] Failed to start recording: {e}", flush=True)
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
            self.is_recording = False
            self._write_recording_status(False)

    def _cancel_recording_muted(self):
        """Cancel recording early due to muted microphone"""
        if not self.is_recording:
            return

        try:
            self.is_recording = False
            self._stop_audio_level_monitoring()
            self._write_recording_status(False)
            self.audio_capture.stop_recording()
            self.audio_manager.play_error_sound()
            self._notify_user("hyprwhspr", "Microphone muted - recording canceled", "normal")
        except Exception as e:
            print(f"[ERROR] Error canceling recording: {e}")

    def _stop_recording(self):
        """Stop voice recording and process audio"""
        if not self.is_recording:
            return

        try:
            self.is_recording = False
            
            # Stop audio level monitoring
            self._stop_audio_level_monitoring()
            
            # Write recording status to file for tray script
            self._write_recording_status(False)
            
            # Check backend type
            backend = self.config.get_setting('transcription_backend', 'pywhispercpp')
            if backend == 'local':
                backend = 'pywhispercpp'
            elif backend == 'remote':
                backend = 'rest-api'
            
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
                # Play error sound but don't set ERR state (likely intentional muting)
                self.audio_manager.play_error_sound()
            else:
                # Valid audio data - process it
                self.audio_manager.play_stop_sound()
                self._process_audio(audio_data)
                
        except Exception as e:
            print(f"[ERROR] Error stopping recording: {e}")

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
                hallucination_markers = ('blank audio', 'blank', 'video playback', 'music', 'music playing')
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
        """Notify user about zero-volume recording and optionally signal waybar"""
        # Print to logs (primary notification)
        print(f"[{log_level}] {message}", flush=True)
        
        # Desktop notification
        self._notify_user("hyprwhspr", message, "normal")
        
        # Optional: Write waybar signal file (atomic, no conflicts)
        # This allows waybar to when mic present but not recording
        try:
            signal_file = Path.home() / '.config' / 'hyprwhspr' / '.mic_zero_volume'
            # Use atomic write (write to temp file, then rename)
            temp_file = signal_file.with_suffix('.tmp')
            temp_file.write_text(str(int(time.time())))
            temp_file.replace(signal_file)
        except Exception:
            pass  # Silently fail - waybar signal is optional

    def _clear_zero_volume_signal(self):
        """Clear zero-volume signal file when valid audio is detected"""
        try:
            signal_file = Path.home() / '.config' / 'hyprwhspr' / '.mic_zero_volume'
            if signal_file.exists():
                signal_file.unlink()
        except Exception:
            pass  # Silently fail - waybar signal cleanup is optional

    def _write_recording_status(self, is_recording):
        """Write recording status to file for tray script"""
        try:
            status_file = Path.home() / '.config' / 'hyprwhspr' / 'recording_status'
            status_file.parent.mkdir(parents=True, exist_ok=True)
            
            if is_recording:
                with open(status_file, 'w') as f:
                    f.write('true')
            else:
                # Remove the file when not recording to avoid stale state
                if status_file.exists():
                    status_file.unlink()
        except Exception as e:
            print(f"[WARN] Failed to write recording status: {e}")

    def _start_audio_level_monitoring(self):
        """Start monitoring and writing audio levels to file"""
        if self.audio_level_thread and self.audio_level_thread.is_alive():
            return

        def monitor_audio_level():
            level_file = Path.home() / '.config' / 'hyprwhspr' / 'audio_level'
            level_file.parent.mkdir(parents=True, exist_ok=True)

            # Muted mic detection: 5e-7 threshold catches true digital silence but not quiet rooms
            zero_samples = 0
            zero_threshold = 5e-7
            samples_to_cancel = 10  # 1 second at 100ms intervals

            while self.is_recording:
                try:
                    level = self.audio_capture.get_audio_level()
                    with open(level_file, 'w') as f:
                        f.write(f'{level:.3f}')

                    # Early muted mic detection
                    if level < zero_threshold:
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

            # Clean up file when not recording
            try:
                if level_file.exists():
                    level_file.unlink()
            except:
                pass

        self.audio_level_thread = threading.Thread(target=monitor_audio_level, daemon=True)
        self.audio_level_thread.start()

    def _stop_audio_level_monitoring(self):
        """Stop audio level monitoring"""
        if self.audio_level_thread and self.audio_level_thread.is_alive():
            # Thread will exit when is_recording becomes False
            pass

    def _attempt_recovery_if_needed(self):
        """
        Check for recovery request from tray script and attempt recovery once per error state.
        
        This is called periodically (e.g., in main loop) to check if recovery is needed.
        Only attempts recovery once per error state to avoid infinite retry loops.
        """
        recovery_file = Path.home() / '.config' / 'hyprwhspr' / 'recovery_requested'
        
        # Check if recovery file exists
        if not recovery_file.exists():
            # No recovery requested - mic is working, reset flag
            if self.recovery_attempted_for_current_error:
                self.recovery_attempted_for_current_error = False
            return
        
        # Recovery file exists - check if we should attempt recovery
        # Don't trigger recovery if transcription is in progress
        if self.is_processing:
            return  # Skip recovery attempt during transcription
        
        # Don't trigger recovery if actively recording - recovery will interfere with recording
        if self.is_recording:
            return  # Skip recovery attempt during active recording
        
        # Check if recovery was already attempted for this error state
        if self.recovery_attempted_for_current_error:
            # Already attempted - don't try again
            return
        
        # Check file age - if very old (>60s), assume recovery was attempted and failed
        try:
            file_age = time.time() - recovery_file.stat().st_mtime
            if file_age > 60:
                # File is old - assume recovery was attempted and failed
                # Clear it to allow new error detection
                recovery_file.unlink()
                self.recovery_attempted_for_current_error = False
                return
        except Exception:
            pass
        
        # Clear the file now that we're about to attempt recovery
        try:
            recovery_file.unlink()
        except Exception as e:
            print(f"[RECOVERY] Warning: Could not clear recovery request file: {e}", flush=True)
        
        # Determine reason for recovery
        was_recording = self.is_recording
        reason = "mic_unavailable" if not was_recording else "mic_no_audio"
        
        print(f"[RECOVERY] Recovery requested by tray script ({reason} detected)", flush=True)
        
        # Mark that we're attempting recovery for this error state
        self.recovery_attempted_for_current_error = True
        
        # Attempt recovery (will handle stopping current recording if needed)
        if self.audio_capture.recover_audio_capture(f"tray_script_request_{reason}"):
            print("[RECOVERY] Recovery successful - mic should now be available", flush=True)
            
            # After successful audio recovery, also reinitialize model if needed
            # This handles suspend/resume cases where CUDA context is invalid
            backend = self.config.get_setting('transcription_backend', 'pywhispercpp')
            if backend == 'local':
                backend = 'pywhispercpp'
            elif backend == 'remote':
                backend = 'rest-api'
            
            if backend == 'pywhispercpp' and hasattr(self.whisper_manager, '_pywhisper_model') and self.whisper_manager._pywhisper_model:
                # Check if model needs reinitialization (long idle = suspend/resume)
                current_time = time.monotonic()
                if hasattr(self.whisper_manager, '_last_use_time'):
                    time_since_last = current_time - self.whisper_manager._last_use_time
                    if time_since_last > 300 and self.whisper_manager._last_use_time > 0:
                        print("[RECOVERY] Reinitializing model after audio recovery (suspend/resume detected)", flush=True)
                        self.whisper_manager._reinitialize_model()
            
            # Reset flag since recovery succeeded
            self.recovery_attempted_for_current_error = False
            
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
            # Keep flag set - recovery was attempted and failed, don't retry

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
        recovery_file = Path.home() / '.config' / 'hyprwhspr' / 'recovery_requested'
        if recovery_file.exists():
            try:
                recovery_file.unlink()
                print("[STARTUP] Removed stale recovery file", flush=True)
            except Exception:
                pass
        
        try:
            # Keep the application running
            while True:
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
    config_dir = Path.home() / '.config' / 'hyprwhspr'
    config_dir.mkdir(parents=True, exist_ok=True)
    _lock_file_path = config_dir / 'hyprwhspr.lock'
    
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
    main()
