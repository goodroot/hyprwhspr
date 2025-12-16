#!/usr/bin/env python3
"""
hyprwhspr - Voice dictation application for Hyprland (Headless Mode)
Fast, reliable speech-to-text with instant text injection
"""

import sys
import time
import threading
import os
import fcntl
import atexit
from pathlib import Path

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

        self.whisper_manager = WhisperManager()
        self.text_injector = TextInjector(self.config)
        self.global_shortcuts = None

        # Application state
        self.is_recording = False
        self.is_processing = False
        self.current_transcription = ""
        self.audio_level_thread = None

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
        if self.is_recording:
            return

        try:
            self.is_recording = True
            
            # Write recording status to file for tray script
            self._write_recording_status(True)
            
            # Play start sound
            self.audio_manager.play_start_sound()
            
            # Check if using realtime-ws backend and get streaming callback
            streaming_callback = self.whisper_manager.get_realtime_streaming_callback()
            if streaming_callback:
                print(f'[DEBUG] Streaming callback obtained for realtime-ws', flush=True)
            
            # Start audio capture (with streaming callback for realtime-ws)
            self.audio_capture.start_recording(streaming_callback=streaming_callback)
            
            # Start audio level monitoring thread
            self._start_audio_level_monitoring()
            
        except Exception as e:
            print(f"[ERROR] Failed to start recording: {e}")
            self.is_recording = False
            self._write_recording_status(False)

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
            
            # Play stop sound
            self.audio_manager.play_stop_sound()
            
            # For realtime-ws, audio was already streamed via callback
            # We still need audio_data for transcribe_audio (it will commit and wait)
            if audio_data is not None:
                self._process_audio(audio_data)
            else:
                print("[WARN] No audio data captured")
                
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
                self.current_transcription = transcription.strip()
                
                # Inject text
                self._inject_text(self.current_transcription)
            else:
                print("[WARN] No transcription generated")
                
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
            
            while self.is_recording:
                try:
                    level = self.audio_capture.get_audio_level()
                    with open(level_file, 'w') as f:
                        f.write(f'{level:.3f}')
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
        
        print("\n[READY] hyprwhspr ready - press shortcut to start dictation")
        
        try:
            # Keep the application running
            while True:
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
