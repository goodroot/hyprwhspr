"""
MicOSD Runner - Daemon-based wrapper for instant mic-osd overlay.

Spawns mic-osd once in daemon mode, then uses SIGUSR1/SIGUSR2 to show/hide.
This eliminates subprocess spawn latency on each recording.
"""

import glob
import subprocess
import signal
import sys
import os
import threading
import time
from pathlib import Path

# Import paths
try:
    from ..src.paths import MIC_OSD_PID_FILE, VISUALIZER_STATE_FILE, TRANSCRIPT_PREVIEW_FILE
except ImportError:
    # Fallback for direct execution
    from src.paths import MIC_OSD_PID_FILE, VISUALIZER_STATE_FILE, TRANSCRIPT_PREVIEW_FILE


class MicOSDRunner:
    """
    Daemon-based runner for the mic-osd overlay.
    
    Spawns mic-osd in daemon mode at init, then signals it to show/hide.
    """

    PREVIEW_WRITE_INTERVAL_SECONDS = 0.05
    
    def __init__(self):
        self._process = None
        self._mic_osd_dir = Path(__file__).parent
        self._orphaned_daemon_pid = None  # Track PID when reusing orphaned daemon
        self._preview_lock = threading.Lock()
        self._last_preview_write_at = 0.0
        self._pending_preview_text = None
        self._preview_flush_timer = None
    
    @staticmethod
    def is_available() -> bool:
        """Check if mic-osd can run."""
        try:
            import cairo  # noqa: F401
            import gi
            gi.require_version('Gtk', '4.0')
            gi.require_version('Gtk4LayerShell', '1.0')
            return True
        except (ImportError, ValueError):
            return False
    
    @staticmethod
    def _get_distro_packages() -> tuple:
        """Return (gtk_pkg, layer_shell_pkg) package names for current distro."""
        # Check for common distro indicators
        try:
            if Path('/etc/debian_version').exists():
                return ('python3-gi python3-cairo gir1.2-gtk-4.0', 'gir1.2-gtk4layershell-1.0')
            elif Path('/etc/arch-release').exists():
                return ('python-gobject python-cairo gtk4', 'gtk4-layer-shell')
            elif Path('/etc/fedora-release').exists():
                return ('python3-gobject python3-cairo gtk4', 'gtk4-layer-shell')
            elif Path('/etc/os-release').exists():
                content = Path('/etc/os-release').read_text(encoding='utf-8').lower()
                if 'debian' in content or 'ubuntu' in content:
                    return ('python3-gi python3-cairo gir1.2-gtk-4.0', 'gir1.2-gtk4layershell-1.0')
                elif 'fedora' in content or 'rhel' in content:
                    return ('python3-gobject python3-cairo gtk4', 'gtk4-layer-shell')
                elif 'suse' in content:
                    return ('python3-gobject python3-pycairo typelib-1_0-Gtk-4_0', 'gtk4-layer-shell')
        except Exception:
            pass
        # Default to Arch-style names
        return ('python-gobject python-cairo gtk4', 'gtk4-layer-shell')

    @staticmethod
    def get_unavailable_reason() -> str:
        """Get reason why mic-osd is unavailable."""
        gtk_pkg, layer_pkg = MicOSDRunner._get_distro_packages()
        try:
            import cairo  # noqa: F401
        except ImportError:
            return f"PyCairo bindings not installed. Install: {gtk_pkg}"
        try:
            import gi
            gi.require_version('Gtk', '4.0')
        except (ImportError, ValueError):
            return f"GTK4 bindings not installed. Install: {gtk_pkg}"
        try:
            gi.require_version('Gtk4LayerShell', '1.0')
        except (ImportError, ValueError):
            return f"gtk4-layer-shell not installed. Install: {layer_pkg}"
        return ""
    
    def _ensure_daemon(self):
        """Ensure the daemon process is running."""
        # Check in-memory reference first
        if self._process is not None and self._process.poll() is None:
            return True  # Already running

        # Check PID file for orphaned daemon (from previous crash)
        if MIC_OSD_PID_FILE.exists():
            try:
                pid = int(MIC_OSD_PID_FILE.read_text().strip())
                # Check if process still exists (signal 0 = existence check)
                os.kill(pid, 0)
                print(f"[MIC-OSD] Found orphaned daemon (PID {pid}), reusing it")
                # Create dummy process reference (we can't use wait() on it)
                # The actual daemon PID is tracked in _orphaned_daemon_pid
                self._process = subprocess.Popen(
                    ['true'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                # Note: Cannot override self._process.pid (read-only), so we track it separately
                self._orphaned_daemon_pid = pid  # Track that we're using an orphaned daemon
                return True
            except (ValueError, ProcessLookupError, PermissionError):
                # Stale PID file, clean it up
                print("[MIC-OSD] Cleaning up stale PID file")
                try:
                    MIC_OSD_PID_FILE.unlink()
                except Exception:
                    pass

        # Build the Python code to run
        lib_dir = self._mic_osd_dir.parent
        code = f"""
import signal
# Ignore SIGUSR1/SIGUSR2 during startup so early signals don't kill the
# process (default disposition is SIG_DFL=terminate).  The real handlers
# are installed inside main() before the GLib main-loop starts.
signal.signal(signal.SIGUSR1, signal.SIG_IGN)
signal.signal(signal.SIGUSR2, signal.SIG_IGN)

import sys
sys.path.insert(0, '{lib_dir}')
from mic_osd.main import main
sys.argv = ['mic-osd', '--daemon']
sys.exit(main())
"""

        # Set LD_PRELOAD for gtk4-layer-shell.
        # Search common library paths including lib64 (Fedora/RHEL) and versioned
        # .so files (distros that only ship the unversioned symlink in -devel).
        env = os.environ.copy()
        lib_path = None
        for pattern in [
            '/usr/lib64/libgtk4-layer-shell.so*',
            '/usr/lib/libgtk4-layer-shell.so*',
            '/usr/local/lib64/libgtk4-layer-shell.so*',
            '/usr/local/lib/libgtk4-layer-shell.so*',
        ]:
            for candidate in sorted(glob.glob(pattern)):
                resolved = os.path.realpath(candidate)
                if os.path.isfile(resolved):
                    lib_path = resolved
                    break
            if lib_path:
                break
        if lib_path:
            env['LD_PRELOAD'] = lib_path

        try:
            python_cmd = sys.executable or 'python3'
            self._process = subprocess.Popen(
                [python_cmd, '-c', code],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )

            # Brief startup health check - give daemon time to crash on init errors
            import time
            time.sleep(0.15)
            if self._process.poll() is not None:
                # Daemon died during startup - capture and log the error
                stderr_output = ""
                try:
                    stderr_output = self._process.stderr.read().decode('utf-8', errors='ignore').strip()
                except Exception:
                    pass
                if stderr_output:
                    print(f"[MIC-OSD] Daemon crashed on startup: {stderr_output}", flush=True)
                else:
                    print(f"[MIC-OSD] Daemon crashed on startup (exit code {self._process.returncode})", flush=True)
                self._process = None
                return False

            # Daemon survived startup - detach stderr so it doesn't block
            try:
                self._process.stderr.close()
            except Exception:
                pass

            # Write PID file
            MIC_OSD_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
            MIC_OSD_PID_FILE.write_text(str(self._process.pid))

            # Clear orphaned daemon flag since we created a new daemon
            self._orphaned_daemon_pid = None

            print(f"[MIC-OSD] Daemon started (PID {self._process.pid})", flush=True)
            return True
        except Exception as e:
            print(f"[MIC-OSD] Failed to start daemon: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self._process = None
            return False
    
    def show(self) -> bool:
        """Show the mic-osd overlay (instant via signal)."""
        if not self.is_available():
            return False
        
        if not self._ensure_daemon():
            return False
        
        try:
            # For orphaned daemons, use the tracked PID
            pid = self._orphaned_daemon_pid if self._orphaned_daemon_pid is not None else self._process.pid
            os.kill(pid, signal.SIGUSR1)
            return True
        except (ProcessLookupError, OSError):
            self._process = None
            self._orphaned_daemon_pid = None
            return False
    
    def hide(self):
        """Hide the mic-osd overlay (instant via signal)."""
        self.clear_preview_text()

        if self._process is None:
            return
        
        # For orphaned daemons, check PID directly instead of poll()
        # (poll() returns exit code of dummy process, not the actual daemon)
        if self._orphaned_daemon_pid is not None:
            try:
                # Verify the orphaned daemon PID is still alive
                os.kill(self._orphaned_daemon_pid, 0)
                # PID exists, send hide signal
                os.kill(self._orphaned_daemon_pid, signal.SIGUSR2)
                return
            except (ProcessLookupError, OSError) as e:
                # Orphaned daemon is dead, clean up and log warning
                print(f"[MIC-OSD] Orphaned daemon (PID {self._orphaned_daemon_pid}) is dead, cleaning up: {e}", flush=True)
                self._process = None
                self._orphaned_daemon_pid = None
                # Clean up stale PID file
                if MIC_OSD_PID_FILE.exists():
                    try:
                        MIC_OSD_PID_FILE.unlink()
                    except Exception:
                        pass
                self.clear_preview_text()
                return
        
        # For normal daemons, verify process is actually alive before signaling
        if self._process.poll() is not None:
            # Process is dead, clean up
            print(f"[MIC-OSD] Daemon process (PID {self._process.pid}) is dead, cleaning up", flush=True)
            self._process = None
            self._orphaned_daemon_pid = None
            # Clean up stale PID file
            if MIC_OSD_PID_FILE.exists():
                try:
                    MIC_OSD_PID_FILE.unlink()
                except Exception:
                    pass
            self.clear_preview_text()
            return
        
        # Verify process is actually alive before sending signal
        try:
            os.kill(self._process.pid, 0)
        except (ProcessLookupError, OSError) as e:
            # Process is dead, clean up
            print(f"[MIC-OSD] Daemon process (PID {self._process.pid}) is dead, cleaning up: {e}", flush=True)
            self._process = None
            self._orphaned_daemon_pid = None
            # Clean up stale PID file
            if MIC_OSD_PID_FILE.exists():
                try:
                    MIC_OSD_PID_FILE.unlink()
                except Exception:
                    pass
            self.clear_preview_text()
            return
        
        # Process is alive, send hide signal
        try:
            os.kill(self._process.pid, signal.SIGUSR2)
        except (ProcessLookupError, OSError) as e:
            print(f"[MIC-OSD] Failed to send SIGUSR2 to daemon (PID {self._process.pid}): {e}", flush=True)
            self._process = None
            self._orphaned_daemon_pid = None
            self.clear_preview_text()

    def set_state(self, state: str):
        """
        Set the visualizer state.

        Args:
            state: One of 'recording', 'paused', 'processing', 'error', 'success'
        """
        try:
            VISUALIZER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            VISUALIZER_STATE_FILE.write_text(state)
        except Exception as e:
            print(f"[MIC-OSD] Failed to write visualizer state: {e}", flush=True)

    def clear_state(self):
        """Clear the visualizer state file."""
        try:
            if VISUALIZER_STATE_FILE.exists():
                VISUALIZER_STATE_FILE.unlink()
        except Exception as e:
            print(f"[MIC-OSD] Failed to clear visualizer state: {e}", flush=True)

    def set_preview_text(self, text: str):
        """Set live transcript preview text."""
        text = (text or "").rstrip('\r\n')

        if not text:
            with self._preview_lock:
                self._pending_preview_text = None
                if self._preview_flush_timer:
                    self._preview_flush_timer.cancel()
                    self._preview_flush_timer = None
            self._write_preview_text_file("")
            return

        now = time.monotonic()
        write_now = False
        with self._preview_lock:
            elapsed = now - self._last_preview_write_at
            if elapsed >= self.PREVIEW_WRITE_INTERVAL_SECONDS:
                self._last_preview_write_at = now
                self._pending_preview_text = None
                write_now = True
            else:
                self._pending_preview_text = text
                if self._preview_flush_timer is None:
                    delay = self.PREVIEW_WRITE_INTERVAL_SECONDS - elapsed
                    self._preview_flush_timer = threading.Timer(delay, self._flush_pending_preview_text)
                    self._preview_flush_timer.daemon = True
                    self._preview_flush_timer.start()

        if write_now:
            self._write_preview_text_file(text)

    def _flush_pending_preview_text(self):
        with self._preview_lock:
            text = self._pending_preview_text
            self._pending_preview_text = None
            self._preview_flush_timer = None
            self._last_preview_write_at = time.monotonic()

        if text is not None:
            self._write_preview_text_file(text)

    def _write_preview_text_file(self, text: str):
        """Write live preview text to the runtime IPC file."""
        try:
            TRANSCRIPT_PREVIEW_FILE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                TRANSCRIPT_PREVIEW_FILE.parent.chmod(0o700)
            except Exception:
                pass
            if text:
                fd = os.open(
                    TRANSCRIPT_PREVIEW_FILE,
                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                    0o600,
                )
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(text)
                try:
                    TRANSCRIPT_PREVIEW_FILE.chmod(0o600)
                except Exception:
                    pass
            elif TRANSCRIPT_PREVIEW_FILE.exists():
                TRANSCRIPT_PREVIEW_FILE.unlink()
        except Exception as e:
            print(f"[MIC-OSD] Failed to write transcript preview: {e}", flush=True)

    def clear_preview_text(self):
        """Clear live transcript preview text."""
        try:
            with self._preview_lock:
                self._pending_preview_text = None
                if self._preview_flush_timer:
                    self._preview_flush_timer.cancel()
                    self._preview_flush_timer = None
            if TRANSCRIPT_PREVIEW_FILE.exists():
                TRANSCRIPT_PREVIEW_FILE.unlink()
        except Exception as e:
            print(f"[MIC-OSD] Failed to clear transcript preview: {e}", flush=True)

    def stop(self):
        """Stop the daemon completely."""
        if self._process is None:
            return

        try:
            # For orphaned daemons, use the tracked PID
            pid = self._orphaned_daemon_pid if self._orphaned_daemon_pid is not None else self._process.pid
            os.kill(pid, signal.SIGTERM)
            # Only wait if it's a normal process (not orphaned)
            if self._orphaned_daemon_pid is None:
                self._process.wait(timeout=1.0)
            else:
                # For orphaned daemons, give it a moment to exit
                import time
                time.sleep(0.5)
        except subprocess.TimeoutExpired:
            pid = self._orphaned_daemon_pid if self._orphaned_daemon_pid is not None else self._process.pid
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        finally:
            self._process = None
            self._orphaned_daemon_pid = None
            # Clean up PID file
            if MIC_OSD_PID_FILE.exists():
                try:
                    MIC_OSD_PID_FILE.unlink()
                except Exception:
                    pass
            self.clear_preview_text()
