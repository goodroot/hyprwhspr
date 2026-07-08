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
    from ..src.paths import (
        MIC_OSD_PID_FILE, VISUALIZER_STATE_FILE, TRANSCRIPT_PREVIEW_FILE,
        MIC_OSD_LEVEL_FEED_FILE,
    )
except ImportError:
    # Fallback for direct execution
    from src.paths import (
        MIC_OSD_PID_FILE, VISUALIZER_STATE_FILE, TRANSCRIPT_PREVIEW_FILE,
        MIC_OSD_LEVEL_FEED_FILE,
    )


class MicOSDRunner:
    """
    Daemon-based runner for the mic-osd overlay.
    
    Spawns mic-osd in daemon mode at init, then signals it to show/hide.
    """

    PREVIEW_WRITE_INTERVAL_SECONDS = 0.05
    LEVEL_FEED_INTERVAL_SECONDS = 1.0 / 30

    def __init__(self, level_source=None):
        """
        Args:
            level_source: Optional callable returning (level, bucket_rms_list)
                or None (e.g. AudioCapture.get_viz_frame). When set, show()
                streams frames to a runtime file the OSD daemon reads instead
                of opening its own audio stream.
        """
        self._process = None
        self._mic_osd_dir = Path(__file__).parent
        self._orphaned_daemon_pid = None  # Track PID when reusing orphaned daemon
        self._preview_lock = threading.Lock()
        self._last_preview_write_at = 0.0
        self._pending_preview_text = None
        self._preview_flush_timer = None
        self._preview_generation = 0
        self._level_source = level_source
        self._level_feed_thread = None
        self._level_feed_stop = threading.Event()
        self._level_feed_lock = threading.Lock()
        self._last_level_feed_error_at = 0.0
    
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
    def _layer_shell_ld_preload() -> str:
        """Resolve the gtk4-layer-shell .so to LD_PRELOAD (same search as the daemon)."""
        for pattern in [
            '/usr/lib64/libgtk4-layer-shell.so*',
            '/usr/lib/libgtk4-layer-shell.so*',
            '/usr/lib/*/libgtk4-layer-shell.so*',
            '/usr/local/lib64/libgtk4-layer-shell.so*',
            '/usr/local/lib/libgtk4-layer-shell.so*',
        ]:
            for candidate in sorted(glob.glob(pattern)):
                resolved = os.path.realpath(candidate)
                if os.path.isfile(resolved):
                    return resolved
        return ""

    @staticmethod
    def layer_shell_active() -> bool:
        """Return True only if the running compositor supports the layer-shell
        protocol (Hyprland, Niri, Sway, ...). False on GNOME/Mutter and on any
        uncertainty, so callers safely fall back to a non-overlay status display.

        Probed in a subprocess to keep GTK out of the main process.
        """
        probe = (
            "import gi; gi.require_version('Gtk', '4.0');"
            "gi.require_version('Gtk4LayerShell', '1.0');"
            "from gi.repository import Gtk, Gtk4LayerShell;"
            "Gtk.init();"
            "print('1' if Gtk4LayerShell.is_supported() else '0')"
        )
        env = os.environ.copy()
        preload = MicOSDRunner._layer_shell_ld_preload()
        if preload:
            env['LD_PRELOAD'] = preload
        try:
            result = subprocess.run(
                [sys.executable or 'python3', '-c', probe],
                env=env, capture_output=True, timeout=5,
            )
            return result.stdout.decode('utf-8', 'ignore').strip() == '1'
        except Exception:
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
                if not self._is_mic_osd_daemon_pid(pid):
                    raise ProcessLookupError(f"PID {pid} is not a mic-osd daemon")
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
            '/usr/lib/*/libgtk4-layer-shell.so*',
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
        env['HYPRWHSPR_MIC_OSD_DAEMON'] = '1'

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

    @staticmethod
    def _is_mic_osd_daemon_pid(pid: int) -> bool:
        """Return True only if pid appears to be this project's mic-osd daemon."""
        if pid <= 0:
            return False

        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError, OSError):
            return False

        proc_path = Path('/proc') / str(pid)
        environ_path = proc_path / 'environ'
        try:
            environ = environ_path.read_bytes().split(b'\x00')
            if b'HYPRWHSPR_MIC_OSD_DAEMON=1' in environ:
                return True
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            pass

        cmdline_path = proc_path / 'cmdline'
        try:
            raw_cmdline = cmdline_path.read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            return False

        cmdline = raw_cmdline.replace(b'\x00', b' ').decode('utf-8', errors='ignore')
        if '--daemon' not in cmdline:
            return False

        return (
            'mic_osd.main' in cmdline
            or 'mic-osd' in cmdline
            or 'com.hyprwhspr.mic-osd' in cmdline
        )

    def _signal_daemon(self, sig: signal.Signals) -> bool:
        """Signal the tracked daemon after validating orphaned PID-file reuse."""
        pid = self._orphaned_daemon_pid if self._orphaned_daemon_pid is not None else self._process.pid
        if self._orphaned_daemon_pid is not None and not self._is_mic_osd_daemon_pid(pid):
            print(f"[MIC-OSD] Refusing to signal non mic-osd PID {pid}", flush=True)
            self._process = None
            self._orphaned_daemon_pid = None
            self._unlink_pid_file()
            return False

        os.kill(pid, sig)
        return True

    def _unlink_pid_file(self):
        try:
            if MIC_OSD_PID_FILE.exists():
                MIC_OSD_PID_FILE.unlink()
        except Exception:
            pass
    
    def _start_level_feed(self):
        """Stream capture levels to the OSD's runtime feed file. Writes the
        first frame synchronously so it's fresh before the daemon's SIGUSR1
        handler picks feed vs. fallback mode.

        Serialized with _stop_level_feed: show() and a previous recording's
        delayed-hide run on different threads, and an interleave there could
        drop the feed for the new recording (reintroducing #205) or orphan a
        thread that double-writes the file.
        """
        with self._level_feed_lock:
            if self._level_source is None or self._level_feed_thread is not None:
                return
            self._level_feed_stop.clear()
            self._write_level_feed_frame()

            def _feed_loop():
                while not self._level_feed_stop.wait(self.LEVEL_FEED_INTERVAL_SECONDS):
                    self._write_level_feed_frame()

            self._level_feed_thread = threading.Thread(target=_feed_loop, daemon=True)
            self._level_feed_thread.start()

    def _stop_level_feed(self):
        with self._level_feed_lock:
            self._level_feed_stop.set()
            thread = self._level_feed_thread
            self._level_feed_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)
        try:
            if MIC_OSD_LEVEL_FEED_FILE.exists():
                MIC_OSD_LEVEL_FEED_FILE.unlink()
        except Exception:
            pass

    def _write_level_feed_frame(self):
        frame = None
        try:
            frame = self._level_source()
        except Exception:
            pass
        # A zero frame keeps the file fresh so the OSD stays in feed mode.
        level, buckets = frame if frame is not None else (0.0, [])
        payload = ' '.join(f"{value:.6f}" for value in [level, *buckets])
        try:
            self._atomic_write_runtime_file(MIC_OSD_LEVEL_FEED_FILE, payload)
        except Exception as e:
            # Transient failures (ENOSPC, replace race) must not kill the feed;
            # a dropped frame just goes stale for one tick. Throttle the log.
            now = time.monotonic()
            if now - self._last_level_feed_error_at > 5.0:
                self._last_level_feed_error_at = now
                print(f"[MIC-OSD] Failed to write level feed: {e}", flush=True)

    def show(self) -> bool:
        """Show the mic-osd overlay (instant via signal)."""
        if not self.is_available():
            return False

        if not self._ensure_daemon():
            return False

        self._start_level_feed()  # before signaling; first frame must be ready

        try:
            return self._signal_daemon(signal.SIGUSR1)
        except (ProcessLookupError, OSError):
            self._stop_level_feed()
            self._process = None
            self._orphaned_daemon_pid = None
            return False

    def hide(self):
        """Hide the mic-osd overlay (instant via signal)."""
        self._stop_level_feed()
        self.clear_preview_text()

        if self._process is None:
            return
        
        # For orphaned daemons, check PID directly instead of poll()
        # (poll() returns exit code of dummy process, not the actual daemon)
        if self._orphaned_daemon_pid is not None:
            try:
                if not self._signal_daemon(signal.SIGUSR2):
                    return
                return
            except (ProcessLookupError, OSError) as e:
                # Orphaned daemon is dead, clean up and log warning
                print(f"[MIC-OSD] Orphaned daemon (PID {self._orphaned_daemon_pid}) is dead, cleaning up: {e}", flush=True)
                self._process = None
                self._orphaned_daemon_pid = None
                # Clean up stale PID file
                self._unlink_pid_file()
                self.clear_preview_text()
                return
        
        # For normal daemons, verify process is actually alive before signaling
        if self._process.poll() is not None:
            # Process is dead, clean up
            print(f"[MIC-OSD] Daemon process (PID {self._process.pid}) is dead, cleaning up", flush=True)
            self._process = None
            self._orphaned_daemon_pid = None
            # Clean up stale PID file
            self._unlink_pid_file()
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
            self._unlink_pid_file()
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

        with self._preview_lock:
            if not text:
                self._cancel_pending_preview_flush()
                self._preview_generation += 1
                self._write_preview_text_file("")
                return

            now = time.monotonic()
            elapsed = now - self._last_preview_write_at
            if elapsed >= self.PREVIEW_WRITE_INTERVAL_SECONDS:
                self._cancel_pending_preview_flush()
                self._last_preview_write_at = now
                self._write_preview_text_file(text)
                return

            self._pending_preview_text = text
            if self._preview_flush_timer is None:
                generation = self._preview_generation
                delay = self.PREVIEW_WRITE_INTERVAL_SECONDS - elapsed
                self._preview_flush_timer = threading.Timer(
                    delay,
                    self._flush_pending_preview_text,
                    args=(generation,),
                )
                self._preview_flush_timer.daemon = True
                self._preview_flush_timer.start()

    def _cancel_pending_preview_flush(self):
        self._pending_preview_text = None
        if self._preview_flush_timer:
            self._preview_flush_timer.cancel()
            self._preview_flush_timer = None

    def _flush_pending_preview_text(self, generation: int):
        with self._preview_lock:
            if generation != self._preview_generation:
                return

            text = self._pending_preview_text
            self._pending_preview_text = None
            self._preview_flush_timer = None

            if text is None:
                return

            self._last_preview_write_at = time.monotonic()
            self._write_preview_text_file(text)

    @staticmethod
    def _atomic_write_runtime_file(path, text: str):
        """Atomically write non-empty text to a 0600 runtime file (temp in the
        same dir + os.replace, so readers never see a torn frame). The temp
        name is per-pid-and-thread so concurrent writers don't collide."""
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            path.parent.chmod(0o700)
        except Exception:
            pass
        temp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(text)
            os.replace(temp_path, path)
        finally:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
        try:
            path.chmod(0o600)
        except Exception:
            pass

    def _write_preview_text_file(self, text: str):
        """Write live preview text to the runtime IPC file."""
        try:
            if text:
                self._atomic_write_runtime_file(TRANSCRIPT_PREVIEW_FILE, text)
            elif TRANSCRIPT_PREVIEW_FILE.exists():
                TRANSCRIPT_PREVIEW_FILE.unlink()
        except Exception as e:
            print(f"[MIC-OSD] Failed to write transcript preview: {e}", flush=True)

    def clear_preview_text(self):
        """Clear live transcript preview text."""
        try:
            with self._preview_lock:
                self._cancel_pending_preview_flush()
                self._preview_generation += 1
                if TRANSCRIPT_PREVIEW_FILE.exists():
                    TRANSCRIPT_PREVIEW_FILE.unlink()
        except Exception as e:
            print(f"[MIC-OSD] Failed to clear transcript preview: {e}", flush=True)

    def stop(self):
        """Stop the daemon completely."""
        self._stop_level_feed()
        if self._process is None:
            return

        try:
            # For orphaned daemons, use the tracked PID
            if not self._signal_daemon(signal.SIGTERM):
                return
            # Only wait if it's a normal process (not orphaned)
            if self._orphaned_daemon_pid is None:
                self._process.wait(timeout=1.0)
            else:
                # For orphaned daemons, give it a moment to exit
                import time
                time.sleep(0.5)
        except subprocess.TimeoutExpired:
            pid = self._orphaned_daemon_pid if self._orphaned_daemon_pid is not None else self._process.pid
            if self._orphaned_daemon_pid is None or self._is_mic_osd_daemon_pid(pid):
                os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        finally:
            self._process = None
            self._orphaned_daemon_pid = None
            # Clean up PID file
            self._unlink_pid_file()
            self.clear_preview_text()
