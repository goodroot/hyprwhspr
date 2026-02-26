"""
TTSOSDRunner - Daemon-based wrapper for tts-osd overlay.

Spawns tts-osd in daemon mode, then uses SIGUSR1/SIGUSR2 to show/hide.
"""

import subprocess
import signal
import os
from pathlib import Path

try:
    from ..src.paths import TTS_OSD_PID_FILE, TTS_OSD_STATE_FILE
except ImportError:
    from src.paths import TTS_OSD_PID_FILE, TTS_OSD_STATE_FILE


class TTSOSDRunner:
    """Daemon-based runner for the tts-osd overlay."""

    def __init__(self):
        self._process = None
        self._tts_osd_dir = Path(__file__).parent
        self._orphaned_daemon_pid = None

    @staticmethod
    def is_available() -> bool:
        """Check if tts-osd can run."""
        try:
            import gi
            gi.require_version('Gtk', '4.0')
            gi.require_version('Gtk4LayerShell', '1.0')
            return True
        except (ImportError, ValueError):
            return False

    @staticmethod
    def get_unavailable_reason() -> str:
        """Get reason why tts-osd is unavailable."""
        try:
            import gi
            gi.require_version('Gtk', '4.0')
        except (ImportError, ValueError):
            return "GTK4 bindings not installed. Install: python-gobject gtk4"
        try:
            gi.require_version('Gtk4LayerShell', '1.0')
        except (ImportError, ValueError):
            return "gtk4-layer-shell not installed. Install: gtk4-layer-shell"
        return ""

    def _ensure_daemon(self):
        """Ensure the daemon process is running."""
        if self._process is not None and self._process.poll() is None:
            return True

        if TTS_OSD_PID_FILE.exists():
            try:
                pid = int(TTS_OSD_PID_FILE.read_text().strip())
                os.kill(pid, 0)
                self._process = subprocess.Popen(
                    ['true'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                self._orphaned_daemon_pid = pid
                return True
            except (ValueError, ProcessLookupError, PermissionError):
                try:
                    TTS_OSD_PID_FILE.unlink()
                except Exception:
                    pass

        lib_dir = self._tts_osd_dir.parent
        code = f"""
import sys
sys.path.insert(0, '{lib_dir}')
from tts_osd.main import main
sys.argv = ['tts-osd', '--daemon']
sys.exit(main())
"""

        env = os.environ.copy()
        lib_path = '/usr/lib/libgtk4-layer-shell.so'
        if os.path.exists(lib_path):
            if os.path.islink(lib_path):
                lib_path = os.path.realpath(lib_path)
            env['LD_PRELOAD'] = lib_path

        try:
            self._process = subprocess.Popen(
                ['/usr/bin/python3', '-c', code],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )

            import time
            time.sleep(0.15)
            if self._process.poll() is not None:
                stderr_output = ""
                try:
                    stderr_output = self._process.stderr.read().decode('utf-8', errors='ignore').strip()
                except Exception:
                    pass
                if stderr_output:
                    print(f"[TTS-OSD] Daemon crashed: {stderr_output}", flush=True)
                self._process = None
                return False

            try:
                self._process.stderr.close()
            except Exception:
                pass

            TTS_OSD_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
            TTS_OSD_PID_FILE.write_text(str(self._process.pid))
            self._orphaned_daemon_pid = None
            return True
        except Exception as e:
            print(f"[TTS-OSD] Failed to start daemon: {e}", flush=True)
            self._process = None
            return False

    def show(self) -> bool:
        """Show the tts-osd overlay."""
        if not self.is_available():
            return False
        if not self._ensure_daemon():
            return False
        try:
            pid = self._orphaned_daemon_pid if self._orphaned_daemon_pid is not None else self._process.pid
            os.kill(pid, signal.SIGUSR1)
            return True
        except (ProcessLookupError, OSError):
            self._process = None
            self._orphaned_daemon_pid = None
            return False

    def hide(self):
        """Hide the tts-osd overlay."""
        if self._process is None:
            return

        if self._orphaned_daemon_pid is not None:
            try:
                os.kill(self._orphaned_daemon_pid, 0)
                os.kill(self._orphaned_daemon_pid, signal.SIGUSR2)
                return
            except (ProcessLookupError, OSError):
                self._process = None
                self._orphaned_daemon_pid = None
                if TTS_OSD_PID_FILE.exists():
                    try:
                        TTS_OSD_PID_FILE.unlink()
                    except Exception:
                        pass
                return

        if self._process.poll() is not None:
            self._process = None
            self._orphaned_daemon_pid = None
            if TTS_OSD_PID_FILE.exists():
                try:
                    TTS_OSD_PID_FILE.unlink()
                except Exception:
                    pass
            return

        try:
            os.kill(self._process.pid, signal.SIGUSR2)
        except (ProcessLookupError, OSError):
            self._process = None
            self._orphaned_daemon_pid = None

    def set_state(self, state: str):
        """Set the visualizer state (generating, speaking, success, error)."""
        try:
            TTS_OSD_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            TTS_OSD_STATE_FILE.write_text(state)
        except Exception as e:
            print(f"[TTS-OSD] Failed to write state: {e}", flush=True)

    def clear_state(self):
        """Clear the state file."""
        try:
            if TTS_OSD_STATE_FILE.exists():
                TTS_OSD_STATE_FILE.unlink()
        except Exception as e:
            print(f"[TTS-OSD] Failed to clear state: {e}", flush=True)

    def stop(self):
        """Stop the daemon completely."""
        if self._process is None:
            return
        try:
            pid = self._orphaned_daemon_pid if self._orphaned_daemon_pid is not None else self._process.pid
            os.kill(pid, signal.SIGTERM)
            if self._orphaned_daemon_pid is None:
                self._process.wait(timeout=1.0)
            else:
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
            if TTS_OSD_PID_FILE.exists():
                try:
                    TTS_OSD_PID_FILE.unlink()
                except Exception:
                    pass
