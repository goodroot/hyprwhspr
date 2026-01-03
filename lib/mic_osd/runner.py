"""
MicOSD Runner - Daemon-based wrapper for instant mic-osd overlay.

Spawns mic-osd once in daemon mode, then uses SIGUSR1/SIGUSR2 to show/hide.
This eliminates subprocess spawn latency on each recording.
"""

import subprocess
import signal
import sys
import os
from pathlib import Path


class MicOSDRunner:
    """
    Daemon-based runner for the mic-osd overlay.
    
    Spawns mic-osd in daemon mode at init, then signals it to show/hide.
    """
    
    def __init__(self):
        self._process = None
        self._mic_osd_dir = Path(__file__).parent
    
    @staticmethod
    def is_available() -> bool:
        """Check if mic-osd can run."""
        try:
            import gi
            gi.require_version('Gtk', '4.0')
            gi.require_version('Gtk4LayerShell', '1.0')
            return True
        except (ImportError, ValueError):
            return False
    
    @staticmethod
    def get_unavailable_reason() -> str:
        """Get reason why mic-osd is unavailable."""
        try:
            import gi
            gi.require_version('Gtk', '4.0')
        except (ImportError, ValueError):
            return "GTK4 (python-gobject) is not installed"
        try:
            gi.require_version('Gtk4LayerShell', '1.0')
        except (ImportError, ValueError):
            return "gtk4-layer-shell is not installed"
        return ""
    
    def _ensure_daemon(self):
        """Ensure the daemon process is running."""
        if self._process is not None and self._process.poll() is None:
            return True  # Already running
        
        # Build the Python code to run
        lib_dir = self._mic_osd_dir.parent
        code = f"""
import sys
sys.path.insert(0, '{lib_dir}')
from mic_osd.main import main
sys.argv = ['mic-osd', '--daemon']
sys.exit(main())
"""
        
        # Set LD_PRELOAD for gtk4-layer-shell
        env = os.environ.copy()
        env['LD_PRELOAD'] = '/usr/lib/libgtk4-layer-shell.so'
        
        try:
            self._process = subprocess.Popen(
                ['/usr/bin/python3', '-c', code],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except Exception:
            self._process = None
            return False
    
    def show(self) -> bool:
        """Show the mic-osd overlay (instant via signal)."""
        if not self.is_available():
            return False
        
        if not self._ensure_daemon():
            return False
        
        try:
            os.kill(self._process.pid, signal.SIGUSR1)
            return True
        except (ProcessLookupError, OSError):
            self._process = None
            return False
    
    def hide(self):
        """Hide the mic-osd overlay (instant via signal)."""
        if self._process is None or self._process.poll() is not None:
            return
        
        try:
            os.kill(self._process.pid, signal.SIGUSR2)
        except (ProcessLookupError, OSError):
            self._process = None
    
    def stop(self):
        """Stop the daemon completely."""
        if self._process is None:
            return
        
        try:
            os.kill(self._process.pid, signal.SIGTERM)
            self._process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            os.kill(self._process.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        finally:
            self._process = None
