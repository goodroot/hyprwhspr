"""Private ydotoold instance owned by hyprwhspr.

hyprwhspr's paste fallback on GNOME/Mutter uses ydotool, whose client talks to the
``ydotoold`` daemon over a socket. Instead of relying on a *shared* system ydotoold
(and a hyprwhspr-managed systemd unit), we run our **own** ydotoold child on a
*private* socket and point the client at it via ``YDOTOOL_SOCKET``. That removes the
managed unit and stops hyprwhspr from touching a daemon other tools also use.

The daemon is started lazily (on first uinput-fallback use, so wlroots/wtype sessions
never spawn it), restarted once if it dies, and torn down on shutdown. ydotoold's
device is named ``ydotoold virtual device`` — the existing virtual-keyboard filters
(``_VIRTUAL_KEYBOARD_TOKENS``) already exclude it, so the hotkey listener won't grab it.
"""

import os
import shutil
import subprocess
import threading
import time
from typing import Callable, Optional


class YdotooldSession:
    """Manages one private ``ydotoold`` process and its socket.

    Thread-safe: a single re-entrant lock guards process state, so the clipboard
    restore thread and the main inject path cannot race on (re)starts.
    """

    def __init__(self, socket_path: Optional[str] = None,
                 spawn: Optional[Callable[[], "subprocess.Popen"]] = None,
                 socket_timeout: float = 3.0, poll_interval: float = 0.05):
        runtime = os.environ.get('XDG_RUNTIME_DIR') or '/tmp'
        # Private path (not the default $XDG_RUNTIME_DIR/.ydotool_socket) so we never
        # collide with a shared/system ydotoold. The path also appears in our
        # ydotoold cmdline, giving a unique `pkill -f` pattern for safety nets.
        self.socket_path = socket_path or os.path.join(runtime, 'hyprwhspr-ydotool.sock')
        self._spawn = spawn or self._default_spawn
        self._socket_timeout = socket_timeout
        self._poll_interval = poll_interval
        self._socket_env = dict(os.environ)
        self._socket_env['YDOTOOL_SOCKET'] = self.socket_path
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ spawn
    def _default_spawn(self) -> "subprocess.Popen":
        return subprocess.Popen(
            ['ydotoold', '-p', self.socket_path, '-P', '0600'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def is_available() -> bool:
        """True when both the daemon and client binaries are present."""
        return shutil.which('ydotoold') is not None and shutil.which('ydotool') is not None

    def is_running(self) -> bool:
        """True if our process is alive. Never spawns (safe to call on wlroots)."""
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def ensure_running(self) -> bool:
        """Start the daemon if needed and wait for its socket.

        Returns True when the private socket is ready to use. Restarts once if the
        process is dead; returns False on failure so the caller can degrade to
        clipboard-only.
        """
        with self._lock:
            if self.is_running() and os.path.exists(self.socket_path):
                return True
            deadline = time.monotonic() + self._socket_timeout
            return self._start_locked(deadline, allow_restart=True)

    def _start_locked(self, deadline: float, allow_restart: bool) -> bool:
        # Drop a dead handle.
        if self._proc is not None and self._proc.poll() is not None:
            self._proc = None

        if self._proc is None:
            # Remove a stale socket from a previous run so ydotoold can bind.
            try:
                if os.path.exists(self.socket_path):
                    os.unlink(self.socket_path)
            except OSError:
                pass
            try:
                self._proc = self._spawn()
            except (OSError, ValueError):
                return False

        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                # Died during startup — retry once from scratch.
                self._proc = None
                return self._start_locked(deadline, allow_restart=False) if allow_restart else False
            if os.path.exists(self.socket_path):
                return True
            time.sleep(self._poll_interval)
        return False

    def socket_env(self) -> dict:
        """Environment for ydotool client calls, pointing at our private socket."""
        return self._socket_env

    def close(self) -> None:
        """Terminate our daemon and remove the socket. Idempotent; never raises."""
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is not None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass
        try:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
        except OSError:
            pass
