"""
MicOSD Runner - Daemon-based wrapper for instant mic-osd overlay.

Spawns mic-osd once in daemon mode, then uses SIGUSR1/SIGUSR2 to show/hide.
This eliminates subprocess spawn latency on each recording.
"""

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

from . import runtime
from .runtime import LayerShellResolutionError


class MicOSDRunner:
    """
    Daemon-based runner for the mic-osd overlay.
    
    Spawns mic-osd in daemon mode at init, then signals it to show/hide.
    """

    PREVIEW_WRITE_INTERVAL_SECONDS = 0.05
    LEVEL_FEED_INTERVAL_SECONDS = 1.0 / 30
    OSD_STYLES = ('waveform', 'vu_meter', 'pill')

    def __init__(self, level_source=None, style='waveform', assessment=None):
        """
        Args:
            level_source: Optional callable returning (level, bucket_rms_list)
                or None (e.g. AudioCapture.get_viz_frame). When set, show()
                streams frames to a runtime file the OSD daemon reads instead
                of opening its own audio stream.
            style: Visualization style for the daemon; unknown values fall
                back to 'waveform'.
            assessment: The caller's OverlayAssessment; assessed on first use if omitted.
        """
        self._style = style if style in self.OSD_STYLES else 'waveform'
        self._process = None
        self._mic_osd_dir = Path(__file__).parent
        self._orphaned_daemon_pid = None  # Track PID when reusing orphaned daemon
        self._preview_lock = threading.Lock()
        self._last_preview_write_at = None
        self._pending_preview_text = None
        self._preview_flush_timer = None
        self._preview_generation = 0
        self._level_source = level_source
        self._level_feed_thread = None
        self._level_feed_stop = threading.Event()
        self._level_feed_lock = threading.RLock()
        self._last_level_feed_error_at = 0.0
        self._visibility_condition = threading.Condition()
        self._hide_generation = 0
        self._hide_teardowns = 0
        self._assessment = assessment
        self._assessment_lock = threading.Lock()
    
    @property
    def assessment(self):
        """The overlay assessment this runner starts daemons from; assessed once."""
        with self._assessment_lock:
            if self._assessment is None:
                self._assessment = runtime.assess_overlay()
            return self._assessment

    def is_available(self) -> bool:
        """An overlay backend was selected for this session."""
        return self.assessment.backend is not None

    def _ensure_daemon(self) -> bool:
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
sys.argv = ['mic-osd', '--daemon', '--viz', '{self._style}']
sys.exit(main())
"""

        assessment = self.assessment
        if assessment.backend is None:
            print(f"[MIC-OSD] Not starting daemon: {assessment.reason}", flush=True)
            return False
        try:
            env = runtime.child_environment(assessment)
        except LayerShellResolutionError as e:
            print(f"[MIC-OSD] Not starting daemon: {e}", flush=True)
            return False
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
        with self._visibility_condition:
            while self._hide_teardowns:
                self._visibility_condition.wait()
            generation = self._hide_generation

        if not self.is_available():
            return False

        if not self._ensure_daemon():
            return False

        try:
            with self._level_feed_lock:
                with self._visibility_condition:
                    if generation != self._hide_generation:
                        return False
                self._start_level_feed()  # first frame must be ready before SIGUSR1
                with self._visibility_condition:
                    # A hide during the first frame cancels this show. Keep
                    # teardown serialized until we have decided whether to signal.
                    if generation != self._hide_generation:
                        return False
                    return self._signal_daemon(signal.SIGUSR1)
        except (ProcessLookupError, OSError):
            self._stop_level_feed()
            self._process = None
            self._orphaned_daemon_pid = None
            return False

    def hide(self):
        """Hide the mic-osd overlay (instant via signal)."""
        # Do not wait for a blocked level source to hide the window. The
        # generation check prevents an in-flight show from undoing this signal,
        # while the teardown counter keeps a newer show from overtaking cleanup.
        try:
            with self._visibility_condition:
                self._hide_generation += 1
                self._hide_teardowns += 1
                self._signal_hide()
        finally:
            try:
                self._stop_level_feed()
                self.clear_preview_text()
            finally:
                with self._visibility_condition:
                    self._hide_teardowns -= 1
                    self._visibility_condition.notify_all()

    def _signal_hide(self):
        """Send SIGUSR2 to the daemon, dropping references to a dead one."""
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
                return
        
        # For normal daemons, verify process is actually alive before signaling
        if self._process.poll() is not None:
            # Process is dead, clean up
            print(f"[MIC-OSD] Daemon process (PID {self._process.pid}) is dead, cleaning up", flush=True)
            self._process = None
            self._orphaned_daemon_pid = None
            # Clean up stale PID file
            self._unlink_pid_file()
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
            return
        
        # Process is alive, send hide signal
        try:
            os.kill(self._process.pid, signal.SIGUSR2)
        except (ProcessLookupError, OSError) as e:
            print(f"[MIC-OSD] Failed to send SIGUSR2 to daemon (PID {self._process.pid}): {e}", flush=True)
            self._process = None
            self._orphaned_daemon_pid = None

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
            # No baseline yet means nothing has been written, so the first
            # frame goes out immediately. monotonic() is an arbitrary epoch
            # (uptime on Linux), so it can never be compared against a fixed
            # sentinel like 0.0 to make that call.
            last = self._last_preview_write_at
            elapsed = None if last is None else now - last
            if elapsed is None or elapsed >= self.PREVIEW_WRITE_INTERVAL_SECONDS:
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
