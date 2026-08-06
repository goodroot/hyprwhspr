"""
mic-osd - A minimal audio visualization OSD for Wayland/Hyprland.

Shows a real-time microphone input visualization overlay.
Supports two modes:
- Standalone: runs until killed (SIGTERM/SIGINT)
- Daemon: stays running, shows on SIGUSR1, hides on SIGUSR2
"""

import json
import sys
import signal
import os
from pathlib import Path

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, GLib


def is_gnome():
    desktop = os.environ.get('XDG_CURRENT_DESKTOP', '').lower()
    return 'gnome' in desktop

_MIC_OSD_IMPORT_ERROR = None
try:
    from .window import OSDWindow, load_css
    from .audio import AudioMonitor, FeedLevelSource
    from .visualizations import VISUALIZATIONS
    from .theme import ThemeWatcher
    from .osd_config import MicOSDConfig
except ImportError as e:
    _MIC_OSD_IMPORT_ERROR = e
    OSDWindow = None
    AudioMonitor = None
    FeedLevelSource = None
    VISUALIZATIONS = {}
    ThemeWatcher = None
    MicOSDConfig = None

# Import paths with fallback for daemon context
try:
    from ..src.paths import (
        CONFIG_FILE, RECORDING_STATUS_FILE, VISUALIZER_STATE_FILE,
        TRANSCRIPT_PREVIEW_FILE, MIC_OSD_LEVEL_FEED_FILE,
        MIC_OSD_VISIBLE_FILE,
    )
except ImportError:
    try:
        from src.paths import (
            CONFIG_FILE, RECORDING_STATUS_FILE, VISUALIZER_STATE_FILE,
            TRANSCRIPT_PREVIEW_FILE, MIC_OSD_LEVEL_FEED_FILE,
            MIC_OSD_VISIBLE_FILE,
        )
    except ImportError:
        # Fallback: construct paths manually if imports fail
        import tempfile
        xdg_runtime = os.environ.get('XDG_RUNTIME_DIR')
        if xdg_runtime:
            runtime_dir = Path(xdg_runtime) / 'hyprwhspr'
        else:
            runtime_dir = Path(tempfile.gettempdir()) / f"hyprwhspr-{os.getuid()}"
        config_home = Path(
            os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')
        )
        CONFIG_FILE = config_home / 'hyprwhspr' / 'config.json'
        RECORDING_STATUS_FILE = runtime_dir / 'recording_status'
        VISUALIZER_STATE_FILE = runtime_dir / 'visualizer_state'
        TRANSCRIPT_PREVIEW_FILE = runtime_dir / 'transcript_preview'
        MIC_OSD_LEVEL_FEED_FILE = runtime_dir / 'mic_osd_level_feed'
        MIC_OSD_VISIBLE_FILE = runtime_dir / 'mic_osd_visible'


class MicOSD:
    """
    Mic-osd application with show/hide support.
    """
    
    def __init__(self, visualization="waveform", width=None, height=None, daemon=False):
        self.main_loop = None
        self.app = None
        self.audio_monitor = None
        self.window = None
        self.update_timer_id = None
        self._auto_hide_timeout_id = None
        self._state_poll_timer_id = None
        self._config_poll_timer_id = None
        self._config_signature = None
        self._last_visualizer_state = None
        self._last_preview_text = None
        self.daemon = daemon
        self.visible = False
        self.theme_watcher = None
        self._should_stop = False

        # Load presentation settings in the OSD process.  They are intentionally
        # separate from the high-rate audio feed and take effect on daemon restart.
        self.osd_config = MicOSDConfig.load()

        # Get visualization
        viz_class = VISUALIZATIONS.get(visualization, VISUALIZATIONS["waveform"])
        if visualization == "spectrum":
            self.visualization = viz_class(osd_config=self.osd_config)
        else:
            self.visualization = viz_class()
        self.width = (
            width
            if width is not None
            else getattr(self.visualization, 'preferred_width', 400)
        )
        self.height = (
            height
            if height is not None
            else getattr(self.visualization, 'preferred_height', 68)
        )

    def run(self):
        """Start the OSD and run until killed."""
        if is_gnome():
            self._run_with_gtk_application()
        else:
            self._run_with_main_loop()

    def _run_with_gtk_application(self):
        self.app = Gtk.Application(application_id="com.hyprwhspr.mic-osd")
        self.app.connect('activate', self._gtk_on_activate)
        self.app.connect('shutdown', lambda _: self._cleanup())
        # Check if stop was requested before running (unlikely but possible)
        if self._should_stop:
            self._cleanup()
            return
        try:
            self.app.run(None)
        except KeyboardInterrupt:
            pass
        finally:
            # Ensure cleanup happens even if exception occurs
            # (shutdown signal may not be emitted on exception)
            self._cleanup()

    def _gtk_on_activate(self, app):
        # Clean up existing resources if activation happens multiple times
        if self.window:
            # Stop timers and audio monitoring before removing window
            if self.update_timer_id:
                GLib.source_remove(self.update_timer_id)
                self.update_timer_id = None
            if self._state_poll_timer_id:
                GLib.source_remove(self._state_poll_timer_id)
                self._state_poll_timer_id = None
            if self._config_poll_timer_id:
                GLib.source_remove(self._config_poll_timer_id)
                self._config_poll_timer_id = None
            if self._auto_hide_timeout_id:
                GLib.source_remove(self._auto_hide_timeout_id)
                self._auto_hide_timeout_id = None
            if self.audio_monitor:
                self.audio_monitor.stop()
                self.audio_monitor = None
            app.remove_window(self.window)
            self.window = None
        
        if self.theme_watcher:
            self.theme_watcher.stop()
            self.theme_watcher = None
        
        load_css()

        self.window = OSDWindow(
            self.visualization,
            self.width,
            self.height,
            osd_config=self.osd_config,
        )
        app.add_window(self.window)

        self.theme_watcher = ThemeWatcher(on_theme_changed=self._on_theme_changed)
        self.theme_watcher.start()

        self._initial_visibility()

    def _run_with_main_loop(self):
        # Initialize GTK
        Gtk.init()

        # Load CSS
        load_css()

        # Create window (hidden in daemon mode)
        self.window = OSDWindow(
            self.visualization,
            self.width,
            self.height,
            osd_config=self.osd_config,
        )

        # Start theme watcher for live theme updates
        self.theme_watcher = ThemeWatcher(on_theme_changed=self._on_theme_changed)
        self.theme_watcher.start()

        self._initial_visibility()

        # Check if stop was requested before main loop was created
        if self._should_stop:
            return

        # Create main loop
        self.main_loop = GLib.MainLoop()

        try:
            self.main_loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()

    def _initial_visibility(self):
        self._start_config_watcher()

        # If a signal handler already set visibility before window creation,
        # respect that state (handles race condition with early signals)
        if self.visible:
            # Signal handler wants to show it
            self._show()
        elif self.daemon:
            # Start hidden, wait for SIGUSR1
            self.window.set_visible(False)
        else:
            # Show immediately
            self._show()

    @staticmethod
    def _get_config_signature():
        try:
            stat = CONFIG_FILE.stat()
            return stat.st_mtime_ns, stat.st_size
        except OSError:
            return None

    def _start_config_watcher(self):
        self._config_signature = self._get_config_signature()
        if not self._config_poll_timer_id:
            self._config_poll_timer_id = GLib.timeout_add(
                400,
                self._poll_config_file,
            )

    def _poll_config_file(self):
        """Hot-reload OSD-only settings after a valid config file change."""
        signature = self._get_config_signature()
        if signature is None or signature == self._config_signature:
            return True

        # Editors that write in place can expose partial JSON for a moment.
        # Keep the previous presentation and retry until the file is valid.
        try:
            json.loads(CONFIG_FILE.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return True

        new_config = MicOSDConfig.load()
        self._config_signature = signature
        if new_config == self.osd_config:
            return True

        self.osd_config = new_config
        apply_visualization = getattr(self.visualization, 'apply_config', None)
        if callable(apply_visualization):
            apply_visualization(new_config)
        if self.window and hasattr(self.window, 'apply_osd_config'):
            self.window.apply_osd_config(new_config)
        print("[MIC-OSD] Reloaded presentation config", flush=True)
        return True

    def _show(self):
        """Show the OSD and start audio monitoring."""
        # If already visible and audio monitoring is running, return early
        # This handles the normal case where _show() is called multiple times
        if self.visible and self.audio_monitor and self.update_timer_id:
            self._mark_visible()
            return

        # If window doesn't exist yet (race condition with signal handlers),
        # just set the visible flag and return. The window will be shown when
        # it's created in _gtk_on_activate().
        if not self.window:
            self.visible = True
            return

        self.visible = True
        self.window.set_visible(True)

        # Prefer the main process's level feed (tracks the recorded device, no
        # contention); fall back to opening the default input directly.
        if not self.audio_monitor:
            if FeedLevelSource is not None and FeedLevelSource.available(MIC_OSD_LEVEL_FEED_FILE):
                self.audio_monitor = FeedLevelSource(MIC_OSD_LEVEL_FEED_FILE)
            else:
                self.audio_monitor = AudioMonitor(samplerate=44100, blocksize=1024)

        try:
            self.audio_monitor.start()
        except RuntimeError as e:
            # Audio monitoring failed (e.g., mic unavailable)
            # Keep window visible (show flat line) — the main hyprwhspr process
            # already verified audio before signaling us to show.
            print(f"[MIC-OSD] Audio monitoring unavailable, showing without waveform: {e}", flush=True)
            self.audio_monitor = None

        # Start update timer (60 FPS)
        if not self.update_timer_id:
            self.update_timer_id = GLib.timeout_add(16, self._update)

        # Start state file polling (100ms interval)
        if not self._state_poll_timer_id:
            self._state_poll_timer_id = GLib.timeout_add(100, self._poll_state_file)

        # The service-owned daemon has a stuck-window safety timeout. A
        # foreground standalone OSD is an intentional preview and stays visible
        # until Ctrl-C, which makes iterative theme/config tuning practical.
        if self._auto_hide_timeout_id:
            GLib.source_remove(self._auto_hide_timeout_id)
            self._auto_hide_timeout_id = None
        if self.daemon:
            self._auto_hide_timeout_id = GLib.timeout_add_seconds(
                30,
                self._auto_hide_callback,
            )

        # A PID alone only proves that a process exists. This acknowledgement
        # proves that its GLib loop handled the show request and requested a
        # window mapping, allowing the runner to replace a wedged daemon.
        self._mark_visible()

    def _mark_visible(self):
        try:
            MIC_OSD_VISIBLE_FILE.parent.mkdir(
                parents=True,
                exist_ok=True,
                mode=0o700,
            )
            MIC_OSD_VISIBLE_FILE.write_text(str(os.getpid()), encoding='utf-8')
            MIC_OSD_VISIBLE_FILE.chmod(0o600)
        except Exception as error:
            print(f"[MIC-OSD] Failed to acknowledge show request: {error}", flush=True)

    @staticmethod
    def _clear_visible_marker():
        try:
            MIC_OSD_VISIBLE_FILE.unlink(missing_ok=True)
        except Exception:
            pass

    def _hide(self):
        """Hide the OSD and stop audio monitoring."""
        self._clear_preview_file()
        self._clear_visible_marker()

        if not self.visible:
            return
        
        # If window doesn't exist yet (race condition with signal handlers),
        # just set the visible flag and return.
        if not self.window:
            self.visible = False
            return
        
        try:
            self.visible = False
            if hasattr(self.window, 'set_preview_text'):
                self.window.set_preview_text("")
            self._last_preview_text = None
            self.window.set_visible(False)
            
            # Stop update timer
            if self.update_timer_id:
                GLib.source_remove(self.update_timer_id)
                self.update_timer_id = None

            # Stop state polling timer
            if self._state_poll_timer_id:
                GLib.source_remove(self._state_poll_timer_id)
                self._state_poll_timer_id = None

            # Cancel auto-hide timeout
            if self._auto_hide_timeout_id:
                GLib.source_remove(self._auto_hide_timeout_id)
                self._auto_hide_timeout_id = None
            
            # Stop audio monitoring
            if self.audio_monitor:
                self.audio_monitor.stop()
                self.audio_monitor = None
        except Exception as e:
            # Ensure window is hidden even if exceptions occur
            print(f"[MIC-OSD] Error in _hide(): {e}", flush=True)
            self.visible = False
            if self.window:
                try:
                    self.window.set_visible(False)
                except Exception:
                    pass
            # Clean up timers on error
            if self.update_timer_id:
                try:
                    GLib.source_remove(self.update_timer_id)
                except Exception:
                    pass
                self.update_timer_id = None
            if self._state_poll_timer_id:
                try:
                    GLib.source_remove(self._state_poll_timer_id)
                except Exception:
                    pass
                self._state_poll_timer_id = None
            if self._auto_hide_timeout_id:
                try:
                    GLib.source_remove(self._auto_hide_timeout_id)
                except Exception:
                    pass
                self._auto_hide_timeout_id = None
            # Clean up audio monitor on error
            if self.audio_monitor:
                try:
                    self.audio_monitor.stop()
                except Exception:
                    pass
                self.audio_monitor = None

    def _clear_preview_file(self):
        try:
            if TRANSCRIPT_PREVIEW_FILE.exists():
                TRANSCRIPT_PREVIEW_FILE.unlink()
        except Exception:
            pass
    
    def _update(self):
        """Update visualization with current audio data."""
        if self.audio_monitor and self.window and self.visible:
            level = self.audio_monitor.get_level()
            samples = self.audio_monitor.get_samples()
            self.window.update(level, samples)
        return True  # Continue timer

    def _poll_state_file(self):
        """Poll the visualizer state file and update visualization state."""
        try:
            if VISUALIZER_STATE_FILE.exists():
                with open(VISUALIZER_STATE_FILE, 'r') as f:
                    state = f.read().strip()
                    if state and state != self._last_visualizer_state:
                        self._last_visualizer_state = state
                        if self.window and hasattr(self.window, 'set_visualizer_state'):
                            self.window.set_visualizer_state(state)
                        # Update visualization state if it has the set_state method
                        if hasattr(self.visualization, 'set_state'):
                            self.visualization.set_state(state)
            else:
                # No state file means default to recording state
                if self._last_visualizer_state != 'recording':
                    self._last_visualizer_state = 'recording'
                    if self.window and hasattr(self.window, 'set_visualizer_state'):
                        self.window.set_visualizer_state('recording')
                    if hasattr(self.visualization, 'set_state'):
                        self.visualization.set_state('recording')

            preview = ""
            if TRANSCRIPT_PREVIEW_FILE.exists():
                preview = TRANSCRIPT_PREVIEW_FILE.read_text(encoding='utf-8').rstrip('\r\n')
            if preview != self._last_preview_text:
                self._last_preview_text = preview
                if self.window and hasattr(self.window, 'set_preview_text'):
                    self.window.set_preview_text(preview)
        except Exception:
            pass  # Ignore file read errors
        return True  # Continue polling

    def _auto_hide_callback(self):
        """Auto-hide callback triggered after 30 seconds of visibility."""
        if not self.visible:
            self._auto_hide_timeout_id = None
            return False  # Don't repeat
        
        # Check if recording is still active before hiding
        # This prevents hiding during normal long recordings
        recording_active = False
        try:
            if RECORDING_STATUS_FILE.exists():
                with open(RECORDING_STATUS_FILE, 'r') as f:
                    status = f.read().strip()
                    if status == 'true':
                        recording_active = True
        except Exception:
            # File read error - assume not recording, allow hide
            pass
        
        if recording_active:
            # Recording is still active - reset timeout instead of hiding
            print("[MIC-OSD] Recording active - resetting auto-hide timeout", flush=True)
            self._auto_hide_timeout_id = GLib.timeout_add_seconds(30, self._auto_hide_callback)
            return False  # Don't repeat (new timeout already set)
        else:
            # Recording not active - window is stuck, hide it
            print("[MIC-OSD] Auto-hiding window after 30 second timeout (recording not active)", flush=True)
            self._hide()
            self._auto_hide_timeout_id = None
            return False  # Don't repeat
    
    def _on_theme_changed(self):
        """Called when the Omarchy theme changes."""
        # Force a redraw to pick up new colors
        if self.window:
            self.window.drawing_area.queue_draw()
    
    def stop(self):
        """Stop the OSD completely."""
        if self.app:
            self.app.quit()
        elif self.main_loop:
            self.main_loop.quit()
        else:
            # Neither app nor main_loop exists yet (early stop request)
            # Set flag and call cleanup directly
            self._should_stop = True
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources."""
        self._clear_preview_file()
        self._clear_visible_marker()

        if self.update_timer_id:
            GLib.source_remove(self.update_timer_id)
            self.update_timer_id = None

        if self._state_poll_timer_id:
            GLib.source_remove(self._state_poll_timer_id)
            self._state_poll_timer_id = None

        if self._config_poll_timer_id:
            GLib.source_remove(self._config_poll_timer_id)
            self._config_poll_timer_id = None

        if self._auto_hide_timeout_id:
            GLib.source_remove(self._auto_hide_timeout_id)
            self._auto_hide_timeout_id = None

        if self.audio_monitor:
            self.audio_monitor.stop()
            self.audio_monitor = None

        if self.window:
            if self.app:
                self.app.remove_window(self.window)
            self.window = None

        if self.theme_watcher:
            self.theme_watcher.stop()
            self.theme_watcher = None


# Global instance for signal handlers
_app = None


def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT - quit."""
    if _app:
        _app.stop()


def _sigusr1_handler(signum=None, frame=None):
    """Fallback Python signal handler when GLib Unix sources are unavailable."""
    if _app:
        GLib.idle_add(_app._show)


def _sigusr2_handler(signum=None, frame=None):
    """Fallback Python signal handler when GLib Unix sources are unavailable."""
    if _app:
        GLib.idle_add(_app._hide)


def _glib_sigusr1_handler():
    """Handle show requests in the GLib loop rather than a Python C callback."""
    if _app:
        _app._show()
    return True


def _glib_sigusr2_handler():
    """Handle hide requests in the GLib loop rather than a Python C callback."""
    if _app:
        _app._hide()
    return True


def main():
    """Entry point."""
    global _app
    
    import argparse
    parser = argparse.ArgumentParser(
        prog="mic-osd",
        description="Show microphone input visualization overlay"
    )
    parser.add_argument(
        "-v", "--viz",
        choices=["waveform", "vu_meter", "pill", "spectrum"],
        default="waveform",
        help="Visualization type (default: waveform)"
    )
    parser.add_argument(
        "-w", "--width",
        type=int,
        default=None,
        help="Window width (default: style-specific)"
    )
    parser.add_argument(
        "-H", "--height",
        type=int,
        default=None,
        help="Window height (default: style-specific)"
    )
    parser.add_argument(
        "-d", "--daemon",
        action="store_true",
        help="Run as daemon (start hidden, show on SIGUSR1, hide on SIGUSR2)"
    )
    args = parser.parse_args()

    if _MIC_OSD_IMPORT_ERROR is not None:
        print(f"[MIC-OSD] Unavailable: {_MIC_OSD_IMPORT_ERROR}", file=sys.stderr, flush=True)
        return 1
    
    # Set up termination handlers first. The runner ignores USR1/USR2 during
    # imports so an early hotkey cannot kill a daemon before these are ready.
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    _app = MicOSD(
        visualization=args.viz,
        width=args.width,
        height=args.height,
        daemon=args.daemon
    )

    # Python signal callbacks can remain pending while the interpreter is
    # blocked inside GLib.MainLoop.run(). Native GLib Unix signal sources wake
    # that loop directly and make repeated daemon show/hide requests reliable.
    if hasattr(GLib, 'unix_signal_add'):
        GLib.unix_signal_add(
            GLib.PRIORITY_DEFAULT,
            signal.SIGUSR1,
            _glib_sigusr1_handler,
        )
        GLib.unix_signal_add(
            GLib.PRIORITY_DEFAULT,
            signal.SIGUSR2,
            _glib_sigusr2_handler,
        )
    else:
        signal.signal(signal.SIGUSR1, _sigusr1_handler)
        signal.signal(signal.SIGUSR2, _sigusr2_handler)

    _app.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
