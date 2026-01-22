"""
mic-osd - A minimal audio visualization OSD for Wayland/Hyprland.

Shows a real-time microphone input visualization overlay.
Supports two modes:
- Standalone: runs until killed (SIGTERM/SIGINT)
- Daemon: stays running, shows on SIGUSR1, hides on SIGUSR2
"""

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

from .window import OSDWindow, load_css
from .audio import AudioMonitor
from .visualizations import VISUALIZATIONS
from .theme import ThemeWatcher

# Import paths with fallback for daemon context
try:
    from ..src.paths import RECORDING_STATUS_FILE, VISUALIZER_STATE_FILE
except ImportError:
    try:
        from src.paths import RECORDING_STATUS_FILE, VISUALIZER_STATE_FILE
    except ImportError:
        # Fallback: construct paths manually if imports fail
        home = Path.home()
        xdg_config = Path(os.environ.get('XDG_CONFIG_HOME', home / '.config'))
        RECORDING_STATUS_FILE = xdg_config / 'hyprwhspr' / 'recording_status'
        VISUALIZER_STATE_FILE = xdg_config / 'hyprwhspr' / 'visualizer_state'


class MicOSD:
    """
    Mic-osd application with show/hide support.
    """
    
    def __init__(self, visualization="waveform", width=400, height=68, daemon=False):
        self.main_loop = None
        self.app = None
        self.audio_monitor = None
        self.window = None
        self.update_timer_id = None
        self._auto_hide_timeout_id = None
        self._state_poll_timer_id = None
        self._last_visualizer_state = None
        self.daemon = daemon
        self.visible = False
        self.theme_watcher = None
        self._should_stop = False

        # Get visualization
        viz_class = VISUALIZATIONS.get(visualization, VISUALIZATIONS["waveform"])
        self.visualization = viz_class()
        self.width = width
        self.height = height

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

        self.window = OSDWindow(self.visualization, self.width, self.height)
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
        self.window = OSDWindow(self.visualization, self.width, self.height)

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

    def _show(self):
        """Show the OSD and start audio monitoring."""
        # If already visible and audio monitoring is running, return early
        # This handles the normal case where _show() is called multiple times
        if self.visible and self.audio_monitor and self.update_timer_id:
            return

        # If window doesn't exist yet (race condition with signal handlers),
        # just set the visible flag and return. The window will be shown when
        # it's created in _gtk_on_activate().
        if not self.window:
            self.visible = True
            return

        self.visible = True
        self.window.set_visible(True)

        # Start audio monitoring
        if not self.audio_monitor:
            self.audio_monitor = AudioMonitor(samplerate=44100, blocksize=1024)

        try:
            self.audio_monitor.start()
        except RuntimeError as e:
            # Audio monitoring failed (e.g., mic unavailable)
            # Hide window and reset state to prevent hanging
            print(f"[MIC-OSD] Failed to start audio monitoring: {e}", flush=True)
            self.visible = False
            if self.window:
                self.window.set_visible(False)

            # Stop update timer if it was started
            if self.update_timer_id:
                GLib.source_remove(self.update_timer_id)
                self.update_timer_id = None

            return  # Exit early - don't start timer

        # Verify that audio stream is actually receiving audio (not just zeros)
        # This prevents showing window when mic is unplugged but stream opens successfully
        import time
        verification_start = time.monotonic()
        verification_duration = 1.5  # 1.5s verification period
        max_zero_level = 1e-6  # Threshold for considering audio as zero (very small)
        audio_detected = False

        while time.monotonic() - verification_start < verification_duration:
            # Check if audio_monitor was cleaned up by a re-activation (race condition)
            if not self.audio_monitor:
                # Audio monitor was stopped by cleanup, abort verification
                self.visible = False
                if self.window:
                    self.window.set_visible(False)
                return  # Exit early - don't start timers
            
            level = self.audio_monitor.get_level()
            if level > max_zero_level:
                audio_detected = True
                break
            time.sleep(0.01)  # Check every 10ms
        
        if not audio_detected:
            # Stream is returning zeros - mic likely unavailable
            print("[MIC-OSD] Audio stream returning zeros - hiding window", flush=True)
            if self.audio_monitor:
                self.audio_monitor.stop()
                self.audio_monitor = None
            self.visible = False
            if self.window:
                self.window.set_visible(False)
            return  # Exit early - don't start timers
        
        # Start update timer (60 FPS)
        if not self.update_timer_id:
            self.update_timer_id = GLib.timeout_add(16, self._update)

        # Start state file polling (100ms interval)
        if not self._state_poll_timer_id:
            self._state_poll_timer_id = GLib.timeout_add(100, self._poll_state_file)

        # Start auto-hide timeout (30 seconds)
        if self._auto_hide_timeout_id:
            GLib.source_remove(self._auto_hide_timeout_id)
        self._auto_hide_timeout_id = GLib.timeout_add_seconds(30, self._auto_hide_callback)
    
    def _hide(self):
        """Hide the OSD and stop audio monitoring."""
        if not self.visible:
            return
        
        # If window doesn't exist yet (race condition with signal handlers),
        # just set the visible flag and return.
        if not self.window:
            self.visible = False
            return
        
        try:
            self.visible = False
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
                        # Update visualization state if it has the set_state method
                        if hasattr(self.visualization, 'set_state'):
                            self.visualization.set_state(state)
            else:
                # No state file means default to recording state
                if self._last_visualizer_state != 'recording':
                    self._last_visualizer_state = 'recording'
                    if hasattr(self.visualization, 'set_state'):
                        self.visualization.set_state('recording')
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
        if self.update_timer_id:
            GLib.source_remove(self.update_timer_id)
            self.update_timer_id = None

        if self._state_poll_timer_id:
            GLib.source_remove(self._state_poll_timer_id)
            self._state_poll_timer_id = None

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


def _sigusr1_handler(signum, frame):
    """Handle SIGUSR1 - show OSD."""
    if _app:
        GLib.idle_add(_app._show)


def _sigusr2_handler(signum, frame):
    """Handle SIGUSR2 - hide OSD."""
    if _app:
        GLib.idle_add(_app._hide)


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
        choices=["waveform", "vu_meter"],
        default="waveform",
        help="Visualization type (default: waveform)"
    )
    parser.add_argument(
        "-w", "--width",
        type=int,
        default=400,
        help="Window width (default: 400)"
    )
    parser.add_argument(
        "-H", "--height",
        type=int,
        default=68,
        help="Window height (default: 68)"
    )
    parser.add_argument(
        "-d", "--daemon",
        action="store_true",
        help="Run as daemon (start hidden, show on SIGUSR1, hide on SIGUSR2)"
    )
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGUSR1, _sigusr1_handler)
    signal.signal(signal.SIGUSR2, _sigusr2_handler)
    
    # Run
    _app = MicOSD(
        visualization=args.viz,
        width=args.width,
        height=args.height,
        daemon=args.daemon
    )
    _app.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
