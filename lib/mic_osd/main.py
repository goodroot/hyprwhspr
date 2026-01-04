"""
mic-osd - A minimal audio visualization OSD for Wayland/Hyprland.

Shows a real-time microphone input visualization overlay.
Supports two modes:
- Standalone: runs until killed (SIGTERM/SIGINT)
- Daemon: stays running, shows on SIGUSR1, hides on SIGUSR2
"""

import sys
import signal

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, GLib

from .window import OSDWindow, load_css
from .audio import AudioMonitor
from .visualizations import VISUALIZATIONS
from .theme import ThemeWatcher


class MicOSD:
    """
    Mic-osd application with show/hide support.
    """
    
    def __init__(self, visualization="waveform", width=400, height=68, daemon=False):
        self.main_loop = None
        self.audio_monitor = None
        self.window = None
        self.update_timer_id = None
        self._auto_hide_timeout_id = None
        self.daemon = daemon
        self.visible = False
        self.theme_watcher = None
        
        # Get visualization
        viz_class = VISUALIZATIONS.get(visualization, VISUALIZATIONS["waveform"])
        self.visualization = viz_class()
        self.width = width
        self.height = height
    
    def run(self):
        """Start the OSD and run until killed."""
        # Initialize GTK
        Gtk.init()
        
        # Load CSS
        load_css()
        
        # Create window (hidden in daemon mode)
        self.window = OSDWindow(self.visualization, self.width, self.height)
        
        # Start theme watcher for live theme updates
        self.theme_watcher = ThemeWatcher(on_theme_changed=self._on_theme_changed)
        self.theme_watcher.start()
        
        if self.daemon:
            # Start hidden, wait for SIGUSR1
            self.window.set_visible(False)
        else:
            # Show immediately
            self._show()
        
        # Create main loop
        self.main_loop = GLib.MainLoop()
        
        try:
            self.main_loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()
    
    def _show(self):
        """Show the OSD and start audio monitoring."""
        if self.visible:
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
            self.window.set_visible(False)
            
            # Stop update timer if it was started
            if self.update_timer_id:
                GLib.source_remove(self.update_timer_id)
                self.update_timer_id = None
            
            return  # Exit early - don't start timer
        
        # Start update timer (60 FPS)
        if not self.update_timer_id:
            self.update_timer_id = GLib.timeout_add(16, self._update)
        
        # Start auto-hide timeout (30 seconds)
        if self._auto_hide_timeout_id:
            GLib.source_remove(self._auto_hide_timeout_id)
        self._auto_hide_timeout_id = GLib.timeout_add_seconds(30, self._auto_hide_callback)
    
    def _hide(self):
        """Hide the OSD and stop audio monitoring."""
        if not self.visible:
            return
        
        try:
            self.visible = False
            self.window.set_visible(False)
            
            # Stop update timer
            if self.update_timer_id:
                GLib.source_remove(self.update_timer_id)
                self.update_timer_id = None
            
            # Cancel auto-hide timeout
            if self._auto_hide_timeout_id:
                GLib.source_remove(self._auto_hide_timeout_id)
                self._auto_hide_timeout_id = None
            
            # Stop audio monitoring
            if self.audio_monitor:
                self.audio_monitor.stop()
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
            if self._auto_hide_timeout_id:
                try:
                    GLib.source_remove(self._auto_hide_timeout_id)
                except Exception:
                    pass
                self._auto_hide_timeout_id = None
    
    def _update(self):
        """Update visualization with current audio data."""
        if self.audio_monitor and self.window and self.visible:
            level = self.audio_monitor.get_level()
            samples = self.audio_monitor.get_samples()
            self.window.update(level, samples)
        return True  # Continue timer
    
    def _auto_hide_callback(self):
        """Auto-hide callback triggered after 30 seconds of visibility."""
        if self.visible:
            print("[MIC-OSD] Auto-hiding window after 30 second timeout", flush=True)
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
        if self.main_loop:
            self.main_loop.quit()
    
    def _cleanup(self):
        """Clean up resources."""
        if self.update_timer_id:
            GLib.source_remove(self.update_timer_id)
            self.update_timer_id = None
        
        if self._auto_hide_timeout_id:
            GLib.source_remove(self._auto_hide_timeout_id)
            self._auto_hide_timeout_id = None
        
        if self.audio_monitor:
            self.audio_monitor.stop()
            self.audio_monitor = None
        
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
