"""
tts-osd - TTS "Reading..." overlay for Wayland/Hyprland.

Shows overlay during TTS synthesis and playback.
Daemon mode: start hidden, show on SIGUSR1, hide on SIGUSR2.
"""

import sys
import signal
import os
from pathlib import Path

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, GLib

from .window import TTSOSDWindow, load_css
from .visualization import SpeakingVisualization

try:
    from ..src.paths import TTS_OSD_STATE_FILE
except ImportError:
    from src.paths import TTS_OSD_STATE_FILE


def is_gnome():
    desktop = os.environ.get('XDG_CURRENT_DESKTOP', '').lower()
    return 'gnome' in desktop


class TTSOSD:
    """TTS OSD application with show/hide support."""

    def __init__(self, width=400, height=68, daemon=False):
        self.main_loop = None
        self.app = None
        self.window = None
        self.update_timer_id = None
        self._auto_hide_timeout_id = None
        self._state_poll_timer_id = None
        self._last_state = None
        self.daemon = daemon
        self.visible = False
        self._should_stop = False

        self.visualization = SpeakingVisualization()
        self.width = width
        self.height = height

    def run(self):
        if is_gnome():
            self._run_with_gtk_application()
        else:
            self._run_with_main_loop()

    def _run_with_gtk_application(self):
        self.app = Gtk.Application(application_id="com.hyprwhspr.tts-osd")
        self.app.connect('activate', self._gtk_on_activate)
        self.app.connect('shutdown', lambda _: self._cleanup())
        if self._should_stop:
            self._cleanup()
            return
        try:
            self.app.run(None)
        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()

    def _gtk_on_activate(self, app):
        if self.window:
            if self.update_timer_id:
                GLib.source_remove(self.update_timer_id)
                self.update_timer_id = None
            if self._state_poll_timer_id:
                GLib.source_remove(self._state_poll_timer_id)
                self._state_poll_timer_id = None
            if self._auto_hide_timeout_id:
                GLib.source_remove(self._auto_hide_timeout_id)
                self._auto_hide_timeout_id = None
            app.remove_window(self.window)
            self.window = None

        load_css()
        self.window = TTSOSDWindow(
            self.visualization, self.width, self.height,
            on_close=None,  # No close button; use shortcut to cancel
        )
        app.add_window(self.window)
        self._initial_visibility()

    def _run_with_main_loop(self):
        Gtk.init()
        load_css()
        self.window = TTSOSDWindow(
            self.visualization, self.width, self.height,
            on_close=None,  # No close button; use shortcut to cancel
        )
        self._initial_visibility()

        if self._should_stop:
            return

        self.main_loop = GLib.MainLoop()
        try:
            self.main_loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()

    def _initial_visibility(self):
        if self.visible:
            self._show()
        elif self.daemon:
            self.window.set_visible(False)
        else:
            self._show()

    def _show(self):
        if self.visible and self.update_timer_id:
            return

        if not self.window:
            self.visible = True
            return

        self.visible = True
        self.window.set_visible(True)

        if not self.update_timer_id:
            self.update_timer_id = GLib.timeout_add(16, self._update)

        if not self._state_poll_timer_id:
            self._state_poll_timer_id = GLib.timeout_add(100, self._poll_state_file)

        if self._auto_hide_timeout_id:
            GLib.source_remove(self._auto_hide_timeout_id)
        self._auto_hide_timeout_id = GLib.timeout_add_seconds(30, self._auto_hide_callback)

    def _hide(self):
        if not self.visible:
            return

        if not self.window:
            self.visible = False
            return

        try:
            self.visible = False
            self.window.set_visible(False)

            if self.update_timer_id:
                GLib.source_remove(self.update_timer_id)
                self.update_timer_id = None
            if self._state_poll_timer_id:
                GLib.source_remove(self._state_poll_timer_id)
                self._state_poll_timer_id = None
            if self._auto_hide_timeout_id:
                GLib.source_remove(self._auto_hide_timeout_id)
                self._auto_hide_timeout_id = None
        except Exception as e:
            print(f"[TTS-OSD] Error in _hide(): {e}", flush=True)
            self.visible = False
            if self.window:
                try:
                    self.window.set_visible(False)
                except Exception:
                    pass
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

    def _update(self):
        if self.window and self.visible:
            self.window.update()
        return True

    def _poll_state_file(self):
        try:
            if TTS_OSD_STATE_FILE.exists():
                with open(TTS_OSD_STATE_FILE, 'r') as f:
                    state = f.read().strip()
                    if state and state != self._last_state:
                        self._last_state = state
                        self.visualization.set_state(state)
            else:
                if self._last_state != 'speaking':
                    self._last_state = 'speaking'
                    self.visualization.set_state('speaking')
        except Exception:
            pass
        return True

    def _auto_hide_callback(self):
        if not self.visible:
            self._auto_hide_timeout_id = None
            return False
        self._hide()
        self._auto_hide_timeout_id = None
        return False

    def stop(self):
        if self.app:
            self.app.quit()
        elif self.main_loop:
            self.main_loop.quit()
        else:
            self._should_stop = True
            self._cleanup()

    def _cleanup(self):
        if self.update_timer_id:
            GLib.source_remove(self.update_timer_id)
            self.update_timer_id = None
        if self._state_poll_timer_id:
            GLib.source_remove(self._state_poll_timer_id)
            self._state_poll_timer_id = None
        if self._auto_hide_timeout_id:
            GLib.source_remove(self._auto_hide_timeout_id)
            self._auto_hide_timeout_id = None
        if self.window:
            if self.app:
                self.app.remove_window(self.window)
            self.window = None


_app = None


def _signal_handler(signum, frame):
    if _app:
        _app.stop()


def _sigusr1_handler(signum, frame):
    if _app:
        GLib.idle_add(_app._show)


def _sigusr2_handler(signum, frame):
    if _app:
        GLib.idle_add(_app._hide)


def main():
    global _app

    import argparse
    parser = argparse.ArgumentParser(
        prog="tts-osd",
        description="Show TTS 'Reading...' overlay"
    )
    parser.add_argument("-w", "--width", type=int, default=400)
    parser.add_argument("-H", "--height", type=int, default=68)
    parser.add_argument("-d", "--daemon", action="store_true",
                        help="Run as daemon (show on SIGUSR1, hide on SIGUSR2)")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGUSR1, _sigusr1_handler)
    signal.signal(signal.SIGUSR2, _sigusr2_handler)

    _app = TTSOSD(width=args.width, height=args.height, daemon=args.daemon)
    _app.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
