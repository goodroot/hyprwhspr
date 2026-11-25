"""X11 overlay: a GTK3 window that never takes focus or clicks."""

from __future__ import annotations

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gdk', '3.0')
import cairo

from gi.repository import Gtk, Gdk, GLib
from .backend import bottom_center
from .render import OSDRenderer


CSS = """
window.mic-osd-x11,
.mic-osd-x11 {
    background-color: transparent;
    background-image: none;
    box-shadow: none;
}
"""

_provider = None


def load_css(css_path=None):
    """Keep the window clear around the pill, above any user stylesheet."""
    global _provider

    screen = Gdk.Screen.get_default()
    if screen is None:
        return

    if _provider is None:
        _provider = Gtk.CssProvider()
        try:
            _provider.load_from_data(CSS.encode('utf-8'))
        except GLib.Error:
            return
        Gtk.StyleContext.add_provider_for_screen(
            screen, _provider, Gtk.STYLE_PROVIDER_PRIORITY_USER + 1)

    if css_path:
        custom_provider = Gtk.CssProvider()
        try:
            custom_provider.load_from_path(css_path)
        except GLib.Error:
            return
        Gtk.StyleContext.add_provider_for_screen(
            screen, custom_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)


class OSDWindow(OSDRenderer, Gtk.Window):
    BOTTOM_MARGIN = 130  # matches the layer-shell overlay

    def __init__(self, visualization, width=300, height=60, transcript_config=None):
        Gtk.Window.__init__(self, type=Gtk.WindowType.TOPLEVEL)
        self._setup_renderer(visualization, width, height, transcript_config)
        self._setup_window()
        self._setup_drawing_area()

    def _setup_window(self):
        self.set_decorated(False)
        self.set_resizable(False)
        self.set_default_size(self._width, self._height)
        # The paste keystroke must reach the focused app.
        self.set_accept_focus(False)
        self.set_focus_on_map(False)
        self.set_keep_above(True)
        self.set_skip_taskbar_hint(True)
        self.set_skip_pager_hint(True)
        self.set_type_hint(Gdk.WindowTypeHint.NOTIFICATION)
        self.set_app_paintable(True)
        self.get_style_context().add_class('mic-osd-x11')
        self._apply_rgba_visual()

        self.connect('screen-changed', self._on_screen_changed)
        self.connect('realize', self._on_realize)
        self.connect('map-event', self._on_map_event)

    def _setup_drawing_area(self):
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_size_request(self._width, self._height)
        self.drawing_area.connect('draw', self._on_draw)
        self.drawing_area.show()
        self.add(self.drawing_area)

    def _apply_rgba_visual(self):
        screen = self.get_screen()
        if screen is None:
            return
        visual = screen.get_rgba_visual()
        if visual is not None:
            self.set_visual(visual)

    def _on_screen_changed(self, widget, previous_screen):
        self._apply_rgba_visual()

    def _on_realize(self, widget):
        self._make_click_through()
        self._reposition()

    def _on_map_event(self, widget, event):
        # The WM may have placed it elsewhere.
        self._reposition()
        self._make_click_through()
        return False

    def _make_click_through(self):
        gdk_window = self.get_window()
        if gdk_window is None:
            return
        try:
            gdk_window.set_pass_through(True)
            return
        except (AttributeError, TypeError):
            pass
        try:
            gdk_window.input_shape_combine_region(cairo.Region())
        except Exception:
            pass

    def _reposition(self):
        screen = self.get_screen()
        if screen is None:
            return
        try:
            monitor = screen.get_primary_monitor()
            geometry = screen.get_monitor_geometry(monitor)
        except Exception:
            return
        x, y = bottom_center(geometry, self._width, self._height, self.BOTTOM_MARGIN)
        self.move(x, y)

    def _queue_redraw(self):
        self.drawing_area.queue_draw()

    def _on_draw(self, widget, cr):
        allocation = widget.get_allocation()
        cr.save()
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.set_source_rgba(0.0, 0.0, 0.0, 0.0)
        cr.paint()
        cr.restore()
        self.draw_osd(cr, allocation.width, allocation.height)
        return False
