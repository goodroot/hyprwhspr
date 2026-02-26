"""
GTK4 Layer Shell window for tts-osd.

Creates an overlay window at the bottom center of the screen
for displaying TTS "Reading..." visualization.
"""

import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Gdk', '4.0')

try:
    gi.require_version('Gtk4LayerShell', '1.0')
    LAYER_SHELL_AVAILABLE = True
except ValueError:
    LAYER_SHELL_AVAILABLE = False

from gi.repository import Gtk, Gdk, GLib

if LAYER_SHELL_AVAILABLE:
    from gi.repository import Gtk4LayerShell


class TTSOSDWindow(Gtk.Window):
    """
    An overlay window for displaying TTS "Reading..." visualization.

    Uses gtk4-layer-shell to create a Wayland layer surface
    that appears above all windows at the bottom of the screen.
    """

    def __init__(self, visualization, width=300, height=60):
        super().__init__()

        self.visualization = visualization
        self._width = width
        self._height = height

        self._setup_layer_shell()
        self._setup_window()
        self._setup_drawing_area()

    def _setup_layer_shell(self):
        """Configure layer shell for overlay behavior."""
        if not LAYER_SHELL_AVAILABLE:
            return

        Gtk4LayerShell.init_for_window(self)
        Gtk4LayerShell.set_namespace(self, "tts-osd")
        Gtk4LayerShell.set_layer(self, Gtk4LayerShell.Layer.OVERLAY)
        Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.BOTTOM, True)
        Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.LEFT, False)
        Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.RIGHT, False)
        Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.TOP, False)
        Gtk4LayerShell.set_margin(self, Gtk4LayerShell.Edge.BOTTOM, 130)
        Gtk4LayerShell.set_exclusive_zone(self, -1)
        Gtk4LayerShell.set_keyboard_mode(self, Gtk4LayerShell.KeyboardMode.NONE)

    def _setup_window(self):
        """Configure basic window properties."""
        self.set_decorated(False)
        self.set_resizable(False)
        self.set_default_size(self._width, self._height)
        self.add_css_class('tts-osd-window')

    def _setup_drawing_area(self):
        """Set up the Cairo drawing area."""
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_content_width(self._width)
        self.drawing_area.set_content_height(self._height)
        self.drawing_area.set_draw_func(self._on_draw)
        self.set_child(self.drawing_area)

    def _on_draw(self, area, cr, width, height):
        """Called when the drawing area needs to be redrawn."""
        self.visualization.draw_background(cr, width, height)
        self.visualization.draw(cr, width, height)

    def update(self):
        """Update the visualization (state-driven, no audio input)."""
        self.visualization.update(0.0, None)
        self.drawing_area.queue_draw()

    def set_visualization(self, visualization):
        """Change the visualization type."""
        self.visualization = visualization
        self.drawing_area.queue_draw()


def load_css(css_path=None):
    """Load CSS styling for the TTS OSD."""
    css_provider = Gtk.CssProvider()
    default_css = """
    .tts-osd-window {
        background-color: transparent;
    }
    """
    if css_path:
        try:
            css_provider.load_from_path(css_path)
        except GLib.Error:
            css_provider.load_from_string(default_css)
    else:
        css_provider.load_from_string(default_css)

    Gtk.StyleContext.add_provider_for_display(
        Gdk.Display.get_default(),
        css_provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )
