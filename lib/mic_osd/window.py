"""
GTK4 Layer Shell window for mic-osd.

Creates an overlay window at the bottom center of the screen
for displaying audio visualizations.
"""

from __future__ import annotations

import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Gdk', '4.0')

try:
    gi.require_version('Gtk4LayerShell', '1.0')
    LAYER_SHELL_AVAILABLE = True
except ValueError:
    LAYER_SHELL_AVAILABLE = False

from gi.repository import Gtk, Gdk, GLib
from .render import OSDRenderer

if LAYER_SHELL_AVAILABLE:
    from gi.repository import Gtk4LayerShell


class OSDWindow(OSDRenderer, Gtk.Window):
    """
    An overlay window for displaying audio visualizations.
    
    Uses gtk4-layer-shell to create a Wayland layer surface
    that appears above all windows at the bottom of the screen.
    """

    def __init__(
        self,
        visualization,
        width=300,
        height=60,
        transcript_config=None,
    ):
        """
        Initialize the OSD window.

        Args:
            visualization: A BaseVisualization instance
            width: Window width in pixels
            height: Window height in pixels
            transcript_config: Validated PillTranscriptConfig for pill previews.
        """
        Gtk.Window.__init__(self)
        self._setup_renderer(visualization, width, height, transcript_config)

        # Layer shell MUST be initialized immediately after window creation
        # and BEFORE any other window configuration
        self._setup_layer_shell()
        self._setup_window()
        self._setup_drawing_area()
    
    def _setup_layer_shell(self):
        """Configure layer shell behavior."""
        if not LAYER_SHELL_AVAILABLE:
            return
        
        # Initialize layer shell - MUST be initialized before other calls
        Gtk4LayerShell.init_for_window(self)
        
        # Set namespace for window rules
        Gtk4LayerShell.set_namespace(self, "mic-osd")
        
        # Put on overlay layer (above everything)
        Gtk4LayerShell.set_layer(self, Gtk4LayerShell.Layer.OVERLAY)
        
        # Anchor to bottom only (centers horizontally)
        Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.BOTTOM, True)
        Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.LEFT, False)
        Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.RIGHT, False)
        Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.TOP, False)
        
        # Margin from bottom
        Gtk4LayerShell.set_margin(self, Gtk4LayerShell.Edge.BOTTOM, 130)
        
        # Don't reserve exclusive space
        Gtk4LayerShell.set_exclusive_zone(self, -1)
        
        # No keyboard input
        Gtk4LayerShell.set_keyboard_mode(self, Gtk4LayerShell.KeyboardMode.NONE)
    
    def _setup_window(self):
        """Configure basic window properties."""
        self.set_decorated(False)
        self.set_resizable(False)
        self.set_default_size(self._width, self._height)
        
        # Make window transparent. GTK gives every window the `background`
        # style class, which is what paints the theme's opaque window colour
        # across the whole surface; dropping it stops class-scoped theme rules
        # from filling the area around the pill.
        self.remove_css_class('background')
        self.add_css_class('mic-osd-window')
    
    def _setup_drawing_area(self):
        """Set up the Cairo drawing area."""
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_content_width(self._width)
        self.drawing_area.set_content_height(self._height)
        self.drawing_area.set_draw_func(self._on_draw)
        self.set_child(self.drawing_area)

    def _queue_redraw(self):
        self.drawing_area.queue_draw()
    
    def _on_draw(self, area, cr, width, height):
        """
        Called when the drawing area needs to be redrawn.
        
        Args:
            area: The DrawingArea widget
            cr: Cairo context
            width: Available width
            height: Available height
        """
        self.draw_osd(cr, width, height)


TRANSPARENCY_CSS = """
window.mic-osd-window,
.mic-osd-window {
    background-color: transparent;
    background-image: none;
    box-shadow: none;
}
"""

# A ~/.config/gtk-4.0/gtk.css sets `window { background-color: ... }` for many
# themed desktops, and GTK loads that at PRIORITY_USER. At the application
# priority our transparency rule loses to it, and the theme's window colour is
# painted as an opaque slab behind the pill (issue #248). Outrank the user
# stylesheet: the OSD is a shaped overlay, so its surface must stay clear
# whatever the desktop theme says about windows.
TRANSPARENCY_PRIORITY = Gtk.STYLE_PROVIDER_PRIORITY_USER + 1

_transparency_provider = None


def load_css(css_path=None):
    """
    Load CSS styling for the OSD.

    The transparency rule is always installed above the user stylesheet. An
    optional css_path uses application priority for pill/content cosmetics.
    It cannot override the higher-priority transparent outer-window surface
    (background, background-image or box-shadow).

    Args:
        css_path: Path to CSS file (optional)
    """
    global _transparency_provider

    display = Gdk.Display.get_default()
    if display is None:
        return

    # Installed once per process; load_css runs again on every GTK activation.
    if _transparency_provider is None:
        _transparency_provider = Gtk.CssProvider()
        _transparency_provider.load_from_string(TRANSPARENCY_CSS)
        Gtk.StyleContext.add_provider_for_display(
            display,
            _transparency_provider,
            TRANSPARENCY_PRIORITY,
        )

    if css_path:
        custom_provider = Gtk.CssProvider()
        try:
            custom_provider.load_from_path(css_path)
        except GLib.Error:
            return
        Gtk.StyleContext.add_provider_for_display(
            display,
            custom_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )
