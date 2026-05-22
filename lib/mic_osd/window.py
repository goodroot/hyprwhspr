"""
GTK4 Layer Shell window for mic-osd.

Creates an overlay window at the bottom center of the screen
for displaying audio visualizations.
"""

from __future__ import annotations

import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Gdk', '4.0')
import cairo

try:
    gi.require_version('Gtk4LayerShell', '1.0')
    LAYER_SHELL_AVAILABLE = True
except ValueError:
    LAYER_SHELL_AVAILABLE = False

from gi.repository import Gtk, Gdk, GLib
from .theme import theme

if LAYER_SHELL_AVAILABLE:
    from gi.repository import Gtk4LayerShell


class OSDWindow(Gtk.Window):
    """
    An overlay window for displaying audio visualizations.
    
    Uses gtk4-layer-shell to create a Wayland layer surface
    that appears above all windows at the bottom of the screen.
    """

    PREVIEW_WORD_LIMIT = 14
    PREVIEW_TIMER_RESERVE = 58
    
    def __init__(self, visualization, width=300, height=60):
        """
        Initialize the OSD window.
        
        Args:
            visualization: A BaseVisualization instance
            width: Window width in pixels
            height: Window height in pixels
        """
        super().__init__()
        
        self.visualization = visualization
        self._width = width
        self._height = height
        self._preview_text = ""
        
        # Layer shell MUST be initialized immediately after window creation
        # and BEFORE any other window configuration
        self._setup_layer_shell()
        self._setup_window()
        self._setup_drawing_area()
    
    def _setup_layer_shell(self):
        """Configure layer shell for overlay behavior."""
        if not LAYER_SHELL_AVAILABLE:
            return
        
        # Initialize layer shell - MUST be called before window is realized
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
        
        # Make window transparent
        self.add_css_class('mic-osd-window')
    
    def _setup_drawing_area(self):
        """Set up the Cairo drawing area."""
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_content_width(self._width)
        self.drawing_area.set_content_height(self._height)
        self.drawing_area.set_draw_func(self._on_draw)
        self.set_child(self.drawing_area)
    
    def _on_draw(self, area, cr, width, height):
        """
        Called when the drawing area needs to be redrawn.
        
        Args:
            area: The DrawingArea widget
            cr: Cairo context
            width: Available width
            height: Available height
        """
        # Draw background
        self.visualization.draw_background(cr, width, height)
        
        # Draw the visualization
        self.visualization.draw(cr, width, height)

        self._draw_preview_text(cr, width, height)
    
    def update(self, level: float, samples=None):
        """
        Update the visualization with new audio data.
        
        Args:
            level: Audio level (0.0 to 1.0)
            samples: Raw audio samples (optional)
        """
        self.visualization.update(level, samples)
        self.drawing_area.queue_draw()

    def set_preview_text(self, text: str):
        """Set compact transcript preview text."""
        self._preview_text = (text or "").strip()
        self.drawing_area.queue_draw()
    
    def set_visualization(self, visualization):
        """Change the visualization type."""
        self.visualization = visualization
        self.drawing_area.queue_draw()

    def _draw_preview_text(self, cr: cairo.Context, width: int, height: int):
        if not self._preview_text:
            return

        padding = 14
        max_width = max(0, width - padding * 2 - self.PREVIEW_TIMER_RESERVE)

        cr.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(12)
        text = self._ellipsize(cr, self._recent_preview_text(self._preview_text), max_width)
        if not text:
            return

        extents = cr.text_extents(text)
        text_width = self._text_extent(extents, 'width', 2)
        text_height = self._text_extent(extents, 'height', 3)

        y = height - 10
        bg_padding_x = 7
        bg_padding_y = 4
        bg_x = padding - bg_padding_x
        bg_y = y - text_height - bg_padding_y - 1
        bg_w = min(max_width + bg_padding_x * 2, text_width + bg_padding_x * 2)
        bg_h = text_height + bg_padding_y * 2 + 2

        bg = theme.background
        cr.set_source_rgba(bg[0], bg[1], bg[2], 0.88)
        cr.rectangle(bg_x, bg_y, bg_w, bg_h)
        cr.fill()

        text_color = theme.text
        cr.set_source_rgba(text_color[0], text_color[1], text_color[2], 0.96)
        cr.move_to(padding, y)
        cr.show_text(text)

    @staticmethod
    def _text_extent(extents, field: str, index: int) -> float:
        if hasattr(extents, field):
            return getattr(extents, field)
        return extents[index]

    def _text_width(self, cr: cairo.Context, text: str) -> float:
        return self._text_extent(cr.text_extents(text), 'width', 2)

    def _text_height(self, cr: cairo.Context, text: str) -> float:
        return self._text_extent(cr.text_extents(text), 'height', 3)

    def _recent_preview_text(self, text: str) -> str:
        words = text.split()
        if len(words) <= self.PREVIEW_WORD_LIMIT:
            return " ".join(words)
        return "... " + " ".join(words[-self.PREVIEW_WORD_LIMIT:])

    def _ellipsize(self, cr: cairo.Context, text: str, max_width: float) -> str:
        if self._text_width(cr, text) <= max_width:
            return text

        prefix = "... "
        if text.startswith(prefix):
            text = text[len(prefix):]

        available = max_width - self._text_width(cr, prefix)
        if available <= 0:
            return ""

        low = 0
        high = len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if self._text_width(cr, text[-mid:]) <= available:
                low = mid
            else:
                high = mid - 1

        truncated = text[-low:].lstrip()
        return prefix + truncated if truncated else prefix
    
    def make_click_through(self):
        """
        Make the window click-through (input passes to windows below).
        
        This needs to be called after the window is realized.
        """
        # For layer shell windows, we just need to not set keyboard mode
        # The default is already non-interactive
        pass


def load_css(css_path=None):
    """
    Load CSS styling for the OSD.
    
    Args:
        css_path: Path to CSS file (optional)
    """
    css_provider = Gtk.CssProvider()
    
    default_css = """
    .mic-osd-window {
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
