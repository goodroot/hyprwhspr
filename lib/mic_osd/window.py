"""
GTK4 Layer Shell window for mic-osd.

Creates an overlay window at the bottom center of the screen
for displaying audio visualizations.
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


class OSDWindow(Gtk.Window):
    """
    An overlay window for displaying audio visualizations.
    
    Uses gtk4-layer-shell to create a Wayland layer surface
    that appears above all windows at the bottom of the screen.
    """
    
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
    
    def update(self, level: float, samples=None):
        """
        Update the visualization with new audio data.
        
        Args:
            level: Audio level (0.0 to 1.0)
            samples: Raw audio samples (optional)
        """
        self.visualization.update(level, samples)
        self.drawing_area.queue_draw()
    
    def set_visualization(self, visualization):
        """Change the visualization type."""
        self.visualization = visualization
        self.drawing_area.queue_draw()
    
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
