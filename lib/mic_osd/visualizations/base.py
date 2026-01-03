"""
Base visualization class for mic-osd.
"""

from abc import ABC, abstractmethod
import cairo
import numpy as np

from ..theme import theme


class BaseVisualization(ABC):
    """
    Abstract base class for audio visualizations.
    
    Subclasses must implement the draw() method to render
    the visualization using Cairo.
    """
    
    def __init__(self):
        self.width = 300
        self.height = 60
        self.audio_level = 0.0
        self.audio_samples = np.zeros(1024)
    
    @property
    def background_color(self):
        """Get background color from theme."""
        bg = theme.background
        if len(bg) == 3:
            return (*bg, 0.95)
        return bg
    
    def update(self, level: float, samples: np.ndarray = None):
        """
        Update visualization with new audio data.
        
        Args:
            level: Audio level (0.0 to 1.0)
            samples: Raw audio samples (optional, for waveform)
        """
        self.audio_level = max(0.0, min(1.0, level))
        if samples is not None:
            self.audio_samples = samples
    
    @abstractmethod
    def draw(self, cr: cairo.Context, width: int, height: int):
        """
        Draw the visualization.
        
        Args:
            cr: Cairo context to draw on
            width: Available width in pixels
            height: Available height in pixels
        """
        pass
    
    def draw_background(self, cr: cairo.Context, width: int, height: int):
        """Draw common background with border."""
        # Fill background
        cr.set_source_rgba(*self.background_color)
        cr.rectangle(0, 0, width, height)
        cr.fill()
        
        # Draw border
        border_color = theme.border
        if border_color:
            if len(border_color) == 3:
                cr.set_source_rgb(*border_color)
            else:
                cr.set_source_rgba(*border_color)
            cr.set_line_width(2)
            cr.rectangle(1, 1, width - 2, height - 2)
            cr.stroke()
